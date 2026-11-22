"""Two independent P104 Batch coordinators, with bounded stages and final publication."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from .checkpointing import sha256_file
from .continuation_cloud import download_source
from .corrected_continuation import verify_corrected_source
from .experiment_runtime import run_child
from .fo2_setup import prepare as prepare_fo2
from .p104_protocol import (SOURCE_URI,POLICY_SHA256,ITERATION,TRAINER_FINGERPRINT,
    P56_AUDIT_URI,P56_POLICY_URI,P56_SHA256,P56_LIBRARY_SHA256)
from .policy_replay_artifacts import prepare,read_metadata
from .storage_transfer import run_transfer
from .strategic_audit.library import digest,validate_library
from .train import write_json


def storage(*args):
    return run_transfer(['gcloud','storage',*map(str,args)],timeout=1800)


def prepare_inputs(root,mode):
    root=Path(root)
    if mode=='regret':
        source,trainer=download_source(SOURCE_URI,root/'source',POLICY_SHA256,104)
        receipt=verify_corrected_source(source,trainer)
        state=source/'training'/f'state_{ITERATION:06d}'
        policy=source/'P104.pkl';contract=source/'training/manifest.json'
    else:
        source=root/'source';source.mkdir()
        state=source/f'state_{ITERATION:06d}'
        prepare(SOURCE_URI+f'/training/state_{ITERATION:06d}',state)
        policy=source/'P104.pkl';storage('cp',SOURCE_URI+'/P104.pkl',policy)
        trainer=contract=None;receipt=dict(source_modified=False,policy_only_download=True)
    _,metadata=read_metadata(state)
    if (sha256_file(policy)!=POLICY_SHA256 or metadata['solver']['num_iteration']!=ITERATION
            or metadata['repository_commit']!=TRAINER_FINGERPRINT):
        raise ValueError('P104 checkpoint identity or corrected trainer fingerprint mismatch')
    frozen=root/'P56.pkl';storage('cp',P56_POLICY_URI,frozen)
    encounters=root/'p56-encounters';encounters.mkdir()
    for name in ('library.json','report.json'):storage('cp',P56_AUDIT_URI+'/encounters/'+name,encounters/name)
    library=validate_library(json.loads((encounters/'library.json').read_text()))
    if sha256_file(frozen)!=P56_SHA256 or digest(library)!=P56_LIBRARY_SHA256:
        raise ValueError('Frozen P56 inputs differ')
    provenance=dict(source_uri=SOURCE_URI,policy_sha256=POLICY_SHA256,source_iteration=ITERATION,
        state_manifest_sha256=sha256_file(state/'manifest.json'),corrected_trainer=receipt,
        p56_policy_sha256=P56_SHA256,p56_library_sha256=P56_LIBRARY_SHA256,
        source_read_only=True)
    return state,policy,trainer,contract,frozen,encounters,provenance


def summary(output,mode):
    def result_line(name,report):
        r=report.get('result')
        if not r:return f'{name}: {report["status"]}; no estimate.'
        return (f'{name}: {r["bb_per_100"]:+.3f} bb/100, {100*r["confidence"]:.1f}% interval '
                f'[{r["interval"][0]:+.3f}, {r["interval"][1]:+.3f}]; {r["conclusion"]}.')
    if mode=='policy':
        fit=json.loads((output/'fit/experiment.json').read_text())
        head=json.loads((output/'head/report.json').read_text());fo2=json.loads((output/'fo2/report.json').read_text())
        lines=['# P104 policy extraction experiment','',result_line('Primary',head),'',
            result_line('FO2 improvement',fo2),'',
            'Intervals condition on two fixed fit seeds. Former validation metrics after all-replay fitting are in-sample.',
            '', '| Seed | Arm | Selected updates | Fit seconds |', '|---|---|---:|---:|']
        for r in fit['results']:
            lines.append(f"| {r['fit_seed']} | {r['arm']} | {r['fit']['selected_update']} | {r['fit'].get('fit_seconds','—')} |")
        data=dict(status='COMPLETE',mode=mode,primary=head,fo2=fo2,policy_refit=fit)
    else:
        diag=json.loads((output/'diagnostic/report.json').read_text())
        lines=['# P104 regret investigation','',f"Selected arm: **{diag['selection']['arm']}**. Both fixed fitting seeds retained.",
            f"Conditional fitting gate passed: {diag['fit_gate']['run_fits']}.",'',
            'Diagnostic regret strategies only; no production checkpoint or average policy was replaced.',
            'Sample-mean noise uses an IID sensitivity calculation, not a cluster-valid confidence interval.',
            '', '| Player | Phase | Repeated preflop groups | Within-state variation | Mean-target error | IID mean-noise scale |',
            '|---|---|---:|---:|---:|---:|']
        for player,phases in enumerate(diag['decomposition']):
            for phase in ('before','after'):
                r=phases[phase]['preflop']['repeated']
                lines.append(f"| {player} | {phase} | {r['groups']} | {r['within_state_variance']} | {r['mean_target_mse']} | {r['iid_mean_noise_scale']} |")
        lines += ['', '| Arm | Seed | Selected update | Validation mean-target MSE |', '|---|---:|---:|---:|']
        for r in diag['results']:
            lines.append(f"| {r['arm']} | {r['seed']} | {r['selected_update']} | {r['validation_preflop_mean_mse']:.6g} |")
        data=dict(status='COMPLETE',mode=mode,diagnostic=diag)
    rivers=json.loads((output/'rivers/report.json').read_text())
    data['river_comparisons']=rivers['comparisons']
    lines += ['', 'River comparisons use the same 90 frozen P56 development positions and conditional ranges.',
              'Values below are conditional bb per decision, not whole-game bb/100. Positive gap reduction is better.',
              '', '| Comparison | Panel | Gap reduction | Improved / regressed |', '|---|---|---:|---:|']
    for name,comparison in rivers['comparisons'].items():
        for r in comparison['summaries']:
            if r['scope']=='all':
                lines.append(f"| {name} | {r['panel']} | {r['weighted']['remaining_gap_reduction_bb']:+.6f} | {r['improved']} / {r['regressed']} |")
    write_json(output/'summary.json',data);(output/'report.md').write_text('\n'.join(lines)+'\n')


def run_cloud(workspace,output_uri,*,mode):
    if mode not in ('policy','regret'):raise ValueError('Unknown P104 experiment')
    output_uri=output_uri.rstrip('/')
    for source in (SOURCE_URI,P56_AUDIT_URI,P56_POLICY_URI.rsplit('/',1)[0]):
        if output_uri==source or output_uri.startswith(source+'/') or source.startswith(output_uri+'/'):
            raise ValueError('Output must not overlap any source run')
    root=Path(workspace);root.mkdir(parents=True,exist_ok=False)
    output=root/'results';output.mkdir()
    started=time.monotonic();deadline=started+(12600 if mode=='policy' else 5400)
    status=dict(status='PREPARING',mode=mode,source_uri=SOURCE_URI,output_uri=output_uri,
        source_iteration=ITERATION,collection_iterations=0 if mode=='policy' else 1)
    stop=threading.Event()
    def stage(name):
        status.update(status=name,elapsed_seconds=time.monotonic()-started);write_json(output/'cloud_status.json',status)
    def sync(final=False):
        storage('rsync','--recursive','--exclude',r'^cloud_status\.json$'+('' if final else r'|\.(jsonl|npz|pt|pkl)$'),output,output_uri)
        storage('cp',output/'cloud_status.json',output_uri+'/cloud_status.json')
    def loop():
        while not stop.wait(120):
            try:sync()
            except Exception as exc:print(f'Periodic publication failed: {exc}; final publication remains mandatory.',flush=True)
    stage('PREPARING');publisher=threading.Thread(target=loop,daemon=True);publisher.start()
    try:
        state,policy,trainer,contract,frozen,encounters,provenance=prepare_inputs(root,mode)
        write_json(output/'provenance.json',provenance)
        def child(name,args,cap,report_name,expected='COMPLETE',cwd=None,module=True):
            remaining=deadline-time.monotonic()
            if remaining<60:raise TimeoutError('Diagnostic budget exhausted before '+name)
            budget=min(cap,remaining-30)
            stage(name.upper())
            command=[sys.executable,*(['-m'] if module else []),*map(str,args)]
            if module:command += ['--max-seconds',str(max(1,budget-15))]
            result=run_child(command,output/f'{name}.log',timeout=budget,cwd=cwd)
            if result['timed_out'] or result['returncode']!=0:raise RuntimeError(name+' failed or exceeded its stage guard')
            report=json.loads((output/name/report_name).read_text())
            if report['status']!=expected:raise RuntimeError(name+' did not complete')
        if mode=='policy':
            assets=root/'fo2-assets';prepare_fo2(assets)  # fail before fitting if unavailable
            child('fit',['hulhe_escher.p104_policy_refit','--state',state,'--policy',policy,'--output',output/'fit','--threads','8'],1800,'experiment.json')
            child('head',['hulhe_escher.p104_policy_evaluate','--kind','head','--experiment',output/'fit',
                '--policy',policy,'--output',output/'head','--workers','8'],5400,'report.json')
            child('fo2',['hulhe_escher.p104_policy_evaluate','--kind','fo2','--experiment',output/'fit',
                '--policy',policy,'--output',output/'fo2','--workers','8','--assets',assets,
                '--primary-report',output/'head/report.json'],2700,'report.json')
            experiment=output/'fit'
        else:
            child('capture',[Path(__file__).with_name('bottleneck_iteration.py'),'--source',trainer,'--state',state,
                '--contract',contract,'--policy',policy,'--output',output/'capture',
                '--capture-all-regrets','--save-regret-models','--skip-control-probe'],900,'report.json',cwd=trainer,module=False)
            child('diagnostic',['hulhe_escher.p104_regret','--captures',output/'capture','--state',state,
                '--policy',policy,'--output',output/'diagnostic','--threads','8'],1200,'report.json')
            experiment=output/'diagnostic'
        child('rivers',['hulhe_escher.p104_rivers','--mode',mode,'--experiment',experiment,'--policy',policy,
            '--state',state,'--frozen-policy',frozen,'--encounters',encounters,'--output',output/'rivers',
            '--workers','8'],2400 if mode=='policy' else 1800,'report.json')
        summary(output,mode);stage('COMPLETE')
    except BaseException as exc:
        status['reason']=f'{type(exc).__name__}: {exc}';stage('FAILED');raise
    finally:
        stop.set();publisher.join(timeout=1900);sync(final=True)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output-uri',required=True)
    p.add_argument('--mode',choices=('policy','regret'),required=True)
    run_cloud(**vars(p.parse_args()))

if __name__=='__main__':main()
