"""Isolated full-P32 diagnostics; source checkpoint and production runs are read-only."""
import argparse
import json
from pathlib import Path
import sys
import threading

from .checkpointing import sha256_file
from .continuation_cloud import extract_source
from .experiment_runtime import run_child
from .fo2_setup import prepare as prepare_fo2
from .policy_replay_artifacts import prepare, read_metadata, safe_path, verify_file
from .storage_transfer import run_transfer
from .train import write_json


def storage(*args):
    run_transfer(['gcloud','storage',*map(str,args)],timeout=3600)


def prepare_full(source_uri,root,expected_policy_sha256,iteration):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    state=root/f'state_{iteration:06d}'
    prepare(f'{source_uri}/training/state_{iteration:06d}',state,metadata_only=True)
    manifest,metadata=read_metadata(state)
    if metadata['solver']['num_iteration']!=iteration: raise ValueError('Recovery iteration mismatch')
    # Download exactly this immutable full checkpoint, never the source run recursively.
    for record in manifest['array_shards']:
        path=safe_path(state,record['path']); path.parent.mkdir(parents=True,exist_ok=True)
        storage('cp',f'{source_uri}/training/state_{iteration:06d}/'+record['path'],path)
        verify_file(path,record)
    policy=root/'P32.pkl'; contract=root/'training_manifest.json'; archive=root/'training_source.tar.gz'
    for name,path in ((f'training/policy_{iteration:06d}.pkl',policy),('training/manifest.json',contract),('training_source.tar.gz',archive)):
        storage('cp',source_uri+'/'+name,path)
    if sha256_file(policy)!=expected_policy_sha256: raise ValueError('P32 policy hash mismatch')
    trainer=root/'archived-trainer'; extract_source(archive,trainer)
    return state,policy,contract,trainer,dict(full_checkpoint_bytes=manifest['total_size_bytes'],
        state_manifest_sha256=sha256_file(state/'manifest.json'),policy_sha256=sha256_file(policy),
        original_source_archive_sha256=sha256_file(archive))


def run_cloud(workspace,source_uri,output_uri,expected_policy_sha256,*,iteration=1132):
    source_uri,output_uri=source_uri.rstrip('/'),output_uri.rstrip('/')
    if source_uri==output_uri or source_uri.startswith(output_uri+'/') or output_uri.startswith(source_uri+'/'):
        raise ValueError('Use a separate sibling output prefix')
    root=Path(workspace); root.mkdir(parents=True,exist_ok=False)
    output=root/'results'; output.mkdir()
    status=dict(status='PREPARING',source_uri=source_uri,output_uri=output_uri,iteration=iteration)
    stop=threading.Event()
    def stage(name):
        status['status']=name; write_json(output/'cloud_status.json',status)
    def sync(final=False):
        exclude=r'^cloud_status\.json$'+(r'|\.jsonl$|\.npz$' if not final else '')
        storage('rsync','--recursive','--exclude',exclude,output,output_uri)
        storage('cp',output/'cloud_status.json',output_uri+'/cloud_status.json')
    def loop():
        while not stop.wait(120):
            try: sync()
            except Exception as exc: print(f'Periodic upload failed; final upload required: {exc}',flush=True)
    stage('PREPARING'); thread=threading.Thread(target=loop,daemon=True); thread.start()
    try:
        assets=root/'fo2-assets'; build=prepare_fo2(assets)
        state,policy,contract,trainer,provenance=prepare_full(source_uri,root/'isolated-source',expected_policy_sha256,iteration)
        write_json(output/'provenance.json',dict(**provenance,fo2_fingerprint=build['fingerprint'],
            code_hashes={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*bottleneck*.py')}))
        stage('UPSTREAM_ITERATION')
        script=Path(__file__).with_name('bottleneck_iteration.py')
        outcome=run_child([sys.executable,str(script),'--source',str(trainer),'--state',str(state),
            '--contract',str(contract),'--policy',str(policy),'--output',str(output/'upstream')],
            output/'upstream.log',timeout=900,cwd=trainer)
        if outcome['returncode']!=0 or outcome['timed_out']:
            raise RuntimeError('Upstream iteration failed or exceeded 15-minute guard; see upstream.log')
        stage('POLICY_REFITS')
        outcome=run_child([sys.executable,'-m','hulhe_escher.bottleneck_refit','--state',str(state),
            '--policy',str(policy),'--output',str(output/'fit'),'--threads','8'],output/'fit.log',timeout=3600)
        if outcome['returncode']!=0 or outcome['timed_out']:
            raise RuntimeError('Policy refits failed or exceeded one-hour guard; partial diagnostics preserved')
        stage('EVALUATING_SELECTED_POLICY')
        outcome=run_child([sys.executable,'-m','hulhe_escher.bottleneck_evaluate','--experiment',str(output/'fit'),
            '--policy',str(policy),'--output',str(output/'evaluation'),'--assets',str(assets)],output/'evaluation.log',timeout=3900)
        if outcome['returncode']!=0 or outcome['timed_out']:
            raise RuntimeError('Selected-policy evaluation incomplete; see evaluation/report.json or log')
        fit=json.loads((output/'fit/experiment.json').read_text())
        upstream=json.loads((output/'upstream/report.json').read_text())
        write_json(output/'summary.json',dict(status='COMPLETE',selection=fit['selection'],
            refit_seconds=fit['elapsed_seconds'],upstream_seconds=upstream['iteration_seconds'],
            evaluation=json.loads((output/'evaluation/report.json').read_text()),
            caution='Inference timings describe the policy candidate. They do not measure collection with larger regret/critic networks; do not infer total training throughput from policy refit speed.'))
        stage('COMPLETE')
    except BaseException as exc:
        status['error']=f'{type(exc).__name__}: {exc}'; stage('FAILED'); raise
    finally:
        stop.set(); thread.join(); sync(final=True)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    for n in ('source-uri','output-uri','expected-policy-sha256'): p.add_argument('--'+n,required=True)
    p.add_argument('--iteration',type=int,default=1132)
    run_cloud(**vars(p.parse_args()))

if __name__=='__main__': main()
