"""Google Batch coordinator for isolated P32 common-position diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import threading

from .bottleneck_cloud import prepare_full, storage
from .checkpointing import sha256_file
from .experiment_runtime import run_child
from .fo2_setup import prepare as prepare_fo2
from .position_investigation.report import render
from .strategic_audit.workflow import write_json


def run_cloud(workspace,source_uri,output_uri,expected_policy_sha256,*,iteration=1132):
    source_uri,output_uri=source_uri.rstrip('/'),output_uri.rstrip('/')
    if source_uri==output_uri or source_uri.startswith(output_uri+'/') or output_uri.startswith(source_uri+'/'):
        raise ValueError('Independent sibling output prefix required')
    root=Path(workspace);root.mkdir(parents=True,exist_ok=False)
    output=root/'results';output.mkdir()
    status=dict(status='PREPARING',source_uri=source_uri,output_uri=output_uri,iteration=iteration)
    stop=threading.Event()
    def stage(name):
        status['status']=name;write_json(output/'cloud_status.json',status)
    def sync(final=False):
        storage('rsync','--recursive','--exclude',r'^cloud_status\.json$'+(r'|\.npz$' if not final else ''),output,output_uri)
        storage('cp',output/'cloud_status.json',output_uri+'/cloud_status.json')
    def loop():
        while not stop.wait(120):
            try: sync()
            except Exception as exc: print(f'Periodic upload failed; final upload required: {exc}',flush=True)
    def child(name,command,timeout,cwd=None):
        stage(name.upper())
        result=run_child(command,output/(name+'.log'),timeout=timeout,cwd=cwd)
        if result['timed_out']: raise TimeoutError(name+' exceeded its process guard')
        if result['returncode']==2: raise TimeoutError(name+' reported incomplete diagnostics')
        if result['returncode']!=0: raise RuntimeError(name+' failed; inspect its log')
    stage('PREPARING');thread=threading.Thread(target=loop,daemon=True);thread.start()
    try:
        assets=root/'fo2-assets';build=prepare_fo2(assets)
        state,policy,contract,trainer,provenance=prepare_full(source_uri,root/'isolated-source',expected_policy_sha256,iteration)
        write_json(output/'provenance.json',dict(**provenance,fo2_fingerprint=build['fingerprint'],
            coordinator_sha256=sha256_file(__file__),protocol='position_investigation_v1'))
        child('encounters',[sys.executable,'-m','hulhe_escher.position_investigation.collect',
            '--policy',str(policy),'--assets',str(assets),'--output',str(output/'encounters'),
            '--hands','20000','--workers','8','--max-seconds','1800'],1860)
        child('regression',[sys.executable,'-m','hulhe_escher.strategic_audit','run',
            '--policy','P32='+str(policy),'--regret-state','Regret='+str(state),
            '--require-suit-invariance','--threads','1','--max-seconds','180','--output',str(output/'regression')],240)
        child('analysis',[sys.executable,'-m','hulhe_escher.position_investigation.analyse',
            '--state',str(state),'--policy',str(policy),'--encounters',str(output/'encounters'),
            '--output',str(output/'analysis'),'--max-seconds','3000'],3060)
        child('upstream',[sys.executable,str(Path(__file__).with_name('position_upstream.py')),
            '--source',str(trainer),'--state',str(state),'--contract',str(contract),'--policy',str(policy),
            '--selection',str(output/'analysis/upstream_positions.json'),'--output',str(output/'upstream'),
            '--repeats','64','--max-seconds','1200'],1500,cwd=trainer)
        summary=render(output)
        stage(summary['status'])
    except TimeoutError as exc:
        status['reason']=str(exc);stage('INCOMPLETE');render(output)
    except BaseException as exc:
        status['reason']=f'{type(exc).__name__}: {exc}';stage('FAILED');render(output);raise
    finally:
        stop.set();thread.join();sync(final=True)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    for n in ('source-uri','output-uri','expected-policy-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--iteration',type=int,default=1132)
    if run_cloud(**vars(p.parse_args()))['status']!='COMPLETE':raise SystemExit(2)

if __name__=='__main__': main()
