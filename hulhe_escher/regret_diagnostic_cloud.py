"""Run the frozen regret-target investigation from small existing cloud artifacts."""
import argparse
from pathlib import Path
import sys

from .checkpointing import sha256_file
from .experiment_runtime import run_child
from .policy_replay_artifacts import prepare
from .storage_transfer import run_transfer
from .strategic_audit.workflow import write_json


def storage(*args):
    return run_transfer(['gcloud','storage',*map(str,args)],timeout=1800)


def run_cloud(workspace,source_uri,investigation_uri,output_uri,expected_policy_sha256,
              expected_capture_report_sha256,*,iteration=1132):
    source_uri,investigation_uri,output_uri=(u.rstrip('/') for u in (source_uri,investigation_uri,output_uri))
    if any(output_uri==u or output_uri.startswith(u+'/') or u.startswith(output_uri+'/')
           for u in (source_uri,investigation_uri)):
        raise ValueError('Diagnostic output must be separate from both source runs')
    root=Path(workspace);root.mkdir(parents=True,exist_ok=False)
    inputs=root/'inputs';inputs.mkdir()
    output=root/'results';output.mkdir()
    status=dict(status='PREPARING',source_uri=source_uri,investigation_uri=investigation_uri,
        source_iteration=iteration,collection=False,full_replay_download=False)
    try:
        for folder,names in [('upstream/iteration',['report.json','regret_0_before.npz','regret_0_after.npz',
                                                   'regret_1_before.npz','regret_1_after.npz']),
                             ('encounters',['report.json','library.json'])]:
            (inputs/folder).mkdir(parents=True)
            for name in names:storage('cp',investigation_uri+'/'+folder+'/'+name,inputs/folder/name)
        if sha256_file(inputs/'upstream/iteration/report.json')!=expected_capture_report_sha256:
            raise ValueError('Source capture report checksum differs')
        policy=inputs/'source-policy.pkl'
        storage('cp',source_uri+f'/training/policy_{iteration:06d}.pkl',policy)
        if sha256_file(policy)!=expected_policy_sha256:raise ValueError('Source policy checksum differs')
        state=inputs/'metadata'
        prepare(source_uri+f'/training/state_{iteration:06d}',state,metadata_only=True)
        status['status']='ANALYSING';write_json(output/'cloud_status.json',status)
        result=run_child([sys.executable,'-m','hulhe_escher.regret_diagnostic',
            '--captures',str(inputs/'upstream/iteration'),'--policy',str(policy),
            '--encounters',str(inputs/'encounters'),'--state',str(state),
            '--output',str(output/'diagnostic'),'--threads','8','--max-seconds','3600'],
            output/'diagnostic.log',timeout=3900)
        if result['timed_out'] or result['returncode']==2:raise TimeoutError('Diagnostic incomplete or process guard reached')
        if result['returncode']!=0:raise RuntimeError('Diagnostic failed; inspect diagnostic.log')
        status['status']='COMPLETE'
    except BaseException as exc:
        status.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        write_json(output/'cloud_status.json',status)
        storage('rsync','--recursive',output,output_uri)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','source-uri','investigation-uri','output-uri','expected-policy-sha256','expected-capture-report-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--iteration',type=int,default=1132)
    run_cloud(**vars(p.parse_args()))

if __name__=='__main__':main()
