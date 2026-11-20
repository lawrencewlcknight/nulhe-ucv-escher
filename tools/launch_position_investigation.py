#!/usr/bin/env python3
"""Build a common-position diagnostic Batch job; --execute explicitly submits it."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from batch_runtime import bootstrap, job_spec
from launch_bottleneck import build_bundle as recovery_bundle


def build_bundle(output,*,checkpoint_label='P32',comparison_label='P32',comparison_uri=None,
                 comparison_policy_uri=None,comparison_policy_sha256=None,
                 comparison_library_sha256=None,**kwargs):
    if not all(re.fullmatch(r'P[0-9]+',label) for label in (checkpoint_label,comparison_label)):
        raise ValueError('Checkpoint label must be P followed by active hours')
    comparison=(comparison_uri,comparison_policy_uri,comparison_policy_sha256,comparison_library_sha256)
    if any(comparison) and not all(comparison):
        raise ValueError('Comparison requires original investigation, policy URI, policy hash and library hash')
    if all(comparison):
        for uri in comparison[:2]:
            if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?',uri):
                raise ValueError('Invalid comparison URI')
        for value in comparison[2:]:
            if not re.fullmatch(r'[0-9a-f]{64}',value):raise ValueError('Invalid comparison hash')
        destination=kwargs['bucket'].rstrip('/')+'/'+kwargs['run_name']
        if any(destination==uri or destination.startswith(uri+'/') or uri.startswith(destination+'/')
               for uri in comparison[:2]):raise ValueError('Comparison inputs must be read-only sibling prefixes')
    # Reuse immutable source packaging, identity validation and guarded GCS
    # submission commands from the full-recovery bottleneck launcher.
    base=recovery_bundle(output,**kwargs)
    output=Path(output)
    command=['python','-m','hulhe_escher.position_cloud','--workspace','/workspace/positions',
        '--source-uri',base['source_uri'],'--output-uri',base['uri'],
        '--expected-policy-sha256',base['expected_policy_sha256'],'--iteration',str(base['iteration']),
        '--checkpoint-label',checkpoint_label]
    if all(comparison):
        command.extend(['--comparison-label',comparison_label])
        for name,value in zip(('comparison-uri','comparison-policy-uri','comparison-policy-sha256',
                               'comparison-library-sha256'),comparison):command.extend(['--'+name,value])
    script=bootstrap(command,base['uri'],base['source_sha256'])
    job=job_spec(script,kwargs['service_account'],boot_disk_gb=150,max_run_seconds=14400,
        labels=dict(experiment='hulhe-positions',checkpoint=checkpoint_label.lower()))
    keep=('deployment','job_name','project','region','uri','source_uri','expected_policy_sha256',
          'iteration','source_sha256','machine','boot_disk_gb','python_container','vm_image',
          'commands','cleanup','automatic_retries','max_task_hours')
    plan={k:base[k] for k in keep}
    plan.update(experiment='hulhe_position_investigation_v1',collection_workers=8,upstream_workers=8,
        analysis_threads=1,hands_per_matchup=20000,river_positions=128,earlier_positions=128,
        upstream_positions=24,estimator_repeats=64,collection_guard_seconds=1860,
        analysis_guard_seconds=3060,upstream_guard_seconds=1500,
        checkpoint_label=checkpoint_label,
        workload=f'Frozen {checkpoint_label} self-play and FO2 encounters; replay scan, exact river diagnostics; one isolated archived iteration.',
        assessment='Locked; development only by default. No candidate selection or strength claim.',
        source_access=f'Read-only {checkpoint_label} checkpoint; isolated local restoration. No production continuation saved.')
    if all(comparison):
        plan.update(comparison_label=comparison_label,comparison_uri=comparison_uri,comparison_policy_uri=comparison_policy_uri,
            comparison_policy_sha256=comparison_policy_sha256,comparison_library_sha256=comparison_library_sha256,
            comparison_guard_seconds=1860,comparison=f'Both exported policies on frozen {comparison_label} development positions, '
            f'same frozen {comparison_label} opponent/posterior and nine synthetic panels; paired descriptive changes, assessment locked.')
    (output/'batch.sh').write_text(script)
    for name,data in [('job.json',job),('launch.json',plan)]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
    return plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project','bucket','service-account','run-name','source-uri','expected-policy-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--region',default='europe-west1');p.add_argument('--iteration',type=int,default=1132)
    p.add_argument('--checkpoint-label',default='P32')
    p.add_argument('--comparison-label',default='P32')
    for name in ('comparison-uri','comparison-policy-uri','comparison-policy-sha256','comparison-library-sha256'):
        p.add_argument('--'+name)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=vars(p.parse_args(argv));execute=args.pop('execute');plan=build_bundle(**args)
    print(json.dumps(plan,indent=2))
    if execute:
        for command in plan['commands']:subprocess.run(command,check=True)

if __name__=='__main__': main()
