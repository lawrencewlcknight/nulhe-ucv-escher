#!/usr/bin/env python3
"""Build a common-position diagnostic Batch job; --execute explicitly submits it."""
import argparse
import json
from pathlib import Path
import subprocess

from batch_runtime import bootstrap, job_spec
from launch_bottleneck import build_bundle as recovery_bundle


def build_bundle(output,**kwargs):
    # Reuse immutable source packaging, identity validation and guarded GCS
    # submission commands from the full-recovery bottleneck launcher.
    base=recovery_bundle(output,**kwargs)
    output=Path(output)
    command=['python','-m','hulhe_escher.position_cloud','--workspace','/workspace/positions',
        '--source-uri',base['source_uri'],'--output-uri',base['uri'],
        '--expected-policy-sha256',base['expected_policy_sha256'],'--iteration',str(base['iteration'])]
    script=bootstrap(command,base['uri'],base['source_sha256'])
    job=job_spec(script,kwargs['service_account'],boot_disk_gb=150,max_run_seconds=14400,
        labels=dict(experiment='hulhe-positions',checkpoint='p32'))
    keep=('deployment','job_name','project','region','uri','source_uri','expected_policy_sha256',
          'iteration','source_sha256','machine','boot_disk_gb','python_container','vm_image',
          'commands','cleanup','automatic_retries','max_task_hours')
    plan={k:base[k] for k in keep}
    plan.update(experiment='hulhe_position_investigation_v1',collection_workers=8,upstream_workers=8,
        analysis_threads=1,hands_per_matchup=20000,river_positions=128,earlier_positions=128,
        upstream_positions=24,estimator_repeats=64,collection_guard_seconds=1860,
        analysis_guard_seconds=3060,upstream_guard_seconds=1500,
        workload='Frozen P32 self-play and FO2 encounters; replay scan, exact river diagnostics; one isolated archived iteration.',
        assessment='Locked; development only by default. No candidate selection or strength claim.',
        source_access='Read-only original P32 checkpoint; isolated local restoration. No production continuation saved.')
    (output/'batch.sh').write_text(script)
    for name,data in [('job.json',job),('launch.json',plan)]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
    return plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project','bucket','service-account','run-name','source-uri','expected-policy-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--region',default='europe-west1');p.add_argument('--iteration',type=int,default=1132)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=vars(p.parse_args(argv));execute=args.pop('execute');plan=build_bundle(**args)
    print(json.dumps(plan,indent=2))
    if execute:
        for command in plan['commands']:subprocess.run(command,check=True)

if __name__=='__main__': main()
