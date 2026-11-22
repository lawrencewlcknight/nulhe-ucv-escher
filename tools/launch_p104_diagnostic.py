#!/usr/bin/env python3
"""Prepare either independent P104 diagnostic Batch job; submit only with --execute."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from batch_runtime import bootstrap,job_spec
from launch_learning_check import source_archive
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hulhe_escher.p104_protocol import (SOURCE_URI,POLICY_SHA256,P56_AUDIT_URI,P56_POLICY_URI,
    P56_SHA256,P56_LIBRARY_SHA256,ITERATION,POLICY_SEEDS,REGRET_SEEDS,POLICY_BUDGETS)


def build_bundle(output,*,mode,run_name,project='clever-overview-399515',region='europe-west1',
        bucket='gs://clever-overview-399515-fhp-escher-results/hulhe',
        service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'):
    if mode not in ('policy','regret'):raise ValueError('Select policy or regret experiment')
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]',run_name):raise ValueError('Invalid Batch job name')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?',bucket):raise ValueError('Invalid bucket')
    uri=bucket.rstrip('/')+'/'+run_name
    for source in (SOURCE_URI,P56_AUDIT_URI,P56_POLICY_URI.rsplit('/',1)[0]):
        if uri==source or uri.startswith(source+'/') or source.startswith(uri+'/'):
            raise ValueError('Output must not overlap any source run')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1],output/'source.tar.gz')
    digest=hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command=['python','-m','hulhe_escher.p104_cloud','--workspace','/workspace/p104-'+mode,
        '--output-uri',uri,'--mode',mode]
    script=bootstrap(command,uri,digest)
    hours=4 if mode=='policy' else 2
    job=job_spec(script,service_account,boot_disk_gb=200,max_run_seconds=hours*3600,
        labels=dict(experiment='hulhe-p104-'+mode,checkpoint='p104'))
    plan=dict(mode=mode,job_name=run_name,project=project,region=region,uri=uri,source_uri=SOURCE_URI,
        source_iteration=ITERATION,policy_sha256=POLICY_SHA256,source_sha256=digest,
        p56_policy_sha256=P56_SHA256,p56_library_sha256=P56_LIBRARY_SHA256,
        machine='c4-standard-16',threads=8,evaluation_workers=8,disk_gb=200,max_task_hours=hours,
        collection_workers=0 if mode=='policy' else 8,collection_iterations=0 if mode=='policy' else 1,
        automatic_retries=0,fit_seeds=list(POLICY_SEEDS if mode=='policy' else REGRET_SEEDS),
        source_read_only=True,assessment_unlocked=False,
        policy_budgets=list(POLICY_BUDGETS) if mode=='policy' else None,
        primary_design=dict(minimum_groups=400000,maximum_groups=1000000,pilot_groups=10000,
            target_effect_bb100=1.,power=.8,alpha=.05) if mode=='policy' else None,
        cleanup='Batch releases VM and disk; diagnostic outputs stay in GCS. Source and continuation jobs remain untouched.')
    commands=[['gcloud','storage','cp','--if-generation-match=0',str(output/name),uri+'/'+name]
              for name in ('source.tar.gz','launch.json','job.json')]
    commands.append(['gcloud','batch','jobs','submit',run_name,'--project',project,'--location',region,'--config',str(output/'job.json')])
    plan['commands']=commands
    for name,data in [('launch.json',plan),('job.json',job)]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('policy','regret'),required=True)
    p.add_argument('--run-name',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',default='clever-overview-399515');p.add_argument('--region',default='europe-west1')
    p.add_argument('--bucket',default='gs://clever-overview-399515-fhp-escher-results/hulhe')
    p.add_argument('--service-account',default='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com')
    p.add_argument('--execute',action='store_true')
    args=vars(p.parse_args(argv));execute=args.pop('execute');plan=build_bundle(**args)
    print(json.dumps(plan,indent=2))
    if execute:
        for cmd in plan['commands']:subprocess.run(cmd,check=True)

if __name__=='__main__':main()
