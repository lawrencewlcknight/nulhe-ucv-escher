#!/usr/bin/env python3
"""Prepare a small captured-regret diagnostic Batch job; --execute submits it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from batch_runtime import bootstrap, job_spec
from launch_learning_check import source_archive

SOURCE = 'gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535'
INVESTIGATION = 'gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-positions-p32-20261118-181957'
POLICY_HASH = '9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b'
CAPTURE_HASH = '8f372d3ba7e96c2d042d084f91b4be2331e963c04d975276ff79adcf508c10ad'


def build_bundle(output,*,run_name,project='clever-overview-399515',region='europe-west1',
                 bucket='gs://clever-overview-399515-fhp-escher-results/hulhe',
                 service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]',run_name):raise ValueError('Invalid Batch job name')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?',bucket):raise ValueError('Invalid bucket')
    uri=bucket.rstrip('/')+'/'+run_name
    if any(uri==u or uri.startswith(u+'/') or u.startswith(uri+'/') for u in (SOURCE,INVESTIGATION)):
        raise ValueError('Output must not overlap source experiments')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1],output/'source.tar.gz')
    sha=hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command=['python','-m','hulhe_escher.regret_diagnostic_cloud','--workspace','/workspace/regret-diagnostic',
        '--source-uri',SOURCE,'--investigation-uri',INVESTIGATION,'--output-uri',uri,
        '--expected-policy-sha256',POLICY_HASH,'--expected-capture-report-sha256',CAPTURE_HASH,'--iteration','1132']
    script=bootstrap(command,uri,sha)
    job=job_spec(script,service_account,boot_disk_gb=50,max_run_seconds=7200,
                 labels=dict(experiment='hulhe-regret-targets',checkpoint='p32'))
    plan=dict(job_name=run_name,project=project,region=region,uri=uri,source_uri=SOURCE,
        investigation_uri=INVESTIGATION,policy_sha256=POLICY_HASH,capture_report_sha256=CAPTURE_HASH,
        source_sha256=sha,machine='c4-standard-16',threads=8,collection_workers=0,
        full_replay_download=False,diagnostic_seconds=3600,process_guard_seconds=3900,max_task_hours=2,
        automatic_retries=0,fit_seeds=[3901,3902],fit_arms=['warm_current','expanded_current'],
        max_updates_per_player_per_fit=500,learning_rate=.0003,
        data='Existing two-player 4096-row captures, source model metadata, policy and frozen encounter library only',
        cleanup='Batch releases VM and disk; source experiments remain unchanged')
    for name,value in [('launch.json',plan),('job.json',job)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',default='clever-overview-399515');p.add_argument('--region',default='europe-west1')
    p.add_argument('--bucket',default='gs://clever-overview-399515-fhp-escher-results/hulhe')
    p.add_argument('--service-account',default='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com')
    p.add_argument('--execute',action='store_true')
    args=vars(p.parse_args());execute=args.pop('execute');plan=build_bundle(**args)
    print(json.dumps(plan,indent=2))
    if execute:
        for name in ('source.tar.gz','launch.json','job.json'):
            subprocess.run(['gcloud','storage','cp','--if-generation-match=0',str(args['output']/name),plan['uri']+'/'+name],check=True)
        subprocess.run(['gcloud','batch','jobs','submit',plan['job_name'],'--project',plan['project'],
            '--location',plan['region'],'--config',str(args['output']/'job.json')],check=True)

if __name__=='__main__':main()
