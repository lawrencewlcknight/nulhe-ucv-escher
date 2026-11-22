#!/usr/bin/env python3
"""Build an immutable P104 ensemble Batch bundle; submit only with --execute."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from batch_runtime import bootstrap, job_spec
from launch_learning_check import source_archive
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hulhe_escher.p104_protocol import SOURCE_URI, POLICY_SHA256, ITERATION, POLICY_BUDGETS


def build_bundle(output, *, run_name, project='clever-overview-399515', region='europe-west1',
                 bucket='gs://clever-overview-399515-fhp-escher-results/hulhe',
                 service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?', bucket):
        raise ValueError('Invalid bucket')
    uri = bucket.rstrip('/')+'/'+run_name
    if uri == SOURCE_URI or uri.startswith(SOURCE_URI+'/') or SOURCE_URI.startswith(uri+'/'):
        raise ValueError('Output must not overlap source')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    script = bootstrap(['python','-m','hulhe_escher.p104_ensemble_cloud',
                        '--workspace','/workspace/p104-ensemble','--output-uri',uri], uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=200, max_run_seconds=12*3600,
                   labels=dict(experiment='hulhe-p104-ensemble', checkpoint='p104'))
    # Keep launcher dependency-free on the user's terminal; constants match runtime tests.
    plan = dict(job_name=run_name, project=project, region=region, uri=uri,
        source_uri=SOURCE_URI, source_iteration=ITERATION, policy_sha256=POLICY_SHA256,
        source_sha256=digest, machine='c4-standard-16', threads=8, evaluation_workers=8,
        disk_gb=200, max_task_hours=12, coordinator_hours=11, automatic_retries=0,
        members=20, fit_seeds=list(range(1045001,1045021)), sampling_seeds=list(range(1046001,1046021)),
        policy_budgets=list(POLICY_BUDGETS), collection_iterations=0, source_read_only=True,
        aggregation='Equal arithmetic mean of legal-action probabilities at each decision.',
        evaluation=dict(pilot_groups=10000, minimum_groups=100000, maximum_groups=1000000,
            target_effect_bb100=2., power=.8, familywise_alpha=.05, alpha_per_test=.05/3,
            tests=['head_to_head','fo2_improvement','ensemble_vs_fo2'],
            pilot_seed=1049701, confirmation_seed=1049702, sample_not_reduced_for_runtime=True),
        cleanup='Batch releases VM/disk; results remain in GCS. Source and continuation untouched.')
    commands = [['gcloud','storage','cp','--if-generation-match=0',str(output/name),uri+'/'+name]
                for name in ('source.tar.gz','launch.json','job.json')]
    commands.append(['gcloud','batch','jobs','submit',run_name,'--project',project,
                     '--location',region,'--config',str(output/'job.json')])
    plan['commands'] = commands
    for name, data in [('launch.json',plan),('job.json',job)]:
        (output/name).write_text(json.dumps(data, indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--project', default='clever-overview-399515')
    p.add_argument('--region', default='europe-west1')
    p.add_argument('--bucket', default='gs://clever-overview-399515-fhp-escher-results/hulhe')
    p.add_argument('--service-account', default='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com')
    p.add_argument('--execute', action='store_true')
    args = vars(p.parse_args(argv))
    execute = args.pop('execute')
    plan = build_bundle(**args)
    print(json.dumps(plan, indent=2))
    if execute:
        for cmd in plan['commands']:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
