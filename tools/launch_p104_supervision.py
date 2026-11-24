#!/usr/bin/env python3
"""Build an immutable P104 supervision audit Batch bundle; submit only with --execute."""
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
from hulhe_escher.p104_protocol import SOURCE_URI, POLICY_SHA256, ITERATION


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
    script = bootstrap(['python','-m','hulhe_escher.p104_supervision_cloud',
                        '--workspace','/workspace/p104-supervision','--output-uri',uri], uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=200, max_run_seconds=8*3600,
                   labels=dict(experiment='hulhe-p104-supervision', checkpoint='p104'))
    # Keep launcher dependency-free on the user's terminal; constants match runtime tests.
    plan = dict(job_name=run_name, project=project, region=region, uri=uri,
        source_uri=SOURCE_URI, source_iteration=ITERATION, policy_sha256=POLICY_SHA256,
        source_sha256=digest, machine='c4-standard-16', workers=8,
        disk_gb=200, max_task_hours=8, coordinator_hours=7, automatic_retries=0,
        production_traversals=20000, positions=96, street_counts=dict(flop=16,turn=32,river=48),
        partition_counts=dict(development=48,validation=24,assessment=24),
        hidden_draws=32, continuations_per_hidden_draw=8, reference_samples_per_action=4096,
        critic_folds=2, exact_river_references=True, training_iterations=0,
        arms=['current','average4','all_actions','uniform'], beta_one_alias='current',
        source_read_only=True, metadata_only_download=True, sample_not_reduced_for_runtime=True,
        selection='Validation only, then freeze one candidate before assessment.',
        primary='Encounter-weighted equal-CPU target MSE on untouched assessment positions.',
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
