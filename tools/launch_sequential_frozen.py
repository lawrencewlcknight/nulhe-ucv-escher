#!/usr/bin/env python3
"""Prepare frozen-target diagnostic Batch; submission requires --execute."""
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
from hulhe_escher import sequential_protocol as protocol


def build_bundle(output, *, run_name, project='clever-overview-399515', region='europe-west1',
                 bucket='gs://clever-overview-399515-fhp-escher-results/hulhe',
                 service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid job name')
    if not re.fullmatch(r'[a-z]+-[a-z]+[0-9]+', region):
        raise ValueError('Invalid region')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_-]+)*', bucket.rstrip('/')):
        raise ValueError('Invalid bucket')
    uri = bucket.rstrip('/')+'/'+run_name
    for source in (protocol.SOURCE_URI, protocol.CAPTURE_URI.rsplit('/', 1)[0]):
        if uri == source or uri.startswith(source+'/') or source.startswith(uri+'/'):
            raise ValueError('Output overlaps source')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], root/'source.tar.gz')
    digest = hashlib.sha256((root/'source.tar.gz').read_bytes()).hexdigest()
    script = bootstrap(['timeout', '--signal=TERM', '--kill-after=60s', '9900s',
        'python', '-m', 'hulhe_escher.sequential_frozen_cloud', '--workspace', '/workspace/sequential-frozen',
        '--output-uri', uri], uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=100, max_run_seconds=10800,
                   labels=dict(experiment='hulhe-sequential-frozen', checkpoint='p104'))
    job['allocationPolicy']['location'] = dict(allowedLocations=['regions/'+region])
    plan = dict(protocol=protocol.PROTOCOL, job_name=run_name, project=project, region=region, uri=uri,
        source_uri=protocol.SOURCE_URI, capture_uri=protocol.CAPTURE_URI, source_sha256=digest,
        machine='c4-standard-16', threads=8, disk_gb=100, max_task_hours=3,
        diagnostic_seconds=5400, collection_iterations=0, automatic_retries=0,
        architecture_parameters=1496067, arms=protocol.ARMS, seeds=protocol.SEEDS,
        datasets=['regret_0', 'regret_1', 'policy'], warmup_updates=protocol.WARMUP_UPDATES,
        budgets=protocol.BUDGETS, max_policy_groups_per_street=protocol.POLICY_GROUPS_PER_STREET,
        note='Diagnostics only; no poker evaluation, playable exports or production changes.')
    for name, value in (('launch.json', plan), ('job.json', job)):
        (root/name).write_text(json.dumps(value, indent=2)+'\n')
    (root/'batch.sh').write_text(script)
    return plan


def submit(output, plan):
    root = Path(output)
    if hashlib.sha256((root/'source.tar.gz').read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Source archive changed')
    # Immutable launch.json reserves a new prefix; never overwrite a prior run.
    for name in ('launch.json', 'source.tar.gz', 'job.json'):
        subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0', str(root/name),
                        plan['uri']+'/'+name], check=True, timeout=300)
    subprocess.run(['gcloud', 'batch', 'jobs', 'submit', plan['job_name'], '--project', plan['project'],
                    '--location', plan['region'], '--config', str(root/'job.json')], check=True, timeout=180)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    plan = build_bundle(args.output, run_name=args.run_name)
    print(json.dumps(plan, indent=2))
    if args.execute:
        submit(args.output, plan)


if __name__ == '__main__':
    main()
