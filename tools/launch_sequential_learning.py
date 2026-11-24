#!/usr/bin/env python3
"""Prepare protected sequential 4+4 Google Batch jobs; submit only explicitly."""
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
from hulhe_escher.sequential_training import sequential_config, COMPARISON_CAVEAT


def build_bundle(output, *, run_name):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,45}[a-z0-9]', run_name):
        raise ValueError('Use a lowercase run name of 3–47 characters')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    uri = 'gs://clever-overview-399515-fhp-escher-results/hulhe/'+run_name
    source_archive(Path(__file__).resolve().parents[1], output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    service = 'fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'
    jobs = {}
    for role, hours, disk in [('train', 18, 200), ('external', 28, 50)]:
        command = ['python', '-m', 'hulhe_escher.sequential_experiment', '--role', role,
                   '--output', '/workspace/sequential-'+role, '--upload-uri', uri]
        script = bootstrap(command, uri, digest)
        job = job_spec(script, service, boot_disk_gb=disk, max_run_seconds=hours*3600,
                       labels=dict(experiment='hulhe-sequential-4plus4', role=role))
        if role == 'external':
            job['allocationPolicy']['instances'][0]['policy']['machineType'] = 'c4-standard-4'
            job['taskGroups'][0]['taskSpec']['computeResource'] = dict(cpuMilli=4000, memoryMib=14000)
        (output/(role+'-job.json')).write_text(json.dumps(job, indent=2)+'\n')
        (output/(role+'-batch.sh')).write_text(script)
        jobs[role] = run_name+'-'+role
    plan = dict(project='clever-overview-399515', region='europe-west1', uri=uri, jobs=jobs,
        source_sha256=digest, config=sequential_config(), active_training_hours=8,
        checkpoint_hours=[4, 8], seed=20261115, collection_workers=8, training_threads=8,
        evaluation_workers=8, external_workers=4, per_network_parameters=1496067,
        machine='c4-standard-16', external_machine='c4-standard-4', automatic_retries=0,
        memory_preflight_hours=2, training_wall_hours=12, local_evaluation_hours=2,
        external_wall_hours=26, max_job_hours=dict(train=18, external=28),
        primary='P8 vs P4: 10,000 pilot pairs, 100,000–1,000,000 confirmation pairs, 2 bb/100, 80% planned power, 95% CI',
        secondary='Paired P8 minus P4 against Fell Omen 2; same counts/precision and primary-test gate',
        controls='P4 and P8 vs random/check-call, descriptive',
        comparison_caveat=COMPARISON_CAVEAT,
        cleanup='Batch releases VM/disks; full latest recovery and P4/P8 remain in GCS.')
    (output/'launch.json').write_text(json.dumps(plan, indent=2)+'\n')
    (output/'sequential-config.json').write_text(json.dumps(plan['config'], indent=2)+'\n')
    return plan


def submit(output):
    output = Path(output).resolve()
    plan = json.loads((output/'launch.json').read_text())
    if hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Prepared source archive hash mismatch')
    for name in ('launch.json', 'source.tar.gz', 'sequential-config.json', 'train-job.json', 'external-job.json'):
        subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0', str(output/name), plan['uri']+'/'+name], check=True)
    created = []
    try:
        # Submit the waiting evaluator first, so training never starts without one.
        for role in ('external', 'train'):
            name = plan['jobs'][role]
            subprocess.run(['gcloud', 'batch', 'jobs', 'submit', name, '--project', plan['project'],
                            '--location', plan['region'], '--config', str(output/(role+'-job.json'))], check=True)
            created.append(name)
    except subprocess.CalledProcessError:
        for name in created:
            subprocess.run(['gcloud', 'batch', 'jobs', 'delete', name, '--project', plan['project'],
                            '--location', plan['region'], '--quiet'], check=False)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-name')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--submit-prepared', action='store_true')
    args = p.parse_args()
    if args.submit_prepared:
        if args.run_name or args.execute:
            p.error('Use --submit-prepared with --output only')
        submit(args.output)
    else:
        if not args.run_name:
            p.error('--run-name is required')
        print(json.dumps(build_bundle(args.output, run_name=args.run_name), indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
