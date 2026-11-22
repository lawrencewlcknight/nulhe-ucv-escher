#!/usr/bin/env python3
"""Prepare a fresh HULHE width-scaling Batch job; submission is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from batch_runtime import bootstrap, job_spec, PYTHON_IMAGE, VM_IMAGE
from launch_learning_check import source_archive

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hulhe_escher.scaling_config import (scaling_config, CHECKPOINT_HOURS, COMPARISON_CAVEAT,
                                        HISTORICAL_P32_URI, HISTORICAL_P32_SHA256)

# The 16-vCPU default preserves the historical worker/thread allocation. The
# smaller option changes throughput and is not an equal-hardware comparison.
MACHINES = {'c4-standard-16': (16000, 60000, 8),
            'c4-highmem-8': (8000, 62000, 4),
            'c4-highmem-16': (16000, 124000, 8)}


def build_bundle(output, *, run_name, project='clever-overview-399515', region='europe-west1',
                 bucket='gs://clever-overview-399515-fhp-escher-results/hulhe',
                 service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com',
                 machine='c4-standard-16'):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?', bucket):
        raise ValueError('Invalid bucket prefix')
    if machine not in MACHINES:
        raise ValueError('Unsupported machine')
    uri = bucket.rstrip('/') + '/' + run_name
    if uri == HISTORICAL_P32_URI or uri.startswith(HISTORICAL_P32_URI + '/'):
        raise ValueError('Use an independent prefix; preserve historical P32')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output / 'source.tar.gz')
    digest = hashlib.sha256((output / 'source.tar.gz').read_bytes()).hexdigest()
    cpu, memory, workers = MACHINES[machine]
    command = ['python', '-m', 'hulhe_escher.scaling_experiment', '--output', '/workspace/scaling',
               '--upload-uri', uri, '--workers', str(workers), '--threads', str(workers)]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=200, max_run_seconds=56*3600,
                   labels=dict(experiment='hulhe-wide32', sampler='objective-mass'))
    job['allocationPolicy']['instances'][0]['policy']['machineType'] = machine
    job['taskGroups'][0]['taskSpec']['computeResource'] = dict(cpuMilli=cpu, memoryMib=memory)
    plan = dict(job_name=run_name, project=project, region=region, uri=uri,
                machine=machine, collection_workers=workers, training_threads=workers,
                source_sha256=digest, config=scaling_config(), seed=20261115,
                initialization='fresh; no inherited models or replay', active_training_hours=32,
                checkpoint_hours=list(CHECKPOINT_HOURS), memory_preflight_hours=2,
                memory_headroom_required='max(8 GiB, 12% of host RAM)',
                training_wall_hours=40, evaluation_wall_hours=12, max_job_hours=56,
                recovery_wall_hours=6, recovery_skip_recent_hours=1,
                historical_policy_uri=HISTORICAL_P32_URI + '/P32.pkl',
                historical_policy_sha256=HISTORICAL_P32_SHA256,
                primary='fresh wide P32 vs historical P32; alpha .05, half evaluation budget',
                secondary='P32/P8 and four adjacent comparisons; alpha .01 each',
                pilot_pairs=10000, minimum_pairs=100000, maximum_pairs=1000000,
                effect_bb100=2, target_power=.8, comparison_caveat=COMPARISON_CAVEAT,
                smaller_vm_caveat='8-vCPU option uses four workers/four threads; fewer samples per hour expected',
                automatic_retries=0, slumbot=False, gtow=False,
                python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
                cleanup='Batch releases VM/disk; verified latest full recovery and all policies remain in GCS')
    for name, value in (('launch.json', plan), ('job.json', job), ('scaling-config.json', plan['config'])):
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    (output / 'batch.sh').write_text(script)
    return plan


def submit(output):
    output = Path(output).resolve()
    plan = json.loads((output / 'launch.json').read_text())
    if hashlib.sha256((output / 'source.tar.gz').read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Prepared source archive hash mismatch')
    # Immutable prefix claim makes a repeated submission fail before launching.
    for name in ('launch.json', 'source.tar.gz', 'job.json', 'scaling-config.json'):
        subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0',
                        str(output / name), plan['uri'] + '/' + name], check=True)
    subprocess.run(['gcloud', 'batch', 'jobs', 'submit', plan['job_name'],
                    '--project', plan['project'], '--location', plan['region'],
                    '--config', str(output / 'job.json')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-name')
    parser.add_argument('--machine', choices=tuple(MACHINES), default='c4-standard-16')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--submit-prepared', action='store_true')
    args = parser.parse_args()
    if args.submit_prepared:
        if args.run_name or args.execute:
            parser.error('Use --submit-prepared with --output only')
        submit(args.output)
    else:
        if not args.run_name:
            parser.error('--run-name is required when preparing a job')
        print(json.dumps(build_bundle(args.output, run_name=args.run_name, machine=args.machine), indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
