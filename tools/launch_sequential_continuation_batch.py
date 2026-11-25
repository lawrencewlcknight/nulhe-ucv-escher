#!/usr/bin/env python3
"""Prepare the pinned protected sequential P8+24 continuation; submission is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from batch_runtime import bootstrap, job_spec, PYTHON_IMAGE, VM_IMAGE
from launch_learning_check import source_archive
from launch_scaled_continuation_batch import submit

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO/'hulhe_escher/data/sequential_continuation_source.json'


def build_bundle(output, *, run_name):
    spec = json.loads(SOURCE.read_text())
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    uri = 'gs://clever-overview-399515-fhp-escher-results/hulhe/'+run_name
    if (uri == spec['source_uri'] or uri.startswith(spec['source_uri']+'/')
            or spec['source_uri'].startswith(uri+'/')):
        raise ValueError('Continuation needs an independent output prefix')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(REPO, output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command = ['python', '-m', 'hulhe_escher.continuation_cloud',
        '--workspace', '/workspace/continuation', '--source-uri', spec['source_uri'],
        '--output-uri', uri, '--expected-baseline-sha256', spec['policy_sha256'],
        '--baseline-hours', '8', '--require-corrected-sampler', '--sequential-source']
    script = bootstrap(command, uri, digest)
    job = job_spec(script, 'fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com',
        boot_disk_gb=200, max_run_seconds=44*3600,
        labels=dict(experiment='hulhe-sequential-p8-plus24', sampler='objective-mass'))
    plan = dict(job_name=run_name, project='clever-overview-399515', region='europe-west1', uri=uri,
        source_sha256=digest, baseline=spec, machine='c4-standard-16',
        collection_workers=8, training_threads=8, evaluation_workers=8,
        labels=['P8', 'P14', 'P20', 'P26', 'P32'], additional_active_hours=24, stage_hours=6,
        initialization='Resume full P8 with its unmodified archived trainer; preserve models, replay, optimizers and RNGs.',
        per_network_parameters=1496067, protection_coefficient=.3,
        policy_replay_capacity=4000000, policy_group_sampling='objective_mass',
        source_modified=False, config_changed=False, verify_each_stage=True,
        training_wall_hours=30, evaluation_wall_hours=12, max_job_hours=44, automatic_retries=0,
        primary='Sequential-P32 versus its own P8; alpha .05',
        secondary='Four adjacent policy comparisons and four paired FO2 improvements from P8; alpha .05/8, gated by primary success.',
        pilot_pairs=10000, minimum_pairs=100000, maximum_pairs=1000000,
        effect_bb100=2., target_power=.8, evaluation_seed_offset=spec['evaluation_seed_offset'],
        absolute_fo2_scores='Descriptive for P8/P14/P20/P26/P32',
        training_seed=20261115, recovery_wall_hours=6, recovery_skip_recent_hours=1,
        python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
        cleanup='Batch releases VM/disk; original run stays read-only; new playable policies and latest full recovery remain in GCS.')
    for name, value in [('launch.json', plan), ('job.json', job)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--submit-prepared', action='store_true')
    args = p.parse_args()
    if args.submit_prepared:
        if args.run_name or args.execute:
            p.error('Use --submit-prepared with --output only')
        submit(args.output)
    else:
        if not args.run_name:
            p.error('Preparing requires --run-name')
        print(json.dumps(build_bundle(args.output, run_name=args.run_name), indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
