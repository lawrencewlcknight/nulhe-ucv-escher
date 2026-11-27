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


def build_bundle(output, *, run_name, baseline_hours=8):
    if baseline_hours not in (8, 32):
        raise ValueError('Sequential continuation supports pinned P8 or P32 sources')
    sources = {8: json.loads(SOURCE.read_text()),
        32: json.loads(SOURCE.with_name('sequential_p32_continuation_source.json').read_text())}
    spec = sources[baseline_hours]
    labels = [f'P{baseline_hours+6*i}' for i in range(5)]
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    uri = 'gs://clever-overview-399515-fhp-escher-results/hulhe/'+run_name
    if any(uri == source['source_uri'] or uri.startswith(source['source_uri']+'/')
            or source['source_uri'].startswith(uri+'/') for source in sources.values()):
        raise ValueError('Continuation needs an independent output prefix')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(REPO, output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command = ['python', '-m', 'hulhe_escher.continuation_cloud',
        '--workspace', '/workspace/continuation', '--source-uri', spec['source_uri'],
        '--output-uri', uri, '--expected-baseline-sha256', spec['policy_sha256'],
        '--baseline-hours', str(baseline_hours), '--require-corrected-sampler', '--sequential-source']
    script = bootstrap(command, uri, digest)
    job = job_spec(script, 'fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com',
        boot_disk_gb=200, max_run_seconds=44*3600,
        labels=dict(experiment=f'hulhe-sequential-p{baseline_hours}-plus24', sampler='objective-mass'))
    plan = dict(job_name=run_name, project='clever-overview-399515', region='europe-west1', uri=uri,
        source_sha256=digest, baseline=spec, machine='c4-standard-16',
        collection_workers=8, training_threads=8, evaluation_workers=8,
        labels=labels, additional_active_hours=24, stage_hours=6,
        initialization=f'Resume full {labels[0]} with its unmodified archived trainer; preserve models, replay, optimizers and RNGs.',
        per_network_parameters=1496067, protection_coefficient=.3,
        policy_replay_capacity=4000000, policy_group_sampling='objective_mass',
        source_modified=False, config_changed=False, verify_each_stage=True,
        training_wall_hours=30, evaluation_wall_hours=12, max_job_hours=44, automatic_retries=0,
        primary=f'Sequential-{labels[-1]} versus its own {labels[0]}; alpha .05',
        secondary=f'Four adjacent policy comparisons and four paired FO2 improvements from {labels[0]}; alpha .05/8, gated by primary success.',
        pilot_pairs=10000, minimum_pairs=100000, maximum_pairs=1000000,
        effect_bb100=2., target_power=.8, evaluation_seed_offset=spec['evaluation_seed_offset'],
        absolute_fo2_scores='Descriptive for '+ '/'.join(labels),
        training_seed=20261115, recovery_wall_hours=6, recovery_skip_recent_hours=1,
        python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
        cleanup='Batch releases VM/disk; original run stays read-only; new playable policies and latest full recovery remain in GCS.')
    for name, value in [('launch.json', plan), ('job.json', job)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def main(baseline_hours=8):
    p = argparse.ArgumentParser(description=f'Prepare the pinned sequential P{baseline_hours}+24 Batch continuation.')
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
        print(json.dumps(build_bundle(args.output, run_name=args.run_name, baseline_hours=baseline_hours), indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
