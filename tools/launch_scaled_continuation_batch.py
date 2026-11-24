#!/usr/bin/env python3
"""Prepare independent wide/deep P32+24 Google Batch continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from batch_runtime import bootstrap, job_spec, PYTHON_IMAGE, VM_IMAGE
from launch_learning_check import source_archive

REPO = Path(__file__).resolve().parents[1]
SOURCES = REPO/'hulhe_escher/data/scaled_continuation_sources.json'


def build_bundle(output, *, variant, run_name):
    sources = json.loads(SOURCES.read_text())
    if variant not in sources:
        raise ValueError('Choose wide or deep')
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    spec = sources[variant]
    uri = 'gs://clever-overview-399515-fhp-escher-results/hulhe/'+run_name
    if any(uri == s['source_uri'] or uri.startswith(s['source_uri']+'/')
           or s['source_uri'].startswith(uri+'/') for s in sources.values()):
        raise ValueError('Continuation needs an independent output prefix')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(REPO, output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command = ['python', '-m', 'hulhe_escher.continuation_cloud',
        '--workspace', '/workspace/continuation', '--source-uri', spec['source_uri'],
        '--output-uri', uri, '--expected-baseline-sha256', spec['policy_sha256'],
        '--baseline-hours', '32', '--require-corrected-sampler', '--scaled-variant', variant]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, 'fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com',
        boot_disk_gb=200, max_run_seconds=44*3600,
        labels=dict(experiment=f'hulhe-{variant}-p32-plus24', sampler='objective-mass'))
    plan = dict(job_name=run_name, project='clever-overview-399515', region='europe-west1', uri=uri,
        variant=variant, source_sha256=digest, baseline=spec,
        machine='c4-standard-16', collection_workers=8, training_threads=8,
        labels=['P32', 'P38', 'P44', 'P50', 'P56'], additional_active_hours=24, stage_hours=6,
        initialization='Resume full P32 state with its unmodified archived trainer; preserve models, replay, optimizers and RNGs.',
        per_network_parameters=777731 if variant == 'wide' else 749827,
        policy_replay_capacity=12000000, policy_group_sampling='objective_mass',
        source_modified=False, config_changed=False, verify_each_stage=True,
        training_wall_hours=30, evaluation_wall_hours=12, max_job_hours=44, automatic_retries=0,
        primary=f'{variant.capitalize()}-P56 versus its own P32; alpha .05',
        secondary='Four adjacent policy comparisons and four paired FO2 improvements from P32; alpha .05/8, gated by primary success.',
        pilot_pairs=10000, minimum_pairs=100000, maximum_pairs=1000000,
        effect_bb100=2., target_power=.8, evaluation_seed_offset=spec['evaluation_seed_offset'],
        absolute_fo2_scores='Descriptive for P32/P38/P44/P50/P56',
        training_seed=20261115, recovery_wall_hours=6, recovery_skip_recent_hours=1,
        python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
        cleanup='Batch releases VM/disk; source run stays read-only; new playable policies and latest full recovery remain in GCS.')
    for name, value in [('launch.json', plan), ('job.json', job)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def submit(output):
    output = Path(output).resolve()
    plan = json.loads((output/'launch.json').read_text())
    if hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Prepared source archive hash mismatch')
    for name in ('launch.json', 'source.tar.gz', 'job.json'):
        subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0',
                        str(output/name), plan['uri']+'/'+name], check=True)
    subprocess.run(['gcloud', 'batch', 'jobs', 'submit', plan['job_name'],
        '--project', plan['project'], '--location', plan['region'],
        '--config', str(output/'job.json')], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('wide', 'deep'))
    p.add_argument('--run-name')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--submit-prepared', action='store_true')
    args = p.parse_args()
    if args.submit_prepared:
        if args.variant or args.run_name or args.execute:
            p.error('Use --submit-prepared with --output only')
        submit(args.output)
    else:
        if not args.variant or not args.run_name:
            p.error('Preparing requires --variant and --run-name')
        print(json.dumps(build_bundle(args.output, variant=args.variant, run_name=args.run_name), indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
