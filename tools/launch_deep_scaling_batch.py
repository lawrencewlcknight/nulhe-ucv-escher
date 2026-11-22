#!/usr/bin/env python3
"""Prepare a fresh deep-residual HULHE Batch experiment or its deferred evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from batch_runtime import bootstrap, job_spec, PYTHON_IMAGE, VM_IMAGE
from launch_learning_check import source_archive
from launch_scaling_batch import submit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hulhe_escher.scaling_config import (scaling_config, deep_scaling_config,
                                        CHECKPOINT_HOURS, DEEP_COMPARISON_CAVEAT)


def validate_plan(plan, *, deep=False):
    expected = deep_scaling_config() if deep else scaling_config()
    if (plan['config'] != expected or plan['seed'] != 20261115
            or plan['collection_workers'] != 8 or plan['training_threads'] != 8
            or plan['machine'] != 'c4-standard-16' or plan['checkpoint_hours'] != list(CHECKPOINT_HOURS)
            or not re.fullmatch(r'[a-f0-9]{64}', plan['source_sha256'])
            or not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+/[a-zA-Z0-9_./-]+', plan['uri'])):
        raise ValueError('Reference plan must match the fresh 32-hour configuration and hardware')
    if deep and plan.get('evaluation_only'):
        raise ValueError('Use the original deep training plan, not an evaluation-only plan')


def build_bundle(output, *, run_name, wide_plan, training_plan=None):
    wide = json.loads(Path(wide_plan).read_text())
    validate_plan(wide)
    deep = json.loads(Path(training_plan).read_text()) if training_plan else None
    if deep:
        validate_plan(deep, deep=True)
        if (deep['wide_uri'] != wide['uri'] or deep['wide_source_sha256'] != wide['source_sha256']):
            raise ValueError('Deferred evaluation must use the originally selected wide control')
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    uri = 'gs://clever-overview-399515-fhp-escher-results/hulhe/' + run_name
    for reference in (wide, deep):
        if reference and (uri == reference['uri'] or uri.startswith(reference['uri']+'/')
                          or reference['uri'].startswith(uri+'/')):
            raise ValueError('Use an independent output prefix; preserve both experiments')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command = ['python', '-m', 'hulhe_escher.deep_scaling_experiment', '--output', '/workspace/deep32',
               '--upload-uri', uri, '--wide-uri', wide['uri'], '--wide-source-sha256', wide['source_sha256']]
    if deep:
        command += ['--deep-uri', deep['uri'], '--deep-source-sha256', deep['source_sha256']]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, 'fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com',
                   boot_disk_gb=200 if not deep else 40, max_run_seconds=(56 if not deep else 14)*3600,
                   labels=dict(experiment='hulhe-deep32-eval' if deep else 'hulhe-deep32', sampler='objective-mass'))
    plan = dict(job_name=run_name, project='clever-overview-399515', region='europe-west1', uri=uri,
        machine='c4-standard-16', collection_workers=8, training_threads=8,
        source_sha256=digest, config=deep_scaling_config(), seed=20261115,
        initialization='fresh models, replay, optimizer and RNG; never warm-start from wide weights',
        active_training_hours=0 if deep else 32, checkpoint_hours=list(CHECKPOINT_HOURS),
        memory_preflight_hours=0 if deep else 2, training_wall_hours=0 if deep else 40,
        evaluation_wall_hours=12, max_job_hours=14 if deep else 56,
        evaluation_only=bool(deep), deep_uri=deep['uri'] if deep else None,
        deep_source_sha256=deep['source_sha256'] if deep else None,
        wide_uri=wide['uri'], wide_source_sha256=wide['source_sha256'],
        per_network_parameters=749827, control_network_parameters=777731,
        primary='Deep-P32 vs Wide-P32; alpha .05',
        secondary='Deep-P32/P8 and Deep/Wide at P8/P14/P20/P26; alpha .01 each',
        strategic_audit='Same fixed development library for both P32 policies; ten minutes within evaluation budget',
        comparison_caveat=DEEP_COMPARISON_CAVEAT,
        pilot_pairs=10000, minimum_pairs=100000, maximum_pairs=1000000,
        effect_bb100=2, target_power=.8, automatic_retries=0, slumbot=False, gtow=False,
        recovery_wall_hours=6, recovery_skip_recent_hours=1,
        missing_control='Exit after publishing training; separate evaluation-only job, no waiting VM or chat schedule',
        python_container=PYTHON_IMAGE, vm_image=VM_IMAGE)
    for name, data in [('launch.json', plan), ('job.json', job), ('scaling-config.json', plan['config'])]:
        (output/name).write_text(json.dumps(data, indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-name')
    p.add_argument('--wide-plan', type=Path)
    p.add_argument('--training-plan', type=Path, help='Original deep launch.json; selects evaluation-only')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--submit-prepared', action='store_true')
    args = p.parse_args()
    if args.submit_prepared:
        if args.execute or args.run_name or args.wide_plan or args.training_plan:
            p.error('Use --submit-prepared with --output only')
        submit(args.output)
    else:
        if not args.run_name or not args.wide_plan:
            p.error('Preparing requires --run-name and --wide-plan')
        print(json.dumps(build_bundle(args.output, run_name=args.run_name, wide_plan=args.wide_plan,
                                      training_plan=args.training_plan), indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
