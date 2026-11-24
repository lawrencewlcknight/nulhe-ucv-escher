#!/usr/bin/env python3
"""Prepare fresh hybrid32 training or evaluation-only Google Batch; submission is explicit."""
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
from hulhe_escher.scaling_config import hybrid_scaling_config, CHECKPOINT_HOURS, HYBRID_COMPARISON_CAVEAT
from hulhe_escher.scaling_corrected_comparison import SOURCES
from hulhe_escher.scaling_corrected_cloud import validate_output


def build_bundle(output, *, run_name, training_plan=None):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    uri = validate_output('gs://clever-overview-399515-fhp-escher-results/hulhe/'+run_name)
    sources = json.loads(SOURCES.read_text())
    training = json.loads(Path(training_plan).read_text()) if training_plan else None
    if training:
        if (training.get('evaluation_only') or training['config'] != hybrid_scaling_config()
                or training['seed'] != 20261115 or training['collection_workers'] != 8
                or training['training_threads'] != 8 or training['machine'] != 'c4-standard-16'
                or training['checkpoint_hours'] != list(CHECKPOINT_HOURS)
                or training['comparison_sources'] != sources
                or not re.fullmatch(r'[a-f0-9]{64}', training['source_sha256'])
                or not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+/[a-zA-Z0-9_./-]+', training['uri'])):
            raise ValueError('Use the original matched hybrid training plan')
        if (uri == training['uri'] or uri.startswith(training['uri']+'/')
                or training['uri'].startswith(uri+'/')):
            raise ValueError('Evaluation output must be independent of training')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command = ['python', '-m', 'hulhe_escher.hybrid_scaling_experiment',
               '--output', '/workspace/hybrid32', '--upload-uri', uri]
    if training:
        command += ['--training-uri', training['uri'], '--training-source-sha256', training['source_sha256']]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, 'fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com',
        boot_disk_gb=40 if training else 200, max_run_seconds=(14 if training else 56)*3600,
        labels=dict(experiment='hulhe-hybrid32-eval' if training else 'hulhe-hybrid32', sampler='objective-mass'))
    plan = dict(job_name=run_name, project='clever-overview-399515', region='europe-west1', uri=uri,
        machine='c4-standard-16', collection_workers=8, training_threads=8,
        source_sha256=digest, config=hybrid_scaling_config(), seed=20261115,
        initialization='Fresh models, replay, optimizers and RNG; no warm start or inherited policies.',
        active_training_hours=0 if training else 32, checkpoint_hours=list(CHECKPOINT_HOURS),
        memory_preflight_hours=0 if training else 2, training_wall_hours=0 if training else 40,
        evaluation_wall_hours=12, max_job_hours=14 if training else 56,
        evaluation_only=bool(training), training_uri=training['uri'] if training else None,
        training_source_sha256=training['source_sha256'] if training else None,
        per_network_parameters=1245699, comparison_sources=sources,
        primary='Hybrid-P32 vs Deep-P32; alpha .05',
        secondary='Hybrid-P32 vs equally weighted corrected P32 refits and vs Fell Omen 2; alpha .025 each',
        progression='P32/P8 and four adjacent checkpoint comparisons; separate family alpha .01 each',
        strategic_audit='Fixed development library shared by Hybrid/Deep/Wide P32; assessment stays locked',
        benchmark_seconds=32400, progression_seconds=10200, strategic_audit_seconds=600,
        pilot_groups=10000, minimum_groups=100000, maximum_groups=1000000,
        effect_bb100=2, target_power=.8, recovery_wall_hours=6, recovery_skip_recent_hours=1,
        automatic_retries=0, fell_omen_2=True, slumbot=False, gtow=False,
        comparison_caveat=HYBRID_COMPARISON_CAVEAT, python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
        cleanup='Batch releases VM/disk; verified full recovery and playable checkpoints remain in GCS.')
    for name, value in [('launch.json', plan), ('job.json', job), ('scaling-config.json', plan['config'])]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-name')
    p.add_argument('--training-plan', type=Path, help='Original hybrid launch.json; evaluation only')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--submit-prepared', action='store_true')
    args = p.parse_args()
    if args.submit_prepared:
        if args.run_name or args.training_plan or args.execute:
            p.error('Use --submit-prepared with --output only')
        submit(args.output)
    else:
        if not args.run_name:
            p.error('Preparing requires --run-name')
        print(json.dumps(build_bundle(args.output, run_name=args.run_name,
                                     training_plan=args.training_plan), indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
