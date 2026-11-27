#!/usr/bin/env python3
"""Prepare the evaluation-only Sequential P32 comparison; submission is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from batch_runtime import bootstrap, job_spec
from launch_learning_check import source_archive


REPO = Path(__file__).resolve().parents[1]
SOURCES = REPO/'hulhe_escher/data/sequential_p32_comparison_sources.json'


def validate_output(uri, sources):
    uri = uri.rstrip('/')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?', uri):
        raise ValueError('Invalid output bucket/prefix')
    parents = {r['uri'].split('/fit/')[0] if '/fit/' in r['uri'] else r['uri'].rsplit('/', 1)[0]
               for r in sources['files'].values()}
    if any(uri == p or uri.startswith(p+'/') or p.startswith(uri+'/') for p in parents):
        raise ValueError('Output overlaps a source experiment')
    return uri


def build_bundle(output, *, run_name, project='clever-overview-399515', region='europe-west1',
                 bucket='gs://clever-overview-399515-fhp-escher-results/hulhe',
                 service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    sources = json.loads(SOURCES.read_text())
    uri = validate_output(bucket.rstrip('/')+'/'+run_name, sources)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(REPO, output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command = ['python', '-m', 'hulhe_escher.sequential_p32_comparison_cloud',
               '--workspace', '/workspace/sequential-p32-comparison', '--output-uri', uri]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=50, max_run_seconds=25200,
                   labels=dict(experiment='hulhe-sequential-p32-vs-all', checkpoint='p32'))
    plan = dict(job_name=run_name, project=project, region=region, uri=uri, source_sha256=digest,
        sources=sources, machine='c4-standard-16', workers=8, boot_disk_gb=50,
        collection=False, retraining=False, replay_download=False, fo2=False,
        primary=['Sequential-P32 vs Small-P32-corrected', 'Sequential-P32 vs Wide-P32',
                 'Sequential-P32 vs Deep-P32', 'Sequential-P32 vs Hybrid-P32'],
        small_comparator='Equal weight on archived corrected P32 refits, seeds 1701 and 1702',
        pilot_seed=927321, confirmation_seed=927322, smoke_seed=927921,
        pilot_groups=10000, minimum_groups=100000, maximum_groups=1000000,
        alpha_per_test=.05/4, confidence=1-.05/4, familywise_alpha=.05,
        power=.8, target_effect_bb100=2., evaluation_seconds=21600,
        process_guard_seconds=21900, max_task_seconds=25200, max_task_hours=7, automatic_retries=0,
        hands_per_group=10,
        unit='Shared deal group; average seats for each matchup and both fixed refits for Small',
        sample_rule='Maximum requirement across all four tests; no reduction for runtime; pilot excluded',
        stopping='Freeze confirmation size after pilot; fail or report incomplete on budget exhaustion',
        cleanup='Batch releases VM and disk; source experiments remain unchanged')
    for name, value in [('launch.json', plan), ('job.json', job)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def submit(output):
    output = Path(output).resolve()
    plan = json.loads((output/'launch.json').read_text())
    validate_output(plan['uri'], plan['sources'])
    if hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Prepared source archive hash mismatch')
    # Claim the fresh run prefix before uploading source or requesting compute.
    for name in ('launch.json', 'source.tar.gz', 'job.json'):
        subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0',
                        str(output/name), plan['uri']+'/'+name], check=True)
    subprocess.run(['gcloud', 'batch', 'jobs', 'submit', plan['job_name'],
        '--project', plan['project'], '--location', plan['region'],
        '--config', str(output/'job.json')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project', default='clever-overview-399515')
    parser.add_argument('--region', default='europe-west1')
    parser.add_argument('--bucket', default='gs://clever-overview-399515-fhp-escher-results/hulhe')
    parser.add_argument('--service-account', default='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--submit-prepared', action='store_true')
    args = vars(parser.parse_args())
    execute = args.pop('execute')
    submit_prepared = args.pop('submit_prepared')
    if submit_prepared:
        if args['run_name']:
            parser.error('Use --submit-prepared with --output only')
        submit(args['output'])
    else:
        if not args['run_name']:
            parser.error('Preparing requires --run-name')
        print(json.dumps(build_bundle(**args), indent=2))
        if execute:
            submit(args['output'])


if __name__ == '__main__':
    main()
