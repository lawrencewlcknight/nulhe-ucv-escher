#!/usr/bin/env python3
"""Prepare the P56 versus corrected P32 Batch job; --execute submits it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from batch_runtime import bootstrap, job_spec
from launch_learning_check import source_archive


def build_bundle(output, *, run_name, project='clever-overview-399515', region='europe-west1',
                 bucket='gs://clever-overview-399515-fhp-escher-results/hulhe',
                 service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?', bucket):
        raise ValueError('Invalid bucket')
    uri = bucket.rstrip('/')+'/'+run_name
    repo = Path(__file__).resolve().parents[1]
    sources = json.loads((repo/'hulhe_escher/data/corrected_comparison_sources.json').read_text())
    parents = {r['uri'].split('/fit/')[0] if '/fit/' in r['uri'] else r['uri'].rsplit('/', 1)[0]
               for r in sources['files'].values()}
    if any(uri == p or uri.startswith(p+'/') or p.startswith(uri+'/') for p in parents):
        raise ValueError('Output overlaps a source experiment')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(repo, output/'source.tar.gz')
    digest = hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command = ['python', '-m', 'hulhe_escher.corrected_comparison_cloud',
               '--workspace', '/workspace/corrected-comparison', '--output-uri', uri]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=50, max_run_seconds=10800,
                   labels=dict(experiment='hulhe-p56-vs-p32-corrected', checkpoint='p56'))
    plan = dict(job_name=run_name, project=project, region=region, uri=uri, source_sha256=digest,
        sources=sources, machine='c4-standard-16', workers=8, retraining=False, replay_download=False,
        primary='P56 win rate against both corrected P32 refits, equally weighted',
        fit_seeds=[1701, 1702], pilot_seed=619501, confirmation_seed=619502,
        pilot_groups=10000, minimum_groups=100000, maximum_groups=1000000,
        alpha=.05, power=.8, target_effect_bb100=2., evaluation_seconds=7200,
        process_guard_seconds=7500, max_task_hours=3, automatic_retries=0,
        unit='Four-hand shared deal group; seat and refit averaging before uncertainty calculation',
        cleanup='Batch releases VM and disk; source experiments remain unchanged')
    for name, value in [('launch.json', plan), ('job.json', job)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    (output/'batch.sh').write_text(script)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project', default='clever-overview-399515')
    parser.add_argument('--region', default='europe-west1')
    parser.add_argument('--bucket', default='gs://clever-overview-399515-fhp-escher-results/hulhe')
    parser.add_argument('--service-account', default='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com')
    parser.add_argument('--execute', action='store_true')
    args = vars(parser.parse_args())
    execute = args.pop('execute')
    plan = build_bundle(**args)
    print(json.dumps(plan, indent=2))
    if execute:
        for name in ('source.tar.gz', 'launch.json', 'job.json'):
            subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0',
                str(args['output']/name), plan['uri']+'/'+name], check=True)
        subprocess.run(['gcloud', 'batch', 'jobs', 'submit', plan['job_name'], '--project', plan['project'],
                        '--location', plan['region'], '--config', str(args['output']/'job.json')], check=True)


if __name__ == '__main__':
    main()
