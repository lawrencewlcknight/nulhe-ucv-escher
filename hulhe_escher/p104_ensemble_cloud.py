"""Independent P104 ensemble Batch job; source read-only and no self-play."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

from .checkpointing import sha256_file
from .experiment_runtime import run_child
from .fo2_setup import prepare as prepare_fo2
from .p104_cloud import storage
from .p104_protocol import SOURCE_URI, POLICY_SHA256, ITERATION, TRAINER_FINGERPRINT
from .policy_replay_artifacts import prepare, read_metadata
from .train import write_json


def prepare_inputs(root):
    source = root/'source'
    source.mkdir()
    state = source/f'state_{ITERATION:06d}'
    prepare(SOURCE_URI+f'/training/state_{ITERATION:06d}', state)
    policy = source/'P104.pkl'
    storage('cp', SOURCE_URI+'/P104.pkl', policy)
    _, metadata = read_metadata(state)
    if (sha256_file(policy) != POLICY_SHA256 or metadata['solver']['num_iteration'] != ITERATION
            or metadata['repository_commit'] != TRAINER_FINGERPRINT):
        raise ValueError('Pinned P104 identity or corrected trainer fingerprint mismatch')
    provenance = dict(source_uri=SOURCE_URI, policy_sha256=POLICY_SHA256, source_iteration=ITERATION,
        state_manifest_sha256=sha256_file(state/'manifest.json'), corrected_trainer=TRAINER_FINGERPRINT,
        source_read_only=True, policy_only_download=True, collection_iterations=0)
    return state, policy, provenance


def summary(output):
    fit = json.loads((output/'fit/experiment.json').read_text())
    evaluation = json.loads((output/'evaluation/report.json').read_text())
    write_json(output/'summary.json', dict(status='COMPLETE', fit=fit, evaluation=evaluation))
    lines = ['# P104 twenty-network ensemble', '',
        'Equal averaging of legal-action probabilities from all 20 all-replay refits.',
        'These results combine ensembling and all-replay export; they do not isolate ensembling.', '',
        '| Comparison | bb/100 | Multiplicity-adjusted interval | Conclusion |',
        '|---|---:|---|---|']
    for name, result in evaluation['results'].items():
        lines.append(f"| {name} | {result['bb_per_100']:+.3f} | "
            f"{100*result['confidence']:.2f}% [{result['interval'][0]:+.3f}, {result['interval'][1]:+.3f}] | "
            f"{result['conclusion']} |")
    lines += ['', f"Confirmation groups: {evaluation['completed_groups']:,}; pilot excluded.",
        f"Power target capped: {evaluation['plan']['capped']}.",
        'Intervals condition on this fixed ensemble and replay, not independent training runs.', '',
        '| Member seed | Selected updates | Selection seconds | Refit seconds |', '|---|---:|---:|---:|']
    for row in fit['members']:
        lines.append(f"| {row['fit_seed']} | {row['selected_update']} | "
            f"{row['selection']['fit_seconds']:.1f} | {row['fit']['fit_seconds']:.1f} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def run_cloud(workspace, output_uri):
    output_uri = output_uri.rstrip('/')
    if output_uri == SOURCE_URI or output_uri.startswith(SOURCE_URI+'/') or SOURCE_URI.startswith(output_uri+'/'):
        raise ValueError('Output must not overlap the source run')
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    output = root/'results'
    output.mkdir()
    started = time.monotonic()
    deadline = started + 11*3600
    status = dict(status='PREPARING', output_uri=output_uri, source_uri=SOURCE_URI,
                  source_iteration=ITERATION, members=20, collection_iterations=0)
    stop = threading.Event()

    def stage(name):
        status.update(status=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'cloud_status.json', status)

    def sync(final=False):
        # Periodically preserve member exports/receipts; no source replay is uploaded.
        exclude = r'^cloud_status\.json$' + ('' if final else r'|\.jsonl$')
        storage('rsync', '--recursive', '--exclude', exclude, output, output_uri)
        storage('cp', output/'cloud_status.json', output_uri+'/cloud_status.json')

    def publish():
        while not stop.wait(120):
            try:
                sync()
            except Exception as exc:
                print(f'Periodic upload failed: {exc}; mandatory final upload remains.', flush=True)

    stage('PREPARING')
    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    try:
        state, policy, provenance = prepare_inputs(root)
        write_json(output/'provenance.json', provenance)
        assets = root/'fo2-assets'
        prepare_fo2(assets)

        def child(name, module, args, cap, report_name):
            remaining = deadline-time.monotonic()
            if remaining < 120:
                raise TimeoutError('Experiment budget exhausted before '+name)
            budget = min(cap, remaining-60)
            stage(name.upper())
            cmd = [sys.executable, '-m', module, *map(str,args), '--max-seconds', str(budget-30)]
            result = run_child(cmd, output/f'{name}.log', timeout=budget)
            if result['timed_out'] or result['returncode'] != 0:
                raise RuntimeError(name+' failed or exceeded its stage guard; inspect partial reports')
            if json.loads((output/name/report_name).read_text())['status'] != 'COMPLETE':
                raise RuntimeError(name+' did not complete')

        child('fit', 'hulhe_escher.p104_ensemble',
            ['--state',state,'--policy',policy,'--output',output/'fit','--threads',8],9000,'experiment.json')
        child('evaluation', 'hulhe_escher.p104_ensemble_evaluate',
            ['--manifest',output/'fit/ensemble.json','--policy',policy,'--output',output/'evaluation',
             '--assets',assets,'--workers',8],32400,'report.json')
        summary(output)
        stage('COMPLETE')
    except BaseException as exc:
        status['reason'] = f'{type(exc).__name__}: {exc}'
        stage('FAILED')
        raise
    finally:
        stop.set()
        publisher.join(timeout=1900)
        sync(final=True)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output-uri', required=True)
    run_cloud(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
