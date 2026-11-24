"""Fresh 32-active-hour HULHE scaling run, restart checks and fixed-policy tests."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

from .checkpointing import sha256_file
from .experiment import publish_policy, upload_checkpoint
from .experiment_runtime import run_child
from .scaling_config import (scaling_config, CHECKPOINT_HOURS, HISTORICAL_P32_URI,
                             HISTORICAL_P32_SHA256, COMPARISON_CAVEAT)
from .storage_transfer import run_transfer
from .train import source_fingerprint, write_json


def transfer(*args):
    return run_transfer(['gcloud', 'storage', *map(str, args)], timeout=1800)


def sync_analysis(root, uri):
    # Explicit suffix allowlist; never copy weights, replay, archives or hand logs.
    # Ordered checkpoint publication owns P*.json and training pointers.
    exclude = (r'(^|/)(?:state_[0-9]+|state)(?:/|$)|^memory-probe-'
               r'|(?:^|/)(?:latest|manifest)\.json$|^P[0-9]+\.json$|^(?!.*\.json$).*')
    transfer('rsync', '--recursive', '--exclude', exclude, root, uri)


def memory_gate(root, config_path, *, workers, threads, small=False):
    import psutil
    command = [sys.executable, '-m', 'hulhe_escher.memory_preflight',
               '--output', str(root / 'memory-preflight.json'), '--config', str(config_path),
               '--workers', str(workers), '--threads', str(threads)]
    if small:
        command.append('--small')
    outcome = run_child(command, root / 'memory-preflight.log', timeout=7200)
    if outcome['returncode'] != 0 or outcome.get('timed_out'):
        raise RuntimeError('Full-replay memory preflight failed; training was not started')
    result = json.loads((root / 'memory-preflight.json').read_text())
    if result['policy_group_sampling'] != 'objective_mass' or not result['checkpoint_roundtrip']:
        raise RuntimeError('Memory preflight did not verify corrected sampling and recovery')
    total = psutil.virtual_memory().total
    reserve = max(8 * 1024**3, .12 * total)
    headroom = total - outcome['peak_host_unavailable_bytes']
    gate = dict(passed=bool(small or (outcome['monitoring_available'] and headroom >= reserve)),
                observed_headroom_bytes=headroom, required_headroom_bytes=reserve,
                resource_measurements=outcome, smoke=small,
                note='Capacity gate on this VM; transient peaks may be missed by 0.5-second sampling')
    write_json(root / 'memory-gate.json', gate)
    if not gate['passed']:
        raise RuntimeError('Insufficient measured memory headroom; no automatic VM upsizing or training')


def train_stages(root, config_path, *, workers=8, threads=8, seed=20261115,
                 wall_seconds=144000, smoke=False, upload_uri=None, comparison_caveat=COMPARISON_CAVEAT):
    training = root / 'training'
    source = Path(__file__).resolve().parents[1]
    deadline = time.monotonic() + wall_seconds
    report = dict(status='TRAINING', fresh_initialization=True, config=json.loads(config_path.read_text()),
                  source_fingerprint=source_fingerprint(), seed=seed, workers=workers, threads=threads,
                  smoke=smoke, active_hours=32, labels=[f'P{h}' for h in CHECKPOINT_HOURS],
                  checkpoints={}, stages={}, comparison_caveat=comparison_caveat)
    write_json(root / 'scaling.json', report)
    previous = 0.
    try:
        for hour in CHECKPOINT_HOURS:
            label = f'P{hour}'
            target = previous + .001 if smoke else hour * 3600.
            duration = target - previous
            if duration <= 0 or time.monotonic() >= deadline:
                raise TimeoutError('Training budget exhausted or iteration crossed the next checkpoint target')
            command = [sys.executable, '-m', 'hulhe_escher.train', '--output', str(training),
                       '--config', str(config_path), '--workers', str(workers), '--threads', str(threads),
                       '--seed', str(seed), '--iterations', '1000000000',
                       '--policy-every', '0', '--policy-seconds', '1000000000',
                       '--recovery-seconds', '21600', '--additional-training-seconds', str(duration)]
            if previous:
                command.append('--resume')
            outcome = run_child(command, root / f'train-{label}.log', timeout=deadline-time.monotonic())
            report['stages'][label] = outcome
            if outcome['returncode'] != 0 or outcome.get('timed_out'):
                raise RuntimeError(f'{label} training failed; automatic retry is disabled')
            record = publish_policy(training, root, label)
            diagnostics = record['metrics']['policy_validation']
            if (diagnostics.get('group_sampling') != 'objective_mass'
                    or diagnostics.get('policy_sampling_seed') != [seed, record['outer_iteration']]):
                raise RuntimeError(f'{label} did not use the P104 objective-mass correction')
            if record['active_seconds'] < target:
                raise RuntimeError('Iteration ceiling reached before the active-time target')
            record.update(target_active_seconds=target, experimental_smoke=smoke)
            write_json(root / f'{label}.json', record)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('No budget for recovery verification')
            # Fresh interpreter: all shard checksums, solver/replay/optimizer/actor
            # restore and exact agreement between exported and restored policy.
            verified = run_child([sys.executable, str(source / 'hulhe_escher/verify_recovery.py'),
                '--source', str(source), '--training', str(training), '--policy', str(root / f'{label}.pkl'),
                '--report', str(root / f'reload-{label}.json')], root / f'reload-{label}.log',
                timeout=min(1800, remaining))
            if verified['returncode'] != 0 or verified.get('timed_out'):
                raise RuntimeError(f'{label} full recovery verification failed')
            report['checkpoints'][label] = record
            previous = record['active_seconds']
            if upload_uri:
                upload_checkpoint(root, upload_uri, label)
            write_json(root / 'scaling.json', report)
        report.update(status='TRAINING_COMPLETE', final_recovery_verified=True,
                      actual_active_seconds=previous)
        rows = [json.loads(p.read_text()) for p in sorted(training.glob('metrics_*.json'))]
        learner_seconds = sum(r['learner_seconds'] for r in rows)
        report['efficiency'] = dict(completed_iterations=rows[-1]['iteration'],
            nodes=rows[-1]['nodes'], trajectories=rows[-1]['trajectories'],
            trajectories_per_active_second=rows[-1]['trajectories']/learner_seconds,
            active_learner_seconds=learner_seconds,
            policy_fit_seconds=sum(r['policy_fit_seconds'] for r in rows),
            persistence_seconds=sum(r['persistence_seconds'] for r in rows),
            collection_and_overhead_seconds=sum(r['collection_and_overhead_seconds'] for r in rows),
            final_policy_replay=rows[-1]['policy_replay'])
    except Exception as exc:
        report.update(status='FAILED', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        write_json(root / 'scaling.json', report)
    return report


def evaluate(root, *, workers=8, seconds=43200, smoke=False, matches=None,
             comparison_caveat=COMPARISON_CAVEAT, seed_base=61001):
    # One primary historical comparison and five adjusted within-run tests.
    # Independent pilot/confirmation deals, duplicate seats, no optional stopping.
    labels = [f'P{h}' for h in CHECKPOINT_HOURS]
    if matches is None:
        matches = [('historical-P32', 'P32', root / 'historical/P32.pkl', .05, .5)]
        matches += [('P32-vs-P8', 'P32', root / 'P8.pkl', .01, .1)]
        matches += [(f'{b}-vs-{a}', b, root / f'{a}.pkl', .01, .1)
                    for a, b in zip(labels, labels[1:])]
    report = dict(status='EVALUATING', tests={}, comparison_caveat=comparison_caveat,
                  primary=matches[0][0], secondary_alpha=.05/5, smoke=smoke)
    deadline = time.monotonic() + seconds
    for index, (name, a, b, alpha, fraction) in enumerate(matches):
        budget = min(seconds * fraction, deadline-time.monotonic())
        if budget <= 1:
            report['tests'][name] = dict(status='SKIPPED_TIME_LIMIT')
            continue
        destination = root / 'evaluation' / name
        command = [sys.executable, '-m', 'hulhe_escher.learning_evaluate', '--a', str(root / f'{a}.pkl'),
                   '--b', str(b), '--output', str(destination), '--workers', str(workers),
                   '--seed', str(seed_base + index*10), '--max-seconds', str(max(.1, budget-30)),
                   '--alpha', str(alpha), '--pilot-pairs', '4' if smoke else '10000',
                   '--minimum', '8' if smoke else '100000', '--maximum', '8' if smoke else '1000000',
                   '--effect-bb100', '2']
        outcome = run_child(command, root / f'eval-{name}.log', timeout=budget)
        path = destination / 'report.json'
        result = json.loads(path.read_text()) if path.exists() else dict(status='EVALUATION_ERROR')
        result['process'] = outcome
        report['tests'][name] = result
        write_json(root / 'evaluation.json', report)
    report['status'] = ('COMPLETE' if all(t['status'] == 'COMPLETE' and
                         t.get('process', {}).get('returncode') == 0 and
                         not t.get('process', {}).get('timed_out') for t in report['tests'].values())
                        else 'INCOMPLETE')
    write_json(root / 'evaluation.json', report)
    return report


def run(root, *, upload_uri=None, workers=8, threads=8, smoke=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if smoke:
        workers = threads = 1
        if upload_uri:
            raise ValueError('Offline smoke must not publish cloud artifacts')
    config_path = root / 'scaling-config.json'
    write_json(config_path, scaling_config(small=smoke))
    stop = threading.Event()
    def publish():
        while not stop.wait(600):
            try:
                sync_analysis(root, upload_uri)
            except Exception:
                pass  # Mandatory final publication surfaces any persistent error.
    publisher = threading.Thread(target=publish, daemon=True) if upload_uri else None
    if publisher:
        publisher.start()
    try:
        if upload_uri:
            # Exact submitted source is retained for future full-state continuation.
            transfer('cp', '--if-generation-match=0', '/workspace/source.tar.gz',
                     upload_uri.rstrip('/') + '/training_source.tar.gz')
            (root / 'historical').mkdir()
            for suffix in ('json', 'pkl'):
                transfer('cp', HISTORICAL_P32_URI + f'/P32.{suffix}', root / f'historical/P32.{suffix}')
            if sha256_file(root / 'historical/P32.pkl') != HISTORICAL_P32_SHA256:
                raise ValueError('Historical P32 hash mismatch; refusing to start training')
        memory_gate(root, config_path, workers=workers, threads=threads, small=smoke)
        train_stages(root, config_path, workers=workers, threads=threads, smoke=smoke, upload_uri=upload_uri)
        write_json(root / 'resume.json', dict(training_source_archive='training_source.tar.gz',
            training_manifest='training/manifest.json', latest_pointer='training/latest.json',
            config='scaling-config.json', source_fingerprint=source_fingerprint(),
            instructions='Use the exact archived source/runtime/config and worker contract with hulhe_escher.train --resume; never resume from playable weights alone.'))
        if upload_uri:
            evaluation = evaluate(root, workers=workers)
            if evaluation['status'] != 'COMPLETE':
                raise RuntimeError('Training complete, evaluations incomplete; see evaluation.json')
    except Exception as exc:
        write_json(root / 'failure.json', dict(error_type=type(exc).__name__, reason=str(exc)))
        raise
    finally:
        stop.set()
        if publisher:
            publisher.join()
            sync_analysis(root, upload_uri)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upload-uri')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.smoke and not args.upload_uri:
        parser.error('Production requires a cloud output prefix')
    if args.workers < 1 or args.threads < 1:
        parser.error('Workers and threads must be positive')
    run(args.output, upload_uri=args.upload_uri, workers=args.workers, threads=args.threads, smoke=args.smoke)


if __name__ == '__main__':
    main()
