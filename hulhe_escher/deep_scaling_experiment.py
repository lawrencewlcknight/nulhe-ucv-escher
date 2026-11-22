"""Fresh deep32 training, or bounded evaluation of immutable deep/wide policies."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from .checkpointing import sha256_file, load_checkpoint_payload
from .experiment_runtime import run_child
from .scaling_config import (deep_scaling_config, scaling_config, CHECKPOINT_HOURS,
                             DEEP_COMPARISON_CAVEAT)
from .scaling_experiment import transfer, sync_analysis, memory_gate, train_stages, evaluate
from .train import write_json, source_fingerprint


def fetch_policies(root, uri, expected_source, *, deep=False):
    """Freeze policies against publication hashes; missing exports defer evaluation.

    Permission/network failures are errors, never interpreted as unfinished
    training. Published config, seed, source archive and objective-mass receipts
    must agree with the planned control before any head-to-head test starts.
    """
    root.mkdir(parents=True, exist_ok=True)
    transfer('cp', uri + '/launch.json', root / 'source-launch.json')
    plan = json.loads((root / 'source-launch.json').read_text())
    expected_config = deep_scaling_config() if deep else scaling_config()
    if (plan['source_sha256'] != expected_source or plan['config'] != expected_config
            or plan['seed'] != 20261115 or plan['collection_workers'] != 8
            or plan['training_threads'] != 8 or plan['machine'] != 'c4-standard-16'
            or plan['checkpoint_hours'] != list(CHECKPOINT_HOURS)):
        raise ValueError('Comparison run does not match the pinned source/configuration/hardware')
    receipts = {}
    for hour in CHECKPOINT_HOURS:
        label = f'P{hour}'
        try:
            transfer('cp', uri + f'/{label}.json', root / f'{label}.json')
        except subprocess.CalledProcessError as exc:
            message = exc.stderr or ''
            if re.search(r'(No URLs matched|matched no objects|does not exist|HTTPError:\s*404)', message):
                return dict(status='WAITING_FOR_POLICIES', missing=label, source_uri=uri)
            raise
        record = json.loads((root / f'{label}.json').read_text())
        if record.get('smoke') or record.get('experimental_smoke') or record['active_seconds'] < hour*3600:
            raise ValueError('Comparison policy is smoke or below its training-time target')
        fit = record['metrics']['policy_validation']
        if (fit['group_sampling'] != 'objective_mass'
                or fit['policy_sampling_seed'] != [20261115, record['outer_iteration']]):
            raise ValueError('Comparison policy lacks corrected policy-group sampling')
        transfer('cp', uri + f'/{label}.pkl', root / f'{label}.pkl')
        if sha256_file(root / f'{label}.pkl') != record['sha256']:
            raise ValueError('Comparison policy checksum mismatch')
        payload = load_checkpoint_payload(root / f'{label}.pkl')
        if (payload['training_config'] != expected_config or payload['seed'] != 20261115
                or payload['outer_iteration'] != record['outer_iteration']):
            raise ValueError('Policy payload differs from its experiment contract')
        receipts[label] = dict(sha256=record['sha256'], outer_iteration=record['outer_iteration'],
                               active_seconds=record['active_seconds'])
    result = dict(status='READY', source_uri=uri, source_sha256=expected_source, policies=receipts)
    write_json(root / 'verified-inputs.json', result)
    return result


def deep_matches(root):
    return [('Deep-P32-vs-Wide-P32', 'P32', root / 'wide/P32.pkl', .05, .5),
            ('Deep-P32-vs-Deep-P8', 'P32', root / 'P8.pkl', .01, .1),
            *[(f'Deep-P{h}-vs-Wide-P{h}', f'P{h}', root / f'wide/P{h}.pkl', .01, .1)
              for h in CHECKPOINT_HOURS[:-1]]]


def evaluate_deep(root, *, workers=8, seconds=43200, smoke=False):
    # Reserve ten minutes of the existing twelve-hour envelope for the same
    # fixed strategic-position library applied to both final policies.
    audit_budget = min(600, seconds * .1)
    started = time.monotonic()
    report = evaluate(root, workers=workers, seconds=seconds-audit_budget, smoke=smoke,
                      matches=deep_matches(root), comparison_caveat=DEEP_COMPARISON_CAVEAT)
    command = [sys.executable, '-m', 'hulhe_escher.strategic_audit', 'run',
               '--policy', 'Deep-P32='+str(root/'P32.pkl'),
               '--policy', 'Wide-P32='+str(root/'wide/P32.pkl'),
               '--require-suit-invariance', '--threads', '1', '--max-seconds', str(max(.1,audit_budget-15)),
               '--output', str(root/'strategic-audit')]
    if smoke:
        command.append('--smoke')
    remaining = seconds-(time.monotonic()-started)
    outcome = (run_child(command, root/'strategic-audit.log', timeout=remaining)
               if remaining > 1 else dict(returncode=None, timed_out=True))
    summary = root/'strategic-audit/summary.json'
    report['strategic_audit'] = dict(process=outcome,
        summary=json.loads(summary.read_text()) if summary.exists() else {'status':'INCOMPLETE'})
    if (outcome['returncode'] != 0 or outcome.get('timed_out')
            or report['strategic_audit']['summary']['status'] != 'COMPLETE'):
        report['status'] = 'INCOMPLETE'
    write_json(root/'evaluation.json', report)
    return report


def run(root, *, upload_uri=None, wide_uri=None, wide_source_sha256=None,
        deep_uri=None, deep_source_sha256=None, smoke=False):
    if upload_uri:
        upload_uri = upload_uri.rstrip('/')
        for source in (wide_uri, deep_uri):
            if source and (source.rstrip('/') == upload_uri
                    or upload_uri.startswith(source.rstrip('/')+'/')
                    or source.rstrip('/').startswith(upload_uri+'/')):
                raise ValueError('Output prefix must be independent of both input runs')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if smoke and (upload_uri or deep_uri):
        raise ValueError('Smoke is offline and cannot publish or fetch cloud policies')
    workers = threads = 1 if smoke else 8
    config_path = root/'scaling-config.json'
    write_json(config_path, deep_scaling_config(small=smoke))
    stop = threading.Event()
    def publish():
        while not stop.wait(600):
            try:
                sync_analysis(root, upload_uri)
            except Exception:
                pass  # Final synchronous publication is mandatory.
    publisher = threading.Thread(target=publish, daemon=True) if upload_uri else None
    if publisher:
        publisher.start()
    try:
        if deep_uri:
            inputs = fetch_policies(root, deep_uri, deep_source_sha256, deep=True)
            if inputs['status'] != 'READY':
                raise RuntimeError('Evaluation-only requires completed deep checkpoints')
        else:
            if upload_uri:
                transfer('cp', '--if-generation-match=0', '/workspace/source.tar.gz',
                         upload_uri+'/training_source.tar.gz')
            memory_gate(root, config_path, workers=workers, threads=threads, small=smoke)
            train_stages(root, config_path, workers=workers, threads=threads, smoke=smoke,
                         upload_uri=upload_uri, comparison_caveat=DEEP_COMPARISON_CAVEAT)
            write_json(root/'resume.json', dict(training_source_archive='training_source.tar.gz',
                training_manifest='training/manifest.json', latest_pointer='training/latest.json',
                config='scaling-config.json', source_fingerprint=source_fingerprint(),
                instructions='Resume only full state with the exact archived source/config/runtime and worker contract.'))
        if not smoke:
            inputs = fetch_policies(root/'wide', wide_uri, wide_source_sha256)
            write_json(root/'comparison-inputs.json', inputs)
            if inputs['status'] != 'READY':
                write_json(root/'evaluation.json', dict(status='DEFERRED_WIDE_NOT_READY',
                    reason=inputs, next_step='Submit a separate evaluation-only Batch after wide checkpoints are ready. Training is complete; do not restart it.'))
                return
            result = evaluate_deep(root, workers=workers)
            if result['status'] != 'COMPLETE':
                raise RuntimeError('Training preserved; evaluations incomplete, see evaluation.json')
    except Exception as exc:
        write_json(root/'failure.json', dict(error_type=type(exc).__name__, reason=str(exc)))
        raise
    finally:
        stop.set()
        if publisher:
            publisher.join()
            sync_analysis(root, upload_uri)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    for arg in ('upload-uri', 'wide-uri', 'wide-source-sha256', 'deep-uri', 'deep-source-sha256'):
        p.add_argument('--'+arg)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not args.smoke and not all((args.upload_uri, args.wide_uri, args.wide_source_sha256)):
        p.error('Production requires output prefix and pinned wide source')
    if bool(args.deep_uri) != bool(args.deep_source_sha256):
        p.error('Evaluation-only requires deep URI and source hash together')
    options = vars(args)
    output = options.pop("output")
    run(output, **options)


if __name__ == '__main__':
    main()
