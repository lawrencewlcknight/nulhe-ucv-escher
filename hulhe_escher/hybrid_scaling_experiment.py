"""Fresh hybrid32 training with verified recovery and bounded frozen-policy tests."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

from .checkpointing import sha256_file
from .deep_scaling_experiment import fetch_policies
from .experiment_runtime import run_child
from .scaling_config import hybrid_scaling_config, CHECKPOINT_HOURS, HYBRID_COMPARISON_CAVEAT
from .scaling_corrected_comparison import SOURCES, verify_inputs
from .scaling_corrected_cloud import validate_output
from .scaling_experiment import transfer, sync_analysis, memory_gate, train_stages, evaluate
from .train import write_json, source_fingerprint


def prepare_inputs(root):
    inputs = root/'inputs'
    inputs.mkdir()
    sources = json.loads(SOURCES.read_text())
    for name, record in sources['files'].items():
        transfer('cp', record['uri'], inputs/name)
        if sha256_file(inputs/name) != record['sha256']:
            raise ValueError('Pinned comparison checksum mismatch: '+name)
    verify_inputs(inputs)
    write_json(root/'comparison-inputs.json', sources)
    # Assets and Java bridge must work before spending 32 hours training.
    result = run_child([sys.executable, '-m', 'hulhe_escher.fo2_setup',
                        '--assets', str(root/'fo2-assets')], root/'fo2-setup.log', timeout=1800)
    if result['returncode'] != 0 or result.get('timed_out'):
        raise RuntimeError('Fell Omen 2 setup failed before training')


def progress_matches(root):
    return [('Hybrid-P32-vs-P8', 'P32', root/'P8.pkl', .01, .2),
        *[(f'Hybrid-P{b}-vs-P{a}', f'P{b}', root/f'P{a}.pkl', .01, .2)
          for a, b in zip(CHECKPOINT_HOURS, CHECKPOINT_HOURS[1:])]]


def evaluate_hybrid(root, *, workers=8, seconds=43200):
    started = time.monotonic()
    report = dict(status='EVALUATING', primary='hybrid_vs_deep',
                  comparison_caveat=HYBRID_COMPARISON_CAVEAT)
    # Nine hours for external benchmarks, 2h50 for progression, ten minutes audit.
    budget = seconds*.75
    result = run_child([sys.executable, '-m', 'hulhe_escher.hybrid_evaluate',
        '--policy', str(root/'P32.pkl'), '--inputs', str(root/'inputs'),
        '--assets', str(root/'fo2-assets'), '--output', str(root/'benchmarks'),
        '--workers', str(workers), '--max-seconds', str(max(.1, budget-30))],
        root/'benchmarks.log', timeout=budget)
    path = root/'benchmarks/report.json'
    report['benchmarks'] = dict(process=result,
        report=json.loads(path.read_text()) if path.exists() else dict(status='INCOMPLETE'))
    audit_budget = min(600, seconds*.1)
    remaining = seconds-(time.monotonic()-started)-audit_budget
    progress_budget = min(seconds*.25-audit_budget, remaining)
    report['progression'] = (evaluate(root, workers=workers, seconds=progress_budget,
        matches=progress_matches(root), comparison_caveat=HYBRID_COMPARISON_CAVEAT, seed_base=743001)
        if progress_budget > 1 else dict(status='INCOMPLETE'))
    # Preserve all stages in the coordinator report after evaluate's standalone report.
    write_json(root/'evaluation.json', report)
    command = [sys.executable, '-m', 'hulhe_escher.strategic_audit', 'run',
        '--policy', 'Hybrid-P32='+str(root/'P32.pkl'),
        '--policy', 'Deep-P32='+str(root/'inputs/Deep-P32.pkl'),
        '--policy', 'Wide-P32='+str(root/'inputs/Wide-P32.pkl'),
        '--require-suit-invariance', '--threads', '1',
        '--max-seconds', str(max(.1, audit_budget-15)), '--output', str(root/'strategic-audit')]
    remaining = min(audit_budget, seconds-(time.monotonic()-started))
    outcome = (run_child(command, root/'strategic-audit.log', timeout=remaining)
               if remaining > 1 else dict(returncode=None, timed_out=True))
    path = root/'strategic-audit/summary.json'
    report['strategic_audit'] = dict(process=outcome,
        report=json.loads(path.read_text()) if path.exists() else dict(status='INCOMPLETE'))
    ok = report['progression']['status'] == 'COMPLETE'
    for name in ('benchmarks', 'strategic_audit'):
        stage = report[name]
        ok &= (stage['report']['status'] == 'COMPLETE'
               and stage['process']['returncode'] == 0 and not stage['process'].get('timed_out'))
    report.update(status='COMPLETE' if ok else 'INCOMPLETE', elapsed_seconds=time.monotonic()-started)
    write_json(root/'evaluation.json', report)
    return report


def run(root, *, upload_uri=None, training_uri=None, training_source_sha256=None, smoke=False):
    if bool(training_uri) != bool(training_source_sha256):
        raise ValueError('Evaluation-only requires training URI and source hash together')
    if smoke and (upload_uri or training_uri):
        raise ValueError('Smoke is offline and cannot publish or fetch cloud policies')
    if not smoke and not upload_uri:
        raise ValueError('Production requires a cloud output prefix')
    if upload_uri:
        upload_uri = validate_output(upload_uri)
        if training_uri and (upload_uri == training_uri.rstrip('/')
                or upload_uri.startswith(training_uri.rstrip('/')+'/')
                or training_uri.rstrip('/').startswith(upload_uri+'/')):
            raise ValueError('Evaluation output must be independent of training')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    workers = threads = 1 if smoke else 8
    config_path = root/'scaling-config.json'
    write_json(config_path, hybrid_scaling_config(small=smoke))
    stop = threading.Event()
    def publish():
        while not stop.wait(600):
            try:
                sync_analysis(root, upload_uri)
            except Exception as exc:
                print(f'Periodic publication failed: {exc}; final publication remains mandatory.', flush=True)
    publisher = threading.Thread(target=publish, daemon=True) if upload_uri else None
    if publisher:
        publisher.start()
    try:
        if not smoke:
            prepare_inputs(root)
        if training_uri:
            inputs = fetch_policies(root, training_uri, training_source_sha256,
                                    expected_config=hybrid_scaling_config())
            if inputs['status'] != 'READY':
                raise RuntimeError('Evaluation-only requires completed hybrid checkpoints')
        else:
            if upload_uri:
                transfer('cp', '--if-generation-match=0', '/workspace/source.tar.gz',
                         upload_uri+'/training_source.tar.gz')
            memory_gate(root, config_path, workers=workers, threads=threads, small=smoke)
            train_stages(root, config_path, workers=workers, threads=threads, smoke=smoke,
                         upload_uri=upload_uri, comparison_caveat=HYBRID_COMPARISON_CAVEAT)
            write_json(root/'resume.json', dict(training_source_archive='training_source.tar.gz',
                training_manifest='training/manifest.json', latest_pointer='training/latest.json',
                config='scaling-config.json', source_fingerprint=source_fingerprint(),
                instructions='Restore full state with exact archived source/runtime/config and worker contract. Playable weights alone cannot resume training.'))
        if not smoke and evaluate_hybrid(root, workers=workers)['status'] != 'COMPLETE':
            raise RuntimeError('Training preserved; evaluations incomplete. Use a separate evaluation-only job.')
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
    for name in ('upload-uri', 'training-uri', 'training-source-sha256'):
        p.add_argument('--'+name)
    p.add_argument('--smoke', action='store_true')
    options = vars(p.parse_args())
    run(options.pop('output'), **options)


if __name__ == '__main__':
    main()
