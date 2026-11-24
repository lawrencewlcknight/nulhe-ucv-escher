"""Fresh protected sequential P4/P8 learning check and independent FO2 worker."""
import argparse
import json
from pathlib import Path
import sys
import time

from .experiment import run_experiment, upload_recovery
from .experiment_report import render
from .experiment_runtime import run_child
from .scaling_experiment import memory_gate, sync_analysis, transfer
from .sequential_training import sequential_config, COMPARISON_CAVEAT
from .train import write_json


def run(root, *, upload_uri=None, smoke=False):
    if smoke and upload_uri:
        raise ValueError('Smoke must remain offline')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config_path = root/'sequential-config.json'
    write_json(config_path, sequential_config(small=smoke))
    write_json(root/'protocol.json', dict(active_hours=8, checkpoint_hours=[4, 8],
        initialization='fresh models, replay and optimizer; no P104 warm start',
        seed=20261115, per_network_parameters=1496067,
        comparison_caveat=COMPARISON_CAVEAT, smoke=smoke,
        primary='P8 vs P4', external='Fell Omen 2, free-fold-to-check',
        workers=1 if smoke else 8, threads=1 if smoke else 8))
    run_root = root/'run'
    try:
        memory_gate(root, config_path, workers=1 if smoke else 8, threads=1 if smoke else 8, small=smoke)
        report = run_experiment(run_root, stage_seconds=.001 if smoke else 14400,
            workers=8, threads=8, local_workers=8, seed=20261115, smoke=smoke,
            skip_preflight=True, training_wall_seconds=240 if smoke else 43200,
            evaluation_seconds=240 if smoke else 7200, upload_uri=upload_uri,
            config_path=config_path, verify_checkpoints=True)
        report['comparison_caveat'] = COMPARISON_CAVEAT
        write_json(run_root/'experiment.json', report)
        for label in ('P4', 'P8'):
            diagnostics = report['checkpoints'][label]['metrics']['policy_validation']
            if (diagnostics['group_sampling'] != 'objective_mass'
                    or diagnostics['protected_training']['protocol'] != 'protected_selective_v1'):
                raise RuntimeError('Missing corrected protected policy fitting')
        if any(row['status'] != 'COMPLETE' for row in report['evaluations'].values()):
            raise RuntimeError('Training preserved; local evaluation incomplete')
        write_json(root/'status.json', dict(status='COMPLETE', smoke=smoke,
            external='independent job' if not smoke else 'offline check/call proxy'))
        if smoke:
            # Real spawned paired benchmark plumbing, without downloading FO2 assets.
            from .benchmark import run_benchmark
            benchmark = run_benchmark(run_root, run_root/'benchmarks/fo2', opponent='check_call',
                workers=1, pilot_pairs=4, minimum=8, maximum=8, max_seconds=120)
            if benchmark['status'] != 'COMPLETE':
                raise RuntimeError('Offline benchmark proxy incomplete')
            render(run_root)
        return report
    except Exception as exc:
        write_json(root/'status.json', dict(status='FAILED', reason=str(exc)))
        if upload_uri:
            write_json(root/'training_failed.json', dict(reason=str(exc)))
            transfer('cp', root/'training_failed.json', upload_uri+'/training_failed.json')
        raise
    finally:
        if upload_uri:
            # Preserve newest full state even if an evaluation or publication failed.
            if (run_root/'training/latest.json').exists():
                upload_recovery(run_root, upload_uri)
            sync_analysis(root, upload_uri+'/coordinator')
            if run_root.exists():
                sync_analysis(run_root, upload_uri)


def run_external(root, uri):
    from .cloud_worker import external_worker
    external_worker(Path(root), uri, max_seconds=26*3600)
    root = Path(root)
    report = json.loads((root/'benchmarks/fo2/report.json').read_text())
    if report['status'] != 'COMPLETE':
        raise RuntimeError('Fell Omen 2 evaluation incomplete; inspect its report')
    # Trainer publishes experiment.json only after local evaluation. Download it
    # when available to render the same combined report as the first experiment.
    from .cloud_worker import fetch_file
    if fetch_file(uri+'/experiment.json', root/'experiment.json'):
        render(root)
        for name in ('learning_report.json', 'learning_report.md'):
            transfer('cp', root/name, uri+'/'+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upload-uri')
    parser.add_argument('--role', choices=('train', 'external'), default='train')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.role == 'external':
        if args.smoke or not args.upload_uri:
            parser.error('External worker requires a cloud prefix, without smoke')
        run_external(args.output, args.upload_uri)
    else:
        run(args.output, upload_uri=args.upload_uri, smoke=args.smoke)


if __name__ == '__main__':
    main()
