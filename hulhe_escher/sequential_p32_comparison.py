"""Four fixed-policy P32 tests of the protected sequential network."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .checkpointing import load_checkpoint_payload, sha256_file
from .evaluation_stats import estimate, sample_plan
from .game import serialisable_game_definition
from .learning_evaluate import collect, initialise, validate_design
from .sequential_training import sequential_config
from .train import write_json

SOURCES = Path(__file__).with_name('data')/'sequential_p32_comparison_sources.json'
SEEDS = (1701, 1702)
CANDIDATE = 'Sequential-P32'
COMPARATORS = ('Small-P32-corrected', 'Wide-P32', 'Deep-P32', 'Hybrid-P32')
DEAL_SEED = 927321
SMOKE_SEED = 927921
ALPHA = .05/4
MODEL_TYPES = {
    CANDIDATE: 'structured_hulhe_protected_sequential_v1',
    'Wide-P32': 'structured_hulhe_mlp_v2',
    'Deep-P32': 'structured_hulhe_deep_residual_v1',
    'Hybrid-P32': 'structured_hulhe_hybrid_residual_v1',
}


def verify_inputs(inputs):
    """Check every pinned byte before unpickling, then check policy provenance."""
    root = Path(inputs)
    sources = json.loads(SOURCES.read_text())
    if sources['fit_seeds'] != list(SEEDS):
        raise ValueError('Both predeclared corrected P32 refits are required')
    for name, record in sources['files'].items():
        if sha256_file(root/name) != record['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
    fit = json.loads((root/'P32-fit.json').read_text())
    if fit['status'] != 'COMPLETE' or fit['smoke'] or fit['source_iteration'] != 1132:
        raise ValueError('Expected completed production corrected P32 refits')
    labels = (CANDIDATE, *COMPARATORS[1:], *(f'P32-corrected-{s}' for s in SEEDS))
    for seed in SEEDS:
        rows = [r for r in fit['results'] if r['arm'] == 'objective_mass' and r['fit_seed'] == seed]
        if (len(rows) != 1 or rows[0]['fit']['group_sampling'] != 'objective_mass'
                or rows[0]['sha256'] != sources['files'][f'P32-corrected-{seed}.pkl']['sha256']):
            raise ValueError('Corrected refit provenance mismatch')
    payloads = {label: load_checkpoint_payload(root/(label+'.pkl')) for label in labels}
    for label, payload in payloads.items():
        pin = sources['policies'][label]
        if (payload['outer_iteration'] != pin['iteration']
                or payload['training_elapsed_seconds'] != pin['active_seconds']
                or payload['game'] != serialisable_game_definition()):
            raise ValueError(f'{label} checkpoint identity/game mismatch')
        for field in ('game', 'feature_encoder', 'input_size', 'num_actions'):
            if payload[field] != payloads[CANDIDATE][field]:
                raise ValueError(f'Policies are not matched on {field}')
        if label in MODEL_TYPES and payload.get('policy_model', {}).get('type') != MODEL_TYPES[label]:
            raise ValueError(f'{label} architecture mismatch')
        if label.startswith('P32-corrected-') and pin['iteration'] != 1132:
            raise ValueError('Corrected smaller P32 iteration mismatch')

    for label in COMPARATORS[1:]:
        plan = json.loads((root/(label+'-launch.json')).read_text())
        training = json.loads((root/(label+'-scaling.json')).read_text())
        marker = training['checkpoints']['P32']
        payload = payloads[label]
        if (training['status'] != 'TRAINING_COMPLETE' or training['smoke']
                or not training['fresh_initialization'] or not training['final_recovery_verified']
                or payload['training_config'] != plan['config'] or training['config'] != plan['config']
                or payload['seed'] != plan['seed'] or training['seed'] != plan['seed']):
            raise ValueError(f'{label} production configuration mismatch')
        verify_marker(label, marker, payload, sources)
        if label == 'Hybrid-P32' and marker != json.loads((root/'Hybrid-P32.json').read_text()):
            raise ValueError('Hybrid P32 publication marker mismatch')

    marker = json.loads((root/'Sequential-P32.json').read_text())
    provenance = json.loads((root/'Sequential-provenance.json').read_text())
    receipt = json.loads((root/'Sequential-sampler-verification.json').read_text())
    payload = payloads[CANDIDATE]
    verify_marker(CANDIDATE, marker, payload, sources)
    policy_fit = marker['metrics']['policy_validation']
    protection = policy_fit.get('protected_training', {})
    if (payload['training_config'] != sequential_config() or payload['seed'] != 20261115
            or policy_fit['policy_sampling_seed'] != [20261115, payload['outer_iteration']]
            or protection.get('protocol') != 'protected_selective_v1' or protection.get('coefficient') != .3
            or receipt['status'] != 'VERIFIED' or receipt['architecture'] != 'protected_sequential_v1'
            or receipt['policy_group_sampling'] != 'objective_mass' or receipt['protection_coefficient'] != .3
            or receipt['source_modified'] or receipt['config_changed']
            or not provenance['sequential_source'] or provenance['policy_group_sampling'] != 'objective_mass'
            or provenance['training_source_fingerprint'] != receipt['training_source_fingerprint']):
        raise ValueError('Sequential P32 protected/corrected production provenance mismatch')
    return {label: dict(kind='checkpoint', checkpoint=str((root/(label+'.pkl')).resolve()))
            for label in labels}, sources


def verify_marker(label, marker, payload, sources):
    pin = sources['policies'][label]
    if (marker['label'] != 'P32' or marker['smoke'] or marker.get('experimental_smoke', False)
            or marker['outer_iteration'] != pin['iteration'] or marker['active_seconds'] != pin['active_seconds']
            or marker['active_seconds'] < 32*3600 or marker['active_seconds'] != payload['training_elapsed_seconds']
            or marker['sha256'] != sources['files'][label+'.pkl']['sha256']
            or marker['metrics']['policy_validation']['group_sampling'] != 'objective_mass'):
        raise ValueError(f'{label} corrected production P32 marker mismatch')


def matchups():
    return [(f'Sequential-vs-Small-{seed}', CANDIDATE, f'P32-corrected-{seed}', 100*(i+1))
            for i, seed in enumerate(SEEDS)] + [
        (f'Sequential-vs-{name.removesuffix("-P32")}', CANDIDATE, name, 100*(i+3))
        for i, name in enumerate(COMPARATORS[1:])]


def observations(rows, comparator):
    if comparator not in COMPARATORS:
        raise ValueError(f'Unknown comparator: {comparator}')
    if comparator == COMPARATORS[0]:
        # Four correlated hands on one deal produce ONE observation, not four.
        return np.asarray([np.mean([row[f'Sequential-vs-Small-{s}']['mean_bb'] for s in SEEDS])
                           for row in rows])
    name = 'Sequential-vs-'+comparator.removesuffix('-P32')
    return np.asarray([row[name]['mean_bb'] for row in rows])


def common_plan(pilot, rate, remaining, *, minimum, maximum):
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('Pilot throughput must be positive and finite')
    plans = {name: sample_plan(observations(pilot, name), minimum=minimum, maximum=maximum,
             effect_bb100=2., alpha=ALPHA, power=.8) for name in COMPARATORS}
    selected = max(p['selected'] for p in plans.values())
    for plan in plans.values():
        plan['estimated_detectable_effect_bb100'] *= np.sqrt(plan['selected']/selected)
        plan.update(selected=selected, capped=plan['required'] > selected)
    expected = selected/rate
    return dict(selected=selected, by_comparator=plans, pilot_groups_per_second=rate,
        expected_confirmation_seconds=expected, full_selected_sample_fits=expected/.8 <= remaining,
        pilot_excluded=True, sample_not_reduced_for_runtime=True, alpha_per_test=ALPHA,
        familywise_alpha=.05,
        rule='Maximum of four pilot requirements, capped at one million groups; no outcome stopping.')


def render(root, report):
    lines = ['# Sequential P32 versus corrected Small, Wide, Deep and Hybrid', '',
        f"Status: {report['status']}", '', 'Positive bb/100 favours Sequential P32.',
        'Four predeclared tests use 98.75% intervals (Bonferroni familywise alpha 5%).',
        'Both corrected smaller refits are equally weighted within each shared deal.', '',
        '| Opponent | Sequential bb/100 | 98.75% interval | Conclusion |', '|---|---:|---|---|']
    for name, r in report['results'].items():
        lines.append(f"| {name} | {r['bb_per_100']:+.3f} | [{r['interval'][0]:+.3f}, {r['interval'][1]:+.3f}] | {r['conclusion']} |")
    lines += ['', f"Confirmation groups: {report['completed_groups']:,}; hands: {report['completed_hands']:,}.",
        'Each group contains ten hands: five pairings with seats swapped. Pilots are excluded.',
        'Per-refit smaller-policy results are descriptive; neither refit is selected or discarded.',
        'Power planning targets 80% estimated power for 2 bb/100 per primary comparison.',
        'The smaller policies were refitted after historical training; the other recipes used corrected sampling.',
        'Equal active training hours do not imply equal iterations, compute or training recipes.',
        'These tests compare saved policies, not architecture alone, training-seed variance or exploitability.',
        'Only the sequential policy is compared with the four alternatives; this is not a full round robin.',
        'No retraining, replay download or Fell Omen 2 evaluation.']
    if report.get('plan'):
        lines += ['Power-capped comparisons: '+(', '.join(k for k, v in report['plan']['by_comparator'].items()
                                                       if v['capped']) or 'none')+'.']
    if report['smoke']:
        lines += ['', 'SMOKE ONLY: not evidence of playing strength.']
    (root/'comparison_report.md').write_text('\n'.join(lines)+'\n')


def run_evaluation(inputs, output, *, workers=8, max_seconds=21600, smoke=False):
    pilot_count, minimum, maximum = (4, 8, 8) if smoke else (10000, 100000, 1000000)
    seed = SMOKE_SEED if smoke else DEAL_SEED
    validate_design(seed, workers, pilot_count, minimum, maximum, max_seconds, 2.)
    specs, sources = verify_inputs(inputs)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(sources=sources, policies=specs, matchups=matchups(),
        candidate=CANDIDATE, comparators=list(COMPARATORS), fit_seeds=list(SEEDS),
        game=serialisable_game_definition(), pilot_seed=seed, confirmation_seed=seed+1, workers=workers,
        pilot_groups=pilot_count, minimum_groups=minimum, maximum_groups=maximum, max_seconds=max_seconds,
        alpha_per_test=ALPHA, familywise_alpha=.05, power=.8, target_effect_bb100=2.,
        unit='Ten-hand shared deal group; average seats, then both small refits within each deal.',
        collection=False, retraining=False, smoke=smoke))
    report = dict(status='STARTING', smoke=smoke, results={}, by_fit_seed={},
                  completed_groups=0, completed_hands=0)
    write_json(root/'report.json', report)
    started = time.monotonic()
    deadline, main = started+max_seconds, []
    try:
        with mp.get_context('spawn').Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            tick = time.monotonic()
            pilot = collect(pool, count=pilot_count, seed=seed, matchups=matchups(),
                path=root/'pilot.jsonl', deadline=deadline, workers=workers)
            if len(pilot) != pilot_count:
                report['status'] = 'INCOMPLETE_PILOT'
            else:
                plan = common_plan(pilot, pilot_count/(time.monotonic()-tick),
                    deadline-time.monotonic(), minimum=minimum, maximum=maximum)
                report['plan'] = plan
                write_json(root/'plan.json', plan)
                if not plan['full_selected_sample_fits']:
                    report['status'] = 'INSUFFICIENT_TIME_FOR_SELECTED_SAMPLE'
                else:
                    report['status'] = 'CONFIRMATION'
                    write_json(root/'report.json', report)
                    collect(pool, count=plan['selected'], seed=seed+1, matchups=matchups(),
                        path=root/'confirmatory.jsonl', deadline=deadline, workers=workers, rows=main)
                    report['status'] = 'COMPLETE' if len(main) == plan['selected'] else 'INCOMPLETE_TIME_LIMIT'
        verify_inputs(inputs)
    except Exception as exc:
        report.update(status='EVALUATION_ERROR', reason=f'{type(exc).__name__}: {exc}')
    finally:
        if len(main) >= 2:
            for name in COMPARATORS:
                result = estimate(observations(main, name), alpha=ALPHA,
                                  complete=report['status'] == 'COMPLETE' and not smoke)
                if smoke:
                    result['conclusion'] = 'smoke_only'
                report['results'][name] = result
            for fit_seed in SEEDS:
                result = estimate([row[f'Sequential-vs-Small-{fit_seed}']['mean_bb'] for row in main],
                                  complete=False)
                result['conclusion'] = 'smoke_only' if smoke else 'descriptive'
                report['by_fit_seed'][str(fit_seed)] = result
        report.update(completed_groups=len(main), completed_hands=10*len(main),
                      elapsed_seconds=time.monotonic()-started)
        write_json(root/'report.json', report)
        render(root, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=21600)
    parser.add_argument('--smoke', action='store_true')
    if run_evaluation(**vars(parser.parse_args()))['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
