"""Evaluate fixed P56 against both saved corrected P32 refits on fresh paired deals."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .checkpointing import load_checkpoint_payload, sha256_file
from .game import serialisable_game_definition
from .learning_evaluate import collect, initialise, runtime_plan, validate_design
from .evaluation_stats import estimate
from .train import write_json

SOURCES = Path(__file__).with_name('data')/'corrected_comparison_sources.json'
SEEDS = (1701, 1702)
DEAL_SEED = 619501  # Disjoint from previous evaluations and development smoke deals.
SMOKE_SEED = 619901


def verify_inputs(inputs):
    """Verify fixed identities before deserialising any checkpoint."""
    root = Path(inputs)
    source = json.loads(SOURCES.read_text())
    for name, record in source['files'].items():
        if sha256_file(root/name) != record['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
    fit = json.loads((root/'P32-fit.json').read_text())
    final = json.loads((root/'P56.json').read_text())
    if fit['status'] != 'COMPLETE' or fit['smoke'] or fit['source_iteration'] != 1132:
        raise ValueError('Expected completed production P32 refits')
    if final['outer_iteration'] != 1943 or final['smoke']:
        raise ValueError('Expected production P56')
    if final['metrics']['policy_validation']['group_sampling'] != 'objective_mass':
        raise ValueError('P56 is not corrected')
    labels = ['P56']
    for seed in SEEDS:
        rows = [r for r in fit['results'] if r['arm'] == 'objective_mass' and r['fit_seed'] == seed]
        if len(rows) != 1 or rows[0]['fit']['group_sampling'] != 'objective_mass':
            raise ValueError('Missing corrected P32 refit')
        label = f'P32-corrected-{seed}'
        if rows[0]['sha256'] != source['files'][label+'.pkl']['sha256']:
            raise ValueError('P32 metadata/checkpoint identity mismatch')
        labels.append(label)
    if final['sha256'] != source['files']['P56.pkl']['sha256']:
        raise ValueError('P56 metadata/checkpoint identity mismatch')
    payloads = [load_checkpoint_payload(root/(label+'.pkl')) for label in labels]
    for field in ('game', 'policy_model', 'feature_encoder', 'policy_network_layers', 'input_size', 'num_actions'):
        if any(p[field] != payloads[0][field] for p in payloads[1:]):
            raise ValueError(f'Policies are not matched on {field}')
    return {label: dict(kind='checkpoint', checkpoint=str((root/(label+'.pkl')).resolve()))
            for label in labels}, source


def observations(rows):
    # Average correlated seat/fit-seed results BEFORE calculating uncertainty.
    return np.asarray([np.mean([r[f'h2h_{s}']['mean_bb'] for s in SEEDS]) for r in rows])


def render(output, report):
    lines = ['# P56 versus corrected P32', '', f"Status: {report['status']}", '',
        'Positive bb/100 favours P56. One primary test: equal-weight mean against both fixed P32 refits.',
        'The independent observation is a shared deal group (four hands); pilot deals are excluded.',
        'Intervals condition on these three saved policies, not independent training runs.', '',
        '| Comparison | bb/100 | 95% interval | Conclusion |', '|---|---:|---|---|']
    results = {'P56 vs corrected P32 (primary)': report.get('result'), **report.get('by_fit_seed', {})}
    for name, result in results.items():
        if result:
            lo, hi = result['interval']
            lines.append(f"| {name} | {result['bb_per_100']:.3f} | [{lo:.3f}, {hi:.3f}] | {result['conclusion']} |")
    if report['smoke']:
        lines += ['', 'SMOKE TEST ONLY: these results are not evidence of playing strength.']
    plan = report.get('plan')
    if plan:
        lines += ['', f"Confirmation groups: {report['completed_groups']:,} / {plan['selected']:,}.",
            f"Target effect: 2 bb/100 at estimated 80% power. Sample cap binding: {plan['capped']}.",
            f"Pilot-estimated detectable effect: {plan['estimated_detectable_effect_bb100']:.3f} bb/100."]
    lines += ['', 'Per-seed comparisons are descriptive; no selection of a winning refit.',
        'This evaluates additional-training progress with corrected sampling at both endpoints.',
        'It does not separate policy-fit randomness from training progress or measure performance against Fell Omen 2.',
        'An inconclusive result does not establish equivalence. No training or replay changes are made.']
    (Path(output)/'comparison_report.md').write_text('\n'.join(lines)+'\n')


def run_evaluation(inputs, output, *, workers=8, max_seconds=7200, smoke=False):
    pilot_count, minimum, maximum = (4, 8, 8) if smoke else (10000, 100000, 1000000)
    validate_design(DEAL_SEED, workers, pilot_count, minimum, maximum, max_seconds, 2.)
    deal_seed = SMOKE_SEED if smoke else DEAL_SEED
    specs, sources = verify_inputs(inputs)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    matchups = [(f'h2h_{s}', 'P56', f'P32-corrected-{s}', 100*(i+1)) for i,s in enumerate(SEEDS)]
    write_json(root/'manifest.json', dict(sources=sources, policies=specs, game=serialisable_game_definition(),
        fit_seeds=list(SEEDS), seed_selection='Both archived corrected refits, equally weighted; no selection',
        matchups=matchups, pilot_seed=deal_seed, confirmation_seed=deal_seed+1,
        workers=workers, pilot_groups=pilot_count, minimum=minimum, maximum=maximum,
        max_seconds=max_seconds, alpha=.05, power=.8, effect_bb100=2., pilot_excluded=True,
        unit='shared deal group: average both seats, then both P32 fit seeds', smoke=smoke,
        collection=False, retraining=False, fo2_evaluation=False))
    report = dict(status='STARTING', result=None, by_fit_seed={}, smoke=smoke, completed_groups=0,
                  inference_scope='Conditional on one fixed P56 and two fixed P32 refits from one training run')
    write_json(root/'report.json', report)
    deadline, main = time.monotonic()+max_seconds, []
    try:
        with mp.get_context('spawn').Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            tick = time.monotonic()
            pilot = collect(pool, count=pilot_count, seed=deal_seed, matchups=matchups,
                path=root/'pilot.jsonl', deadline=deadline, workers=workers)
            if len(pilot) != pilot_count:
                report['status'] = 'INCOMPLETE_PILOT'
            else:
                plan = runtime_plan(observations(pilot), pilot_count/(time.monotonic()-tick),
                    deadline-time.monotonic(), minimum=minimum, maximum=maximum, effect_bb100=2.)
                report['plan'] = plan
                write_json(root/'plan.json', plan)
                if not plan['minimum_fits_budget']:
                    report['status'] = 'INSUFFICIENT_TIME_FOR_MINIMUM'
                else:
                    report['status'] = 'CONFIRMATION'
                    write_json(root/'report.json', report)
                    collect(pool, count=plan['selected'], seed=deal_seed+1, matchups=matchups,
                        path=root/'confirmatory.jsonl', deadline=deadline, workers=workers, rows=main)
                    report['status'] = 'COMPLETE' if len(main) == plan['selected'] else 'INCOMPLETE_TIME_LIMIT'
    except Exception as exc:
        report.update(status='EVALUATION_ERROR', reason=f'{type(exc).__name__}: {exc}')
    finally:
        if len(main) >= 2:
            report['result'] = estimate(observations(main), complete=report['status']=='COMPLETE' and not smoke)
            if smoke:
                report['result']['conclusion'] = 'smoke_only'
            for seed in SEEDS:
                result = estimate([r[f'h2h_{seed}']['mean_bb'] for r in main], complete=False)
                result['conclusion'] = 'descriptive'
                report['by_fit_seed'][str(seed)] = result
        report.update(completed_groups=len(main), completed_hands=4*len(main))
        write_json(root/'report.json', report)
        render(root, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=7200)
    parser.add_argument('--smoke', action='store_true')
    result = run_evaluation(**vars(parser.parse_args()))
    if result['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
