"""Pinned Wide/Deep/corrected Small round robins on fresh duplicate deals."""
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
from .train import write_json

SOURCES = Path(__file__).with_name('data')/'p56_round_robin_sources.json'
LABELS = ('Wide-P56', 'Deep-P56', 'Small-P56')
MATCHUPS = [('Wide-vs-Deep', LABELS[0], LABELS[1], 100),
            ('Wide-vs-Small', LABELS[0], LABELS[2], 200),
            ('Deep-vs-Small', LABELS[1], LABELS[2], 300)]
DEAL_SEED = 926561
SMOKE_SEED = 926961
ALPHA = .05/3


def source_file(checkpoint_hours=56):
    if checkpoint_hours not in (56, 104):
        raise ValueError('Round robin requires a pinned P56 or P104 checkpoint')
    return SOURCES if checkpoint_hours == 56 else SOURCES.with_name('p104_round_robin_sources.json')


def matchups_for(checkpoint_hours):
    source_file(checkpoint_hours)  # Reject unsupported checkpoints before any work.
    labels = tuple(f'{variant}-P{checkpoint_hours}' for variant in ('Wide', 'Deep', 'Small'))
    return [('Wide-vs-Deep', labels[0], labels[1], 100),
            ('Wide-vs-Small', labels[0], labels[2], 200),
            ('Deep-vs-Small', labels[1], labels[2], 300)]


def verify_inputs(inputs, *, checkpoint_hours=56):
    """Hash all six pinned inputs before loading any trusted pickle."""
    root = Path(inputs)
    sources = json.loads(source_file(checkpoint_hours).read_text())
    labels = tuple(f'{variant}-P{checkpoint_hours}' for variant in ('Wide', 'Deep', 'Small'))
    for name, record in sources['files'].items():
        if sha256_file(root/name) != record['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
    payloads = {}
    for label in labels:
        marker = json.loads((root/(label+'.json')).read_text())
        pin = sources['policies'][label]
        if (marker['label'] != f'P{checkpoint_hours}' or marker['smoke']
                or marker.get('experimental_smoke', False)
                or marker['outer_iteration'] != pin['iteration']
                or marker['active_seconds'] != pin['active_seconds']
                or marker['active_seconds'] < checkpoint_hours*3600
                or marker['sha256'] != sources['files'][label+'.pkl']['sha256']
                or marker['metrics']['policy_validation']['group_sampling'] != 'objective_mass'):
            raise ValueError(f'{label} corrected production P{checkpoint_hours} provenance mismatch')
        payload = load_checkpoint_payload(root/(label+'.pkl'))
        if (payload['outer_iteration'] != pin['iteration']
                or payload['training_elapsed_seconds'] != pin['active_seconds']
                or payload['game'] != serialisable_game_definition()):
            raise ValueError(f'{label} checkpoint identity/game mismatch')
        payloads[label] = payload
    for label, payload in payloads.items():
        for field in ('game', 'feature_encoder', 'input_size', 'num_actions'):
            if payload[field] != payloads[labels[0]][field]:
                raise ValueError(f'Policies are not matched on {field}')
    return {label: dict(kind='checkpoint', checkpoint=str((root/(label+'.pkl')).resolve()))
            for label in labels}, sources


def observations(rows, matchup):
    # Each observation averages the two seat-swapped hands for ONE deal.
    return np.asarray([row[matchup]['mean_bb'] for row in rows])


def common_plan(pilot, rate, remaining, *, minimum, maximum):
    plans = {name: sample_plan(observations(pilot, name), minimum=minimum, maximum=maximum,
             effect_bb100=2., alpha=ALPHA, power=.8) for name, *_ in MATCHUPS}
    selected = max(p['selected'] for p in plans.values())
    for plan in plans.values():
        plan['estimated_detectable_effect_bb100'] *= np.sqrt(plan['selected']/selected)
        plan.update(selected=selected, capped=plan['required'] > selected)
    expected = selected/rate
    return dict(selected=selected, by_matchup=plans, pilot_groups_per_second=rate,
        expected_confirmation_seconds=expected, full_selected_sample_fits=expected/.8 <= remaining,
        pilot_excluded=True, sample_not_reduced_for_runtime=True,
        alpha_per_test=ALPHA, familywise_alpha=.05,
        rule='Maximum of three pilot requirements, capped at one million groups; no outcome stopping.')


def render(root, report):
    hours = report.get('checkpoint_hours', 56)
    lines = [f'# P{hours} Wide / Deep / corrected Small round robin', '', f"Status: {report['status']}", '',
        'Positive bb/100 favours the first policy in each matchup.',
        'Three predeclared tests use 98.333% intervals (Bonferroni familywise alpha 5%).',
        'One observation per matchup is the mean of two seat-swapped hands on a shared deal.',
        'Pilot deals are excluded. Intervals condition on these saved policies, not training seeds.', '',
        '| Matchup | bb/100 | 98.333% interval | Conclusion |', '|---|---:|---|---|']
    for name, r in report['results'].items():
        lines.append(f"| {name} | {r['bb_per_100']:+.3f} | [{r['interval'][0]:+.3f}, {r['interval'][1]:+.3f}] | {r['conclusion']} |")
    lines += ['', f"Confirmation groups: {report['completed_groups']:,}; total hands: {report['completed_hands']:,}.",
        'Each group contains six hands: all three pairings with seats swapped.',
        'All three exports use objective-mass sampling. The smaller policy has a different historical',
        'training recipe; these are fixed-policy comparisons, not an architecture-only ablation.',
        'Pairwise outcomes need not imply a transitive ranking or lower exploitability.',
        'No training, replay downloads, policy selection or additional Fell Omen 2 evaluation.']
    if report.get('plan'):
        lines += ['', 'Planning target: 80% estimated power for 2 bb/100 per matchup.',
            'Power-capped matchups: '+(', '.join(k for k,v in report['plan']['by_matchup'].items()
                                                if v['capped']) or 'none')+'.']
    if report['smoke']:
        lines += ['', 'SMOKE ONLY: not evidence of playing strength.']
    (root/'comparison_report.md').write_text('\n'.join(lines)+'\n')


def run_evaluation(inputs, output, *, workers=8, max_seconds=10800, smoke=False, checkpoint_hours=56):
    matchups = matchups_for(checkpoint_hours)
    pilot_count, minimum, maximum = (4, 8, 8) if smoke else (10000, 100000, 1000000)
    seed = (SMOKE_SEED if smoke else DEAL_SEED) if checkpoint_hours == 56 else (928941 if smoke else 928041)
    validate_design(seed, workers, pilot_count, minimum, maximum, max_seconds, 2.)
    specs, sources = verify_inputs(inputs, checkpoint_hours=checkpoint_hours)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', dict(sources=sources, policies=specs, matchups=matchups,
        checkpoint_hours=checkpoint_hours,
        pilot_seed=seed, confirmation_seed=seed+1, workers=workers,
        pilot_groups=pilot_count, minimum_groups=minimum, maximum_groups=maximum,
        max_seconds=max_seconds, alpha_per_test=ALPHA, familywise_alpha=.05, power=.8,
        target_effect_bb100=2., unit='Six-hand shared deal group; seat means per matchup.',
        collection=False, retraining=False, smoke=smoke))
    report = dict(status='STARTING', smoke=smoke, checkpoint_hours=checkpoint_hours,
                  results={}, completed_groups=0, completed_hands=0)
    write_json(root/'report.json', report)
    started = time.monotonic()
    deadline, main = started+max_seconds, []
    try:
        with mp.get_context('spawn').Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            tick = time.monotonic()
            pilot = collect(pool, count=pilot_count, seed=seed, matchups=matchups,
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
                    collect(pool, count=plan['selected'], seed=seed+1, matchups=matchups,
                        path=root/'confirmatory.jsonl', deadline=deadline, workers=workers, rows=main)
                    report['status'] = 'COMPLETE' if len(main) == plan['selected'] else 'INCOMPLETE_TIME_LIMIT'
        verify_inputs(inputs, checkpoint_hours=checkpoint_hours)  # Guard against source replacement.
    except Exception as exc:
        report.update(status='EVALUATION_ERROR', reason=f'{type(exc).__name__}: {exc}')
    finally:
        if len(main) >= 2:
            for name, *_ in MATCHUPS:
                r = estimate(observations(main, name), alpha=ALPHA,
                             complete=report['status']=='COMPLETE' and not smoke)
                if smoke:
                    r['conclusion'] = 'smoke_only'
                report['results'][name] = r
        report.update(completed_groups=len(main), completed_hands=6*len(main),
                      elapsed_seconds=time.monotonic()-started)
        write_json(root/'report.json', report)
        render(root, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-seconds', type=float, default=10800)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--checkpoint-hours', type=int, choices=(56, 104), default=56)
    if run_evaluation(**vars(p.parse_args()))['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
