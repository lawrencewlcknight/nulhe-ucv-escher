"""One excluded pilot and three preregistered tests on fresh shared deal groups."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .checkpointing import sha256_file
from .evaluation_stats import estimate, sample_plan
from .fo2_setup import verify_build
from .learning_evaluate import initialise, collect, validate_design
from .p104_ensemble import verified_members, PILOT_SEED, CONFIRMATION_SEED
from .train import write_json


ALPHA = .05 / 3
EFFECT_BB100 = 2.
TESTS = ('head_to_head', 'fo2_improvement', 'ensemble_vs_fo2')
MATCHUPS = [('head', 'ensemble', 'P104', 100),
            ('ensemble_fo2', 'ensemble', 'opponent', 200),
            ('P104_fo2', 'P104', 'opponent', 200)]


def observations(rows):
    # Shared cards and action streams in the FO2 comparison; subtract WITHIN group.
    ensemble = np.asarray([r['ensemble_fo2']['mean_bb'] for r in rows])
    baseline = np.asarray([r['P104_fo2']['mean_bb'] for r in rows])
    return dict(head_to_head=np.asarray([r['head']['mean_bb'] for r in rows]),
        fo2_improvement=ensemble-baseline, ensemble_vs_fo2=ensemble)


def plan_from_pilot(pilot, rate, remaining, *, smoke=False):
    minimum, maximum = (8,8) if smoke else (100000,1000000)
    tests = {name:sample_plan(values, minimum=minimum, maximum=maximum,
        effect_bb100=EFFECT_BB100, alpha=ALPHA, power=.8)
        for name,values in observations(pilot).items()}
    # Sample size depends on pilot variance, never on significance or available time.
    selected = max(p['selected'] for p in tests.values())
    for p in tests.values():
        p['estimated_detectable_effect_bb100'] *= np.sqrt(p['selected']/selected)
        p.update(selected=selected, capped=p['required']>selected)
    seconds = selected/rate
    return dict(tests=tests, selected=selected, minimum=minimum, maximum=maximum,
        pilot_excluded=True, pilot_groups_per_second=rate,
        expected_confirmation_seconds=seconds, budgeted_confirmation_seconds=seconds/0.8,
        remaining_seconds=remaining, full_selected_sample_fits=seconds/0.8 <= remaining,
        sample_not_reduced_for_runtime=True, capped=any(p['capped'] for p in tests.values()))


def evaluate(manifest, policy, output, *, assets=None, workers=8, max_seconds=32400,
             smoke=False, opponent='fo2'):
    manifest, policy = Path(manifest).resolve(), Path(policy).resolve()
    _, fit = verified_members(manifest, policy=policy, smoke=smoke)
    if opponent not in ('fo2','check_call') or (not smoke and opponent != 'fo2'):
        raise ValueError('Production benchmark requires Fell Omen 2')
    pilot_n = 4 if smoke else 10000
    validate_design(PILOT_SEED, workers, pilot_n, 8 if smoke else 100000,
                    8 if smoke else 1000000, max_seconds, EFFECT_BB100)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    build = verify_build(assets) if opponent == 'fo2' else None
    specs = dict(ensemble=dict(kind='ensemble', manifest=str(manifest)),
        P104=dict(kind='checkpoint', checkpoint=str(policy)),
        opponent=dict(kind=opponent, assets=str(Path(assets or '.').resolve())))
    ensemble_hash = sha256_file(manifest)
    report = dict(status='STARTING', smoke=smoke, ensemble_sha256=ensemble_hash,
        source_policy_sha256=fit['source_policy_sha256'], completed_groups=0,
        alpha_per_test=ALPHA, familywise_alpha=.05,
        inference_scope='Conditional on this fixed 20-member ensemble and replay. Deal groups, not hands or members, are independent units.',
        attribution='Versus saved P104 tests the combined ensemble and all-replay export change, not ensembling alone.')
    write_json(output/'manifest.json', dict(**report, policies=specs, opponent_build=build,
        matchups=MATCHUPS, workers=workers, max_seconds=max_seconds, tests=list(TESTS),
        pilot_seed=PILOT_SEED, confirmation_seed=CONFIRMATION_SEED, pilot_groups=pilot_n,
        pilot_excluded=True, target_effect_bb100=EFFECT_BB100, power=.8,
        multiplicity='Bonferroni across head-to-head, FO2 improvement and absolute ensemble versus FO2.'))
    started = time.monotonic()
    deadline = started + max_seconds
    main = []
    try:
        with mp.get_context('spawn').Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            tick = time.monotonic()
            pilot = collect(pool, count=pilot_n, seed=PILOT_SEED, matchups=MATCHUPS,
                path=output/'pilot.jsonl', deadline=deadline, workers=workers)
            if len(pilot) != pilot_n:
                raise TimeoutError('Incomplete pilot')
            plan = plan_from_pilot(pilot, pilot_n/(time.monotonic()-tick),
                                   deadline-time.monotonic(), smoke=smoke)
            report['plan'] = plan
            write_json(output/'plan.json', plan)
            print(json.dumps(dict(stage='PILOT_COMPLETE', **plan)), flush=True)
            if not plan['full_selected_sample_fits']:
                report['status'] = 'INSUFFICIENT_TIME_FOR_SELECTED_SAMPLE'
            else:
                report['status'] = 'CONFIRMING'
                write_json(output/'report.json', report)
                collect(pool, count=plan['selected'], seed=CONFIRMATION_SEED, matchups=MATCHUPS,
                    path=output/'confirmatory.jsonl', deadline=deadline, workers=workers, rows=main)
                report['status'] = 'COMPLETE' if len(main) == plan['selected'] else 'INCOMPLETE_TIME_LIMIT'
        if build is not None and verify_build(assets) != build:
            raise ValueError('Fell Omen 2 build changed during evaluation')
        if sha256_file(manifest) != ensemble_hash or sha256_file(policy) != fit['source_policy_sha256']:
            raise ValueError('Policy identity changed during evaluation')
    except TimeoutError as exc:
        report.update(status='INCOMPLETE_TIME_LIMIT', reason=str(exc))
    except BaseException as exc:
        report.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if len(main) >= 2:
            complete = report['status'] == 'COMPLETE' and not smoke
            report['results'] = {name:estimate(values, alpha=ALPHA, complete=complete)
                                 for name,values in observations(main).items()}
            report['descriptive_95'] = {name:estimate(values, complete=False)
                                        for name,values in observations(main).items()}
            report['p104_vs_fo2_descriptive_95'] = estimate(
                [r['P104_fo2']['mean_bb'] for r in main], complete=False)
        report.update(completed_groups=len(main), completed_hands=len(main)*6,
                      elapsed_seconds=time.monotonic()-started)
        write_json(output/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','policy','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--assets', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-seconds', type=float, default=32400)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--opponent', default='fo2')
    if evaluate(**vars(p.parse_args()))['status'] != 'COMPLETE':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
