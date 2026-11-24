"""Frozen Hybrid-P32 benchmarks: deep primary, corrected P32 and FO2 secondary."""
import argparse
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .checkpointing import load_checkpoint_payload, sha256_file
from .evaluation_stats import estimate, sample_plan
from .fo2_setup import verify_build
from .learning_evaluate import initialise, collect, validate_design
from .scaling_corrected_comparison import verify_inputs, SEEDS
from .scaling_config import hybrid_scaling_config, HYBRID_COMPARISON_CAVEAT
from .train import write_json

PILOT_SEED = 726401
ALPHAS = {'hybrid_vs_deep': .05, 'hybrid_vs_corrected': .025, 'hybrid_vs_fo2': .025}
MATCHUPS = [('deep', 'hybrid', 'Deep-P32', 100),
            *[(f'corrected_{s}', 'hybrid', f'P32-corrected-{s}', 200+i*100)
              for i, s in enumerate(SEEDS)],
            ('fo2', 'hybrid', 'opponent', 400)]


def observations(rows):
    return dict(hybrid_vs_deep=np.asarray([r['deep']['mean_bb'] for r in rows]),
        hybrid_vs_corrected=np.asarray([np.mean([r[f'corrected_{s}']['mean_bb'] for s in SEEDS])
                                        for r in rows]),
        hybrid_vs_fo2=np.asarray([r['fo2']['mean_bb'] for r in rows]))


def plan_from_pilot(pilot, rate, remaining, *, smoke=False):
    minimum, maximum = (8, 8) if smoke else (100000, 1000000)
    tests = {name: sample_plan(values, minimum=minimum, maximum=maximum,
             effect_bb100=2., alpha=ALPHAS[name], power=.8)
             for name, values in observations(pilot).items()}
    selected = max(p['selected'] for p in tests.values())
    for p in tests.values():
        p['estimated_detectable_effect_bb100'] *= np.sqrt(p['selected']/selected)
        p.update(selected=selected, capped=p['required'] > selected)
    expected = selected/rate
    return dict(tests=tests, selected=selected, minimum=minimum, maximum=maximum,
        pilot_excluded=True, expected_confirmation_seconds=expected,
        full_selected_sample_fits=expected/.8 <= remaining,
        sample_not_reduced_for_runtime=True, capped=any(p['capped'] for p in tests.values()))


def evaluate(policy, inputs, output, *, assets=None, workers=8, max_seconds=32400,
             smoke=False, opponent='fo2'):
    policy, inputs, output = Path(policy).resolve(), Path(inputs).resolve(), Path(output).resolve()
    specs, sources = verify_inputs(inputs)
    if opponent not in ('fo2', 'check_call') or (not smoke and opponent != 'fo2'):
        raise ValueError('Production requires Fell Omen 2')
    candidate = load_checkpoint_payload(policy)
    reference = load_checkpoint_payload(inputs/'Deep-P32.pkl')
    for field in ('game', 'feature_encoder', 'input_size', 'num_actions'):
        if candidate[field] != reference[field]:
            raise ValueError('Hybrid and controls differ on '+field)
    if candidate['training_config'] != hybrid_scaling_config(small=smoke):
        raise ValueError('Candidate must use the versioned hybrid configuration')
    if not smoke and (candidate['seed'] != 20261115
                      or candidate['training_elapsed_seconds'] < 32*3600):
        raise ValueError('Production benchmark requires Hybrid-P32 at the declared training seed')
    pilot_n = 4 if smoke else 10000
    validate_design(PILOT_SEED, workers, pilot_n, 8 if smoke else 100000,
                    8 if smoke else 1000000, max_seconds, 2.)
    build = verify_build(assets) if opponent == 'fo2' else None
    specs.pop('Wide-P32')
    specs.update(hybrid=dict(kind='checkpoint', checkpoint=str(policy)),
                 opponent=dict(kind=opponent, assets=str(Path(assets or '.').resolve())))
    output.mkdir(parents=True, exist_ok=False)
    candidate_hash = sha256_file(policy)
    report = dict(status='STARTING', smoke=smoke, completed_groups=0, results={},
        primary='hybrid_vs_deep', alpha_by_test=ALPHAS, candidate_sha256=candidate_hash,
        comparison_caveat=HYBRID_COMPARISON_CAVEAT)
    write_json(output/'manifest.json', dict(**report, sources=sources, policies=specs,
        matchups=MATCHUPS, opponent_build=build, workers=workers, max_seconds=max_seconds,
        pilot_seed=PILOT_SEED, confirmation_seed=PILOT_SEED+1, pilot_groups=pilot_n,
        power=.8, effect_bb100=2., unit='Eight-hand deal group; corrected refits equally weighted.',
        multiplicity='Primary alpha .05; separate two-test secondary family alpha .05.'))
    started = time.monotonic()
    deadline, main = started+max_seconds, []
    write_json(output/'report.json', report)
    try:
        with mp.get_context('spawn').Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            tick = time.monotonic()
            pilot = collect(pool, count=pilot_n, seed=PILOT_SEED, matchups=MATCHUPS,
                path=output/'pilot.jsonl', deadline=deadline, workers=workers)
            if len(pilot) != pilot_n:
                report['status'] = 'INCOMPLETE_PILOT'
            else:
                plan = plan_from_pilot(pilot, pilot_n/(time.monotonic()-tick),
                                       deadline-time.monotonic(), smoke=smoke)
                report['plan'] = plan
                write_json(output/'plan.json', plan)
                if not plan['full_selected_sample_fits']:
                    report['status'] = 'INSUFFICIENT_TIME_FOR_SELECTED_SAMPLE'
                else:
                    report['status'] = 'CONFIRMING'
                    write_json(output/'report.json', report)
                    collect(pool, count=plan['selected'], seed=PILOT_SEED+1, matchups=MATCHUPS,
                        path=output/'confirmatory.jsonl', deadline=deadline, workers=workers, rows=main)
                    report['status'] = 'COMPLETE' if len(main) == plan['selected'] else 'INCOMPLETE_TIME_LIMIT'
        if build is not None and verify_build(assets) != build:
            raise ValueError('FO2 build changed during evaluation')
        if sha256_file(policy) != candidate_hash:
            raise ValueError('Candidate changed during evaluation')
        verify_inputs(inputs)
    except Exception as exc:
        report.update(status='EVALUATION_ERROR', reason=f'{type(exc).__name__}: {exc}')
    finally:
        if len(main) >= 2:
            report['results'] = {name: estimate(values, alpha=ALPHAS[name],
                complete=report['status'] == 'COMPLETE' and not smoke)
                for name, values in observations(main).items()}
            report['corrected_by_fit_seed_descriptive'] = {str(s): estimate(
                [r[f'corrected_{s}']['mean_bb'] for r in main], complete=False) for s in SEEDS}
        report.update(completed_groups=len(main), completed_hands=8*len(main),
                      elapsed_seconds=time.monotonic()-started)
        write_json(output/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('policy', 'inputs', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--assets', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-seconds', type=float, default=32400)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--opponent', default='fo2')
    if evaluate(**vars(p.parse_args()))['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
