"""Predeclared final-versus-baseline primary and eight Bonferroni-adjusted secondary tests."""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm

from .benchmark import wait_policy
from .continuation import LABELS, checkpoint_labels
from .evaluation_stats import estimate, sample_plan
from .fo2_setup import DEFAULT_ASSETS, verify_build
from .learning_evaluate import initialise, collect, run_evaluation, validate_design
from .train import write_json

SECONDARY_ALPHA = .05 / 8  # Four adjacent head-to-head + four FO2 changes vs baseline.


def fo2_plan(pilot, *, minimum, maximum, affordable, effect_bb100=2., alpha=SECONDARY_ALPHA, labels=LABELS):
    base = np.asarray([r[labels[0]]['mean_bb'] for r in pilot])
    differences = {label: sample_plan(np.asarray([r[label]['mean_bb'] for r in pilot])-base,
        minimum=minimum, maximum=maximum, effect_bb100=effect_bb100, alpha=alpha)
        for label in labels[1:]}
    # Absolute win rates are descriptive; also target +/-2 bb/100 precision.
    precision = {label: max(2, math.ceil(norm.ppf(.975)**2 *
        np.var([r[label]['mean_bb'] for r in pilot], ddof=1)/(.02**2))) for label in labels}
    required = max(*[p['required'] for p in differences.values()], *precision.values())
    selected = max(minimum, min(maximum, affordable, max(minimum, required)))
    factor = norm.ppf(1-alpha/2)+norm.ppf(.8)
    return dict(required=required, selected=selected, capped=selected < required,
        minimum_fits_budget=affordable >= minimum, affordable_groups=affordable,
        differences=differences, precision_required=precision, alpha=alpha,
        estimated_detectable_effect_bb100={k: float(100*factor*math.sqrt(v['pilot_variance_bb2']/selected))
                                          for k,v in differences.items()},
        pilot_excluded=True, unit='shared_deal_ten_hand_group')


def run_fo2(experiment, output, *, workers=8, pilot_pairs=10000, minimum=100000,
            maximum=1000000, max_seconds=34560, seed=29501, assets=DEFAULT_ASSETS,
            opponent='fo2', primary_passed=False, labels=LABELS):
    validate_design(seed, workers, pilot_pairs, minimum, maximum, max_seconds, 2.)
    if opponent not in ('fo2', 'check_call', 'random'):
        raise ValueError('Invalid opponent')
    root, output = Path(experiment), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic()+max_seconds
    report = dict(status='STARTING', estimates={}, improvements={}, alpha=SECONDARY_ALPHA,
                  primary_passed=primary_passed, checkpoints={}, smoke=False,
                  opponent_variant='Fell Omen 2, free-fold-to-check adaptation' if opponent == 'fo2' else 'OFFLINE TEST: '+opponent)
    main = []
    try:
        specs = {'fo2': dict(kind=opponent, assets=str(Path(assets).resolve()))}
        for label in labels:
            policy, record = wait_policy(root, label, deadline)
            specs[label] = dict(kind='checkpoint', checkpoint=str(policy))
            report['checkpoints'][label] = {k:record[k] for k in ('sha256', 'outer_iteration', 'active_seconds')}
            report['smoke'] |= record.get('smoke', False)
        build = verify_build(assets) if opponent == 'fo2' else None
        write_json(output / 'manifest.json', dict(policies=report['checkpoints'], build=build,
            source='continuation.json', pilot_seed=seed, confirmation_seed=seed+1,
            workers=workers, pilot_pairs=pilot_pairs, minimum=minimum, maximum=maximum,
            max_seconds=max_seconds, alpha=SECONDARY_ALPHA, opponent_variant=report['opponent_variant'],
            action_streams={label:100*(i+1) for i,label in enumerate(labels)},
            secondary_family=f'four adjacent head-to-head plus four FO2 changes from {labels[0]}',
            unit='shared_deal_ten_hand_group'))
        matchups = [(label,label,'fo2',100*(i+1)) for i,label in enumerate(labels)]
        report['status'] = 'PILOT'
        write_json(output / 'report.json', report)
        with mp.get_context('spawn').Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            started = time.monotonic()
            pilot = collect(pool, count=pilot_pairs, seed=seed, matchups=matchups,
                            path=output / 'pilot.jsonl', deadline=deadline, workers=workers)
            if len(pilot) != pilot_pairs:
                raise TimeoutError('Incomplete FO2 pilot')
            rate = pilot_pairs/(time.monotonic()-started)
            plan = fo2_plan(pilot, minimum=minimum, maximum=maximum,
                           affordable=int(.8*rate*max(0, deadline-time.monotonic())), labels=labels)
            plan['pilot_groups_per_second'] = rate
            report['plan'] = plan
            write_json(output / 'plan.json', plan)
            if not plan['minimum_fits_budget']:
                report['status'] = 'INSUFFICIENT_TIME_FOR_MINIMUM'
            else:
                report['status'] = 'MAIN'
                write_json(output / 'report.json', report)
                collect(pool, count=plan['selected'], seed=seed+1, matchups=matchups,
                        path=output / 'confirmatory.jsonl', deadline=deadline, workers=workers, rows=main)
                report['status'] = 'COMPLETE' if len(main) == plan['selected'] else 'INCOMPLETE_TIME_LIMIT'
        if opponent == 'fo2' and verify_build(assets) != build:
            report['status'] = 'INVALID_OPPONENT_VERSION_CHANGE'
    except TimeoutError as exc:
        report.update(status='INCOMPLETE_TIME_LIMIT', reason=str(exc))
    except Exception as exc:
        report.update(status='EVALUATION_ERROR', error_type=type(exc).__name__, reason=str(exc))
    finally:
        usable = (report['status']=='COMPLETE' and opponent=='fo2' and
                  not report['smoke'] and primary_passed)
        if len(main) >= 2:
            scores = {label:np.asarray([r[label]['mean_bb'] for r in main]) for label in labels}
            for label, values in scores.items():
                report['estimates'][label] = estimate(values, complete=False)
                report['estimates'][label]['conclusion'] = 'descriptive'
            report['improvements'] = {label:estimate(scores[label]-scores[labels[0]],
                alpha=SECONDARY_ALPHA, complete=usable) for label in labels[1:]}
        report['completed_groups'] = len(main)
        report['completed_hands_per_policy'] = 2*len(main)
        report['adaptations'] = {label:sum(r[label]['adaptations'] for r in main) for label in labels}
        write_json(output / 'report.json', report)
    return report


def gate_secondary(result, primary_passed, smoke):
    if result and (not primary_passed or smoke):
        result.update(confirmatory=False, conclusion='descriptive_primary_gate_not_passed' if not smoke else 'smoke_only')
    return result


def render(root, report):
    labels = report.get('labels', LABELS)
    lines = ['# Limit Holdem 24-hour continuation', '',
             f'Primary: {labels[-1]} vs {labels[0]}. Secondary: four adjacent matches and four FO2 changes vs {labels[0]}.',
             'Secondary intervals are 99.375% (Bonferroni across eight); claims require the primary test to pass.',
             'One training seed. Absolute FO2 scores are descriptive; capped/incomplete evaluations are labelled.', '',
             '| Comparison | Status | bb/100 | Interval | Conclusion |',
             '|---|---|---:|---|---|']
    def row(name, status, result):
        if result:
            lo,hi=result['interval']
            lines.append(f"| {name} | {status} | {result['bb_per_100']:.2f} | {100*result['confidence']:.3f}% [{lo:.2f}, {hi:.2f}] | {result['conclusion']} |")
        else:
            lines.append(f'| {name} | {status} | — | — | — |')
    for name, item in report.get('head_to_head', {}).items():
        row(name,item['status'],item.get('result'))
    fo2=report.get('fo2', {})
    for label in labels:
        row(label+' vs FO2',fo2.get('status','PENDING'),fo2.get('estimates',{}).get(label))
    for label in labels[1:]:
        row(label+f' minus {labels[0]} vs FO2',fo2.get('status','PENDING'),fo2.get('improvements',{}).get(label))
    (Path(root)/'continuation_report.md').write_text('\n'.join(lines)+'\n')


def run_suite(experiment, output, *, workers=8, max_seconds=43200,
              assets=DEFAULT_ASSETS, smoke=False, opponent='fo2'):
    root, output = Path(experiment), Path(output)
    training = json.loads((root/'continuation.json').read_text())
    if training['status'] != 'TRAINING_COMPLETE' or not training.get('final_recovery_verified'):
        raise ValueError('Evaluate only after all stages and the final reload check complete')
    if bool(training['smoke']) != bool(smoke):
        raise ValueError('Explicit smoke mode must match training')
    if not smoke and opponent != 'fo2':
        raise ValueError('Production evaluation requires Fell Omen 2')
    baseline_hours = training.get('baseline_hours', 8)
    labels = checkpoint_labels(baseline_hours)
    if set(training['checkpoints']) != set(labels):
        raise ValueError('Completed checkpoints differ from the declared continuation grid')
    seed_offset = 1000*(baseline_hours-8)
    validate_design(28301+seed_offset, workers, 4 if smoke else 10000, 8 if smoke else 100000,
                    8 if smoke else 1000000, max_seconds, 2.)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='EVALUATING', smoke=smoke, primary=f'{labels[-1]}-vs-{labels[0]}', labels=list(labels), head_to_head={},
                  secondary_alpha=SECONDARY_ALPHA, secondary_family_size=8)
    write_json(output/'design.json', dict(labels=labels, primary=report['primary'],
        secondary_alpha=SECONDARY_ALPHA, secondary_family_size=8, primary_alpha=.05,
        effect_bb100=2, power=.8, pilot_excluded=True, workers=workers,
        max_seconds=max_seconds, smoke=smoke,
        head_to_head_pilot_seeds=[28301+seed_offset+10*i for i in range(5)],
        fo2_pilot_seed=29501+seed_offset))
    deadline=time.monotonic()+max_seconds
    # Validate ALL exported identities before the first evaluation starts.
    for label in labels:
        _, record=wait_policy(root,label,deadline)
        if record['sha256'] != training['checkpoints'][label]['sha256']:
            raise ValueError('Policy differs from completed training record')
    limits=dict(pilot_pairs=4, minimum=8, maximum=8) if smoke else {}
    matches=[(labels[-1],labels[0]), *zip(labels[1:],labels[:-1])]
    h2h_deadline=time.monotonic()+.2*max_seconds
    primary_passed=False
    for i,(a,b) in enumerate(matches):
        name=a+'-vs-'+b
        budget=(h2h_deadline-time.monotonic())/(len(matches)-i)
        if budget <= 1:
            item=dict(status='SKIPPED_TIME_LIMIT',result=None)
        else:
            item=run_evaluation(root/f'{a}.pkl',str(root/f'{b}.pkl'),output/name,
                workers=workers, max_seconds=budget, seed=28301+seed_offset+10*i,
                alpha=.05 if i==0 else SECONDARY_ALPHA, **limits)
        if i==0:
            primary_passed=(not smoke and item['status']=='COMPLETE' and
                (item.get('result') or {}).get('confirmatory',False) and item['result']['conclusion']=='improved')
            if smoke and item.get('result'):
                gate_secondary(item['result'],False,True)
        else:
            gate_secondary(item.get('result'),primary_passed,smoke)
        report['head_to_head'][name]=item
        write_json(output/'report.json',report)
        if (output/name/'report.json').exists():
            write_json(output/name/'report.json',item)
    budget=deadline-time.monotonic()
    report['fo2']=run_fo2(root,output/'fo2',workers=workers,max_seconds=budget,
        assets=assets,opponent=opponent,primary_passed=primary_passed,
        labels=labels,seed=29501+seed_offset,**limits) if budget>1 else dict(status='SKIPPED_TIME_LIMIT')
    report['status']='COMPLETE' if (all(x['status']=='COMPLETE' for x in report['head_to_head'].values())
                                    and report['fo2']['status']=='COMPLETE') else 'EVALUATION_INCOMPLETE'
    report['primary_passed']=primary_passed
    write_json(output/'report.json',report)
    render(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-seconds',type=float,default=43200)
    p.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--opponent',choices=['fo2','check_call','random'],default='fo2')
    result=run_suite(**vars(p.parse_args()))
    if result['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__=='__main__':
    main()
