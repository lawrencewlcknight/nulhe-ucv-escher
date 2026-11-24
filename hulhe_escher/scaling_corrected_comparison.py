"""Two fixed-policy comparisons against both corrected smaller P32 exports."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .checkpointing import load_checkpoint_payload, sha256_file
from .evaluation_stats import estimate
from .game import serialisable_game_definition
from .learning_evaluate import collect, initialise, runtime_plan, validate_design
from .train import write_json

SOURCES = Path(__file__).with_name('data')/'scaling_corrected_sources.json'
SEEDS = (1701, 1702)
CANDIDATES = ('Wide-P32', 'Deep-P32')
DEAL_SEED = 624701
SMOKE_SEED = 624901
ALPHA = .025  # Bonferroni: two primary tests, familywise alpha .05.


def verify_inputs(inputs):
    root = Path(inputs)
    sources = json.loads(SOURCES.read_text())
    for name, record in sources['files'].items():
        if sha256_file(root/name) != record['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
    fit = json.loads((root/'P32-fit.json').read_text())
    if fit['status'] != 'COMPLETE' or fit['smoke'] or fit['source_iteration'] != 1132:
        raise ValueError('Expected completed production corrected P32 refits')
    labels = list(CANDIDATES)
    for seed in SEEDS:
        label = f'P32-corrected-{seed}'
        rows = [r for r in fit['results'] if r['arm']=='objective_mass' and r['fit_seed']==seed]
        if (len(rows)!=1 or rows[0]['fit']['group_sampling']!='objective_mass'
                or rows[0]['sha256']!=sources['files'][label+'.pkl']['sha256']):
            raise ValueError('Corrected refit provenance mismatch')
        labels.append(label)
    payloads = {label:load_checkpoint_payload(root/(label+'.pkl')) for label in labels}
    for label,payload in payloads.items():
        if payload['game'] != serialisable_game_definition():
            raise ValueError('Policies must use the specified limit game')
        # Architectures intentionally differ; information/action contracts must agree.
        for field in ('game','feature_encoder','input_size','num_actions'):
            if payload[field] != payloads[CANDIDATES[0]][field]:
                raise ValueError(f'Policies are not matched on {field}')
        if label.startswith('P32-corrected') and payload['outer_iteration'] != 1132:
            raise ValueError('Corrected P32 iteration mismatch')
    for label, iteration in zip(CANDIDATES,(953,868)):
        plan = json.loads((root/(label+'-launch.json')).read_text())
        training = json.loads((root/(label+'-scaling.json')).read_text())
        record = training['checkpoints']['P32']
        payload = payloads[label]
        if (training['status']!='TRAINING_COMPLETE' or training['smoke']
                or not training['fresh_initialization'] or not training['final_recovery_verified']
                or record['smoke'] or record.get('experimental_smoke',False)
                or record['active_seconds'] < 32*3600 or record['outer_iteration']!=iteration
                or record['sha256']!=sources['files'][label+'.pkl']['sha256']
                or payload['outer_iteration']!=iteration
                or payload['training_config']!=plan['config'] or training['config']!=plan['config']
                or payload['seed']!=plan['seed'] or training['seed']!=plan['seed']
                or record['metrics']['policy_validation']['group_sampling']!='objective_mass'):
            raise ValueError(f'{label} production identity/configuration mismatch')
    return {label:dict(kind='checkpoint',checkpoint=str((root/(label+'.pkl')).resolve()))
            for label in labels},sources


def matchups():
    return [(f'{candidate}/{seed}',candidate,f'P32-corrected-{seed}',100*(2*i+j+1))
            for i,candidate in enumerate(CANDIDATES) for j,seed in enumerate(SEEDS)]


def observations(rows,candidate):
    # Four hands per candidate, eight per shared group. Never treat hands/refits
    # sharing a deal as independent observations.
    return np.asarray([np.mean([row[f'{candidate}/{seed}']['mean_bb'] for seed in SEEDS]) for row in rows])


def common_plan(pilot,rate,remaining,*,minimum,maximum):
    plans = {name:runtime_plan(observations(pilot,name),rate,remaining,
        minimum=minimum,maximum=maximum,effect_bb100=2.,alpha=ALPHA) for name in CANDIDATES}
    selected = max(p['selected'] for p in plans.values())
    for plan in plans.values():
        original = plan['selected']
        plan['selected'] = selected
        plan['estimated_detectable_effect_bb100'] *= np.sqrt(original/selected)
        plan['capped'] = plan['required'] > selected
    return dict(selected=selected,by_candidate=plans,
        minimum_fits_budget=all(p['minimum_fits_budget'] for p in plans.values()),
        pilot_excluded=True,alpha_per_test=ALPHA,familywise_alpha=.05,
        rule='Larger of the two pilot-based sample requirements, subject to declared/runtime caps; no outcome stopping.')


def render(root,report):
    lines = ['# Wide-P32 and Deep-P32 versus corrected smaller P32','',f"Status: {report['status']}",'',
        'Positive bb/100 favours the named candidate. Both corrected P32 fit seeds are equally weighted.',
        'Two primary tests use 97.5% intervals (Bonferroni familywise alpha 5%).',
        'The observation is a shared deal group; pilot deals are excluded.', '',
        '| Candidate | bb/100 | 97.5% interval | Conclusion |','|---|---:|---|---|']
    for name,r in report.get('results',{}).items():
        lines.append(f"| {name} | {r['bb_per_100']:+.3f} | [{r['interval'][0]:+.3f}, {r['interval'][1]:+.3f}] | {r['conclusion']} |")
    lines += ['',f"Confirmation groups: {report['completed_groups']:,}; hands: {report['completed_hands']:,}.",
        'Each group contains eight hands: both seats against both refits for both candidates.',
        'Per-refit results in report.json are descriptive, not additional primary tests.',
        'The smaller baseline was refitted with corrected sampling after its historical training.',
        'The larger models used corrected sampling throughout fresh training and also differ in replay/configuration.',
        'These comparisons test saved training recipes, not architecture alone or generalization across training seeds.',
        'No training, replay download, policy selection or Fell Omen 2 evaluation.']
    if report.get('plan'):
        lines += ['', 'Pilot power planning targets 2 bb/100 at estimated 80% power.',
            'Capped tests: '+', '.join(k for k,v in report['plan']['by_candidate'].items() if v['capped'])+'.']
    if report['smoke']:
        lines += ['', 'SMOKE ONLY: not evidence of playing strength.']
    (root/'comparison_report.md').write_text('\n'.join(lines)+'\n')


def run_evaluation(inputs,output,*,workers=8,max_seconds=10800,smoke=False):
    pilot_count,minimum,maximum = (4,8,8) if smoke else (10000,100000,1000000)
    seed = SMOKE_SEED if smoke else DEAL_SEED
    validate_design(seed,workers,pilot_count,minimum,maximum,max_seconds,2.)
    specs,sources = verify_inputs(inputs)
    root = Path(output)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'manifest.json',dict(sources=sources,policies=specs,matchups=matchups(),
        fit_seeds=list(SEEDS),pilot_seed=seed,confirmation_seed=seed+1,workers=workers,
        pilot_groups=pilot_count,minimum_groups=minimum,maximum_groups=maximum,max_seconds=max_seconds,
        alpha_per_test=ALPHA,familywise_alpha=.05,power=.8,target_effect_bb100=2.,
        unit='Eight-hand shared deal group; seats then refits averaged within each candidate.',
        collection=False,retraining=False,smoke=smoke))
    report = dict(status='STARTING',smoke=smoke,results={},by_fit_seed={},completed_groups=0,completed_hands=0)
    write_json(root/'report.json',report)
    started = time.monotonic()
    deadline,main = started+max_seconds,[]
    try:
        with mp.get_context('spawn').Pool(workers,initializer=initialise,initargs=(specs,)) as pool:
            tick = time.monotonic()
            pilot = collect(pool,count=pilot_count,seed=seed,matchups=matchups(),path=root/'pilot.jsonl',
                deadline=deadline,workers=workers)
            if len(pilot)!=pilot_count:
                report['status'] = 'INCOMPLETE_PILOT'
            else:
                plan = common_plan(pilot,pilot_count/(time.monotonic()-tick),deadline-time.monotonic(),
                    minimum=minimum,maximum=maximum)
                report['plan'] = plan
                write_json(root/'plan.json',plan)
                if not plan['minimum_fits_budget']:
                    report['status'] = 'INSUFFICIENT_TIME_FOR_MINIMUM'
                else:
                    report['status'] = 'CONFIRMATION'
                    write_json(root/'report.json',report)
                    collect(pool,count=plan['selected'],seed=seed+1,matchups=matchups(),path=root/'confirmatory.jsonl',
                        deadline=deadline,workers=workers,rows=main)
                    report['status'] = 'COMPLETE' if len(main)==plan['selected'] else 'INCOMPLETE_TIME_LIMIT'
    except Exception as exc:
        report.update(status='EVALUATION_ERROR',reason=f'{type(exc).__name__}: {exc}')
    finally:
        if len(main)>=2:
            for name in CANDIDATES:
                result = estimate(observations(main,name),alpha=ALPHA,complete=report['status']=='COMPLETE' and not smoke)
                if smoke:
                    result['conclusion']='smoke_only'
                report['results'][name] = result
                report['by_fit_seed'][name] = {}
                for fit_seed in SEEDS:
                    r = estimate([row[f'{name}/{fit_seed}']['mean_bb'] for row in main],complete=False)
                    r['conclusion']='descriptive'
                    report['by_fit_seed'][name][str(fit_seed)] = r
        report.update(completed_groups=len(main),completed_hands=8*len(main),elapsed_seconds=time.monotonic()-started)
        write_json(root/'report.json',report)
        render(root,report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-seconds',type=float,default=10800)
    p.add_argument('--smoke',action='store_true')
    if run_evaluation(**vars(p.parse_args()))['status']!='COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
