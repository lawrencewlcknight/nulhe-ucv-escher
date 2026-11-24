"""Outcome-blind encounter selection and independent reference/target sampling."""
from collections import Counter, defaultdict
import time

import numpy as np

from .position_investigation.common import visible_position, families, protected_orbits, orbit
from .strategic_audit.library import digest, canonical_board
from .supervision_runtime import conditional_ranges, exact_river_q


SEED = 10411001
QUOTAS = {3:16,4:32,5:48}
PARTITIONS = ('development','validation','assessment')


def rng_for(*parts):
    return np.random.default_rng(np.random.SeedSequence([SEED,*map(int,parts)]))


def encounters(runtime, start, count, check=lambda:None):
    rows = []
    for trajectory in range(start,start+count):
        rng = rng_for(1,trajectory)
        hero,fold = (trajectory//2)%2,trajectory%2
        state = runtime.game.new_initial_state()
        decision = 0
        while not state.is_terminal():
            check()
            runtime.chance(state,rng)
            if state.is_terminal():
                break
            if state.current_player() == hero:
                position = visible_position(state)
                if len(position['board']) >= 3:
                    feature = runtime.encoder.information_state(state,hero)
                    rows.append(dict(**position,id=f't{trajectory}-d{decision}',trajectory=trajectory,
                        fold=fold,priority=digest([SEED,trajectory,decision]),
                        family=families(feature,runtime.encoder)[0]))
                _,_,sample,_,_ = runtime.controls(state,hero,fold)
            else:
                sample = runtime.policy(state)
            state.apply_action(int(rng.choice(3,p=sample)))
            decision += 1
    return rows


def select_positions(rows, *, smoke=False):
    """Connected flop/full-board suit orbits stay together across all three splits."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    keys = []
    for i,row in enumerate(rows):
        row_keys = [('flop',canonical_board(row['board'][:3])),('full',canonical_board(row['board']))]
        keys.append(row_keys)
        for key in row_keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    strata = defaultdict(list)
    protected = protected_orbits()
    for ids in components.values():
        identity = digest(sorted(set(k for i in ids for k in keys[i])))
        split = int(identity[:8],16)%4
        part = 'development' if split < 2 else ('validation' if split == 2 else 'assessment')
        if any(orbit(rows[i]) in protected for i in ids):
            part = 'assessment'
        for i in ids:
            row = dict(rows[i],partition=part,component=identity)
            strata[(part,len(row['board']),row['hero'])].append(row)
    selected,shortages = [],{}
    for part in PARTITIONS:
        for street,total in QUOTAS.items():
            for hero in (0,1):
                count = 1 if smoke else total//(4 if part == 'development' else 8)
                candidates = sorted(strata[(part,street,hero)],key=lambda p:p['priority'])
                if len(candidates) < count:
                    shortages[f'{part}/{street}/{hero}'] = dict(wanted=count,available=len(candidates))
                for row in candidates[:count]:
                    selected.append(dict(row,encounter_weight=len(candidates)/min(count,len(candidates)),
                        stratum_encounters=len(candidates)))
    return dict(status='COMPLETE' if not shortages else 'INCOMPLETE_COVERAGE',positions=selected,
        shortages=shortages,observed_positions=len(rows),
        strata={f'{p}/{s}/{h}':len(v) for (p,s,h),v in strata.items()},
        families=dict(Counter(row['family'] for row in rows)),
        selection='Bottom-k encounter hashes within partition/street/seat; frozen before target/error measurement.',
        weighting='N_partition_street_seat/n_selected; describes frozen production visitation, not played-hand population.',
        split='Connected flop/full-board suit orbits; static protected assessment orbits remain assessment.')


def references(runtime, roots, weights, position, *, samples=4096, check=lambda:None, seed_tag=0):
    hero = position['hero']
    policy = runtime.policy(roots[0])
    legal = np.asarray(roots[0].legal_actions_mask(),bool)
    tick = time.process_time()
    if len(position['board']) == 5:
        matrix = exact_river_q(runtime,roots,hero,check)
        means = weights@matrix
        covariance = np.zeros((3,3,3))
        raw = None
        exact = True
    else:
        mixture = weights[0]*.5+weights[1]*.25+weights[2]*.25
        raw = np.zeros((samples,3,3))
        for index in range(samples):
            check()
            rng = rng_for(2,seed_tag,index)
            hand = int(rng.choice(len(roots),p=mixture))
            ratios = weights[:,hand]/mixture[hand]
            for action in np.flatnonzero(legal):
                # Same independent stream per row across forced actions; retain covariance.
                value = runtime.direct(roots[hand].child(int(action)),hero,rng_for(3,seed_tag,index),check)
                raw[index,:,action] = ratios*value
                if action == 0:  # Known folding value does not need importance sampling.
                    raw[index,:,action] = value
        means = raw.mean(axis=0)
        covariance = np.asarray([np.cov(raw[:,r,:],rowvar=False,ddof=1)/samples for r in range(3)])
        matrix,exact = None,False
    transform = np.eye(3)-policy[:,None]
    transform[:,~legal] = 0
    advantage = means@transform
    advantage_cov = np.asarray([transform.T@c@transform for c in covariance])
    result = dict(exact=exact,reference_samples=0 if exact else samples,
        q=means.tolist(),advantage=advantage.tolist(),
        q_mean_covariance=covariance.tolist(),advantage_mean_covariance=advantage_cov.tolist(),
        cpu_seconds=time.process_time()-tick,
        scope='Frozen instantaneous strategies; terminal utilities. Not equilibrium/optimal action values.')
    return result,matrix,raw


def analyse_position(runtime, position, *, hidden=32, repeats=8, samples=4096,
                     arms=('current','average4','all_actions','uniform'), check=lambda:None, pilot=False):
    if hidden%4 or repeats<2 or hidden<4:
        raise ValueError('Nested design needs a multiple of four hidden draws and >=2 repeats')
    tag = int(digest([position['id'],pilot])[:8],16)
    started = time.process_time()
    hands,roots,weights,range_report = conditional_ranges(runtime,position,check)
    range_seconds = time.process_time()-started
    ref,matrix,raw_reference = references(runtime,roots,weights,position,samples=samples,check=check,seed_tag=tag)
    policy = runtime.policy(roots[0])
    legal = np.asarray(roots[0].legal_actions_mask(),bool)
    arrays = dict(range_weights=weights,opponent_hands=np.asarray(hands),
                  reference_mean=np.asarray(ref['advantage']),reference_cov=np.asarray(ref['advantage_mean_covariance']))
    if raw_reference is not None:
        arrays['reference_q_samples'] = raw_reference
    if matrix is not None:
        arrays['exact_q_by_hand'] = matrix
    records = []
    for fold in (0,1):
        ids = rng_for(4,tag,fold).choice(len(roots),size=hidden,p=weights[fold+1])
        arrays[f'fold{fold}_hand_ids'] = ids
        arrays[f'fold{fold}_block_ids'] = np.arange(hidden)
        outputs = {arm:np.zeros((hidden,repeats,3)) for arm in arms if arm != 'average4'}
        if 'current' not in outputs:
            raise ValueError('Production control is always required')
        costs = {arm:0. for arm in outputs}
        corrections,probabilities,critics = [],[],[]
        for block,hand in enumerate(ids):
            state = roots[int(hand)]
            critics.append(runtime.controls(state,position['hero'],fold)[0])
            for replicate in range(repeats):
                for arm in outputs:
                    check()
                    tick = time.process_time()
                    _,draw = runtime.estimate(state.clone(),position['hero'],fold,
                        rng_for(5,tag,fold,block,replicate),arm=arm,check=check)
                    costs[arm] += time.process_time()-tick
                    outputs[arm][block,replicate] = draw['advantage']
                    if arm == 'current':
                        corrections.append(abs(draw['correction']))
                        probabilities.append(draw['sample_probability'])
        for arm,data in outputs.items():
            arrays[f'fold{fold}_{arm}'] = data
        if 'average4' in arms:
            # Four independent hidden draws, not four rollouts conditional on one hand.
            arrays[f'fold{fold}_average4'] = outputs['current'].reshape(hidden//4,4,repeats,3).mean(axis=1)
            costs['average4'] = costs['current']
        # Production sampling costs are measured independently of range/reference setup.
        mean_cost = {arm:cost/(hidden*repeats if arm != 'average4' else hidden//4*repeats)
                     for arm,cost in costs.items()}
        q_critic = np.mean(critics,axis=0)
        expected_q = np.asarray(ref['q'])[fold+1]
        record = dict(fold=fold,draws=hidden,repeats=repeats,cpu_seconds=costs,
            cpu_seconds_per_target=mean_cost,critic_mean_q=q_critic.tolist(),
            critic_mean_discrepancy=(q_critic-expected_q).tolist(),
            critic_caveat=f'Mean over {hidden} hidden draws; discrepancy also has hidden-draw and reference uncertainty.',
            correction_abs_p95=float(np.quantile(corrections,.95)),
            correction_abs_p99=float(np.quantile(corrections,.99)),
            minimum_root_sampling_probability=float(min(probabilities)))
        if matrix is not None:
            record['critic_river_mse_on_draws'] = float(np.mean((np.asarray(critics)[:,legal]-matrix[ids][:,legal])**2))
        records.append(record)
    gap = np.asarray(ref['advantage'])[1:]-np.asarray(ref['advantage'])[0]
    range_report['production_minus_opponent_reach_advantage_bb'] = (gap*runtime.bb_scale).tolist()
    gap_se = np.zeros((2,3))
    if raw_reference is not None:
        transform = np.eye(3)-policy[:,None]
        transform[:,~legal] = 0
        for fold in (0,1):
            difference = (raw_reference[:,fold+1]-raw_reference[:,0])@transform
            gap_se[fold] = difference.std(axis=0,ddof=1)/np.sqrt(len(difference))*runtime.bb_scale
    range_report['production_minus_opponent_reach_advantage_se_bb'] = gap_se.tolist()
    initial = runtime.update_strategy(roots[0],np.zeros(3))
    updated = [runtime.update_strategy(roots[0],a) for a in np.asarray(ref['advantage'])]
    result = dict(id=position['id'],position=position,range=range_report,reference=ref,folds=records,
        legal_mask=legal.tolist(),policy=policy.tolist(),bb_scale=runtime.bb_scale,
        raw_utility_units='Normalized by game.max_utility; multiply by bb_scale for big blinds.',
        initial_update_strategy=initial.tolist(),reference_update_strategies=[p.tolist() for p in updated],
        stage_cpu_seconds=dict(ranges=range_seconds,reference=ref['cpu_seconds'],
            sampling=sum(sum(v for k,v in r['cpu_seconds'].items() if k != 'average4') for r in records)),
        total_cpu_seconds=time.process_time()-started,pilot=pilot,
        beta_one_alias=runtime.config['fixed_control_variate_beta'] == 1.)
    # Hypothetical target updates are local diagnostics, never replacement policies.
    result['sample_mean_update_strategies'] = {f'fold{fold}/{arm}':
        runtime.update_strategy(roots[0],arrays[f'fold{fold}_{arm}'].mean(axis=(0,1))).tolist()
        for fold in (0,1) for arm in arms}
    return result,arrays
