"""Cluster-aware summaries for frozen regret supervision (utilities in bb)."""
from collections import defaultdict

import numpy as np

from .supervision_positions import rng_for


def decomposition(data):
    """H independent hidden draws, R independent continuations per draw."""
    data = np.asarray(data, float)
    h, r, _ = data.shape
    if h < 2 or r < 2 or not np.all(np.isfinite(data)):
        raise ValueError('Need finite targets, >=2 hidden blocks and >=2 continuations')
    within = data.var(axis=1, ddof=1).mean(axis=0)
    block_variance = data.mean(axis=1).var(axis=0, ddof=1)
    between_raw = block_variance - within/r
    between = np.maximum(between_raw, 0)
    return dict(mean=data.mean(axis=(0, 1)), within=within, between=between,
                between_unclipped=between_raw, variance=within+between,
                mean_variance=block_variance/h)


def measurements(data, reference, ref_cov, cost, budget):
    d = decomposition(data)
    bias = d['mean']-reference
    noise = d['mean_variance']+np.diag(ref_cov)
    # Truncated noise correction is a diagnostic, not a proof of bias/unbiasedness.
    bias_squared = np.maximum(bias*bias-noise, 0)
    return dict(**d, bias=bias, bias_se=np.sqrt(noise),
                bias_squared_estimate=bias_squared,
                equal_cpu_mse=bias_squared+d['variance']*cost/budget,
                variance_times_cpu=d['variance']*cost)


def summarise_position(report, arrays, *, bootstrap=300):
    mask = np.asarray(report['legal_mask'], bool)
    scale = report['bb_scale']
    output = dict(id=report['id'], position=report['position'], folds=[])
    replicas = {}
    rng = rng_for(8, int(report['position']['priority'][:8], 16))
    ref_mean = arrays['reference_mean']*scale
    if 'reference_q_samples' in arrays:
        transform = np.eye(3)-np.asarray(report['policy'])[:,None]
        transform[:,~mask] = 0
        raw = (arrays['reference_q_samples']@transform*scale).reshape(-1,9)
        joint_cov = np.cov(raw,rowvar=False,ddof=1)/len(raw)
        ref_draws = rng.multivariate_normal(ref_mean.ravel(),joint_cov,size=bootstrap).reshape(bootstrap,3,3)
    else:
        ref_draws = np.broadcast_to(ref_mean,(bootstrap,3,3))
    for fold, record in enumerate(report['folds']):
        reference = arrays['reference_mean'][fold+1]*scale
        covariance = arrays['reference_cov'][fold+1]*scale**2
        costs = record['cpu_seconds_per_target']
        current = arrays[f'fold{fold}_current']*scale
        # Equal compute for 256 independent production estimates, excluding setup.
        budget = 256*costs['current']
        metrics = {}
        for arm, cost in costs.items():
            data = arrays[f'fold{fold}_{arm}']*scale
            m = measurements(data, reference, covariance, cost, budget)
            if arm == 'average4':
                empirical_variance = m['variance'].copy()
                m = measurements(current,reference,covariance,cost/4,budget)
                m['empirical_four_sample_variance'] = empirical_variance
                for key in ('within','between','between_unclipped','variance'):
                    m[key] = m[key]/4
                m['averaging_note'] = 'Four independent draws divide variance by four and multiply cost by four; equal-CPU MSE equals current analytically.'
            mean = m['mean']
            reference_se = np.sqrt(np.diag(covariance))
            confident_sign = mask & (np.abs(reference) > 1.96*reference_se)
            m['wrong_sign_fraction_when_reference_resolved'] = (
                float(np.mean((mean[confident_sign]*reference[confident_sign]) < 0))
                if confident_sign.any() else None)
            best = int(np.argmax(np.where(mask, reference, -np.inf)))
            resolved = all(reference[best]-reference[a] > 1.96*np.sqrt(max(
                0., covariance[best,best]+covariance[a,a]-2*covariance[best,a]))
                for a in np.flatnonzero(mask) if a != best)
            m['reference_ranking_resolved'] = bool(resolved)
            m['wrong_best_action'] = (bool(np.argmax(np.where(mask,mean,-np.inf)) != best)
                                     if resolved else None)
            delta = (arrays[f'fold{fold}_{arm}'].mean(axis=(0,1))
                     - arrays[f'fold{fold}_current'].mean(axis=(0,1)))*scale
            m['paired_mean_delta_vs_current_bb'] = delta
            if arm == 'average4':
                m['paired_mean_delta_se_bb'] = np.zeros(3)
            else:
                diff = (data-current).mean(axis=1)
                m['paired_mean_delta_se_bb'] = diff.std(axis=0,ddof=1)/np.sqrt(len(diff))
            policy = np.asarray(report['sample_mean_update_strategies'][f'fold{fold}/{arm}'])
            base = np.asarray(report['initial_update_strategy'])
            q = np.asarray(report['reference']['q'])[fold+1]*scale
            m['hypothetical_update_value_change_bb'] = float((policy-base)@q)
            metrics[arm] = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()}
            replicas.setdefault(arm, []).append([])
        # Shared block indices and reference perturbations preserve arm covariance.
        # Finite-reference mean uncertainty uses a Gaussian approximation; raw draws retained.
        for b in range(bootstrap):
            indices = rng.integers(len(current),size=len(current))
            ref = ref_draws[b,fold+1]
            for arm,cost in costs.items():
                if arm == 'average4':
                    # Exact iid averaging identity avoids selecting this arm on
                    # the noise of eight grouped variance estimates.
                    data = current[indices]
                    cost = cost/4
                else:
                    data = arrays[f'fold{fold}_{arm}'][indices]*scale
                m = measurements(data,ref,covariance,cost,budget)
                replicas[arm][-1].append(float(m['equal_cpu_mse'][mask].mean()))
        output['folds'].append(dict(fold=fold, equal_cpu_budget_seconds=budget, arms=metrics,
            reference_max_95ci_half_width_bb=float(1.96*np.sqrt(np.diag(covariance)[mask]).max())))
    # Do not count folds as independent positions.
    replicas = {arm:np.mean(v,axis=0) for arm,v in replicas.items()}
    return output,replicas


def scores(rows, arm):
    values,weights = [],[]
    for row in rows:
        mask = np.asarray(row['legal_mask'],bool)
        summary = row['statistics']
        values.append(np.mean([np.asarray(f['arms'][arm]['equal_cpu_mse'])[mask].mean()
                               for f in summary['folds']]))
        weights.append(row['position']['encounter_weight'])
    return float(np.average(values,weights=weights))


def select(rows):
    if not rows or any(r['position']['partition'] != 'validation' for r in rows):
        raise ValueError('Selection requires validation positions only')
    arms = list(rows[0]['statistics']['folds'][0]['arms'])
    values = {arm:scores(rows,arm) for arm in arms}
    chosen = min(arms,key=lambda arm:values[arm])
    return dict(selected=chosen, validation_equal_cpu_mse_bb2=values,
                rule='Lowest encounter-weighted equal-CPU estimated MSE; current wins exact ties.',
                assessment_used=False, minimum_practical_reduction=.25,
                material_bias_margin_bb=.05)


def aggregate(rows, replicas, candidate, *, bootstrap=2000):
    """Resample board components, then paired nested-target bootstrap replicates."""
    if not rows:
        raise ValueError('Empty assessment')
    components = defaultdict(list)
    for i,r in enumerate(rows):
        components[r['position']['component']].append(i)
    groups = list(components.values())
    rng = rng_for(9, len(rows))
    reductions = []
    for _ in range(bootstrap):
        ids = [i for g in rng.integers(len(groups),size=len(groups)) for i in groups[g]]
        w = [rows[i]['position']['encounter_weight'] for i in ids]
        a,c = [],[]
        # Repeated copies of a component share the same sampled measurement.
        draws = {i:int(rng.integers(len(replicas[rows[i]['id']]['current']))) for i in set(ids)}
        for i in ids:
            rep = replicas[rows[i]['id']]
            a.append(rep[candidate][draws[i]])
            c.append(rep['current'][draws[i]])
        baseline = np.average(c,weights=w)
        reductions.append(1-np.average(a,weights=w)/baseline if baseline > 0 else 0.)
    baseline = scores(rows,'current')
    point = 1-scores(rows,candidate)/baseline if baseline > 0 else 0.
    interval = np.quantile(reductions,[.025,.975]).tolist()
    bias_agreement = all(
        np.all((np.abs(f['arms'][candidate]['bias'])+1.96*np.asarray(f['arms'][candidate]['bias_se']))[
            np.asarray(r['legal_mask'],bool)] <= .05)
        for r in rows for f in r['statistics']['folds'])
    return dict(positions=len(rows),components=len(groups),candidate=candidate,
        estimated_equal_cpu_mse_reduction=point,interval95=interval,
        bias_agreement_within_point05_bb=bias_agreement,
        uncertainty='Board-component bootstrap with paired hidden-block resampling and Gaussian reference-mean uncertainty; conditional on frozen P104.',
        bias_caveat='Squared-bias estimate is noise-corrected and truncated; limited repetitions can leave it unresolved.',
        evidence_for_short_continuation=bool(candidate != 'current' and point >= .25
            and interval[0] > 0 and bias_agreement and len(groups) >= 8))
