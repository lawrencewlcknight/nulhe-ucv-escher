"""Matched training-only and all-replay exports; no collection or source writes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from unbiased_escher.holdem_fitting import policy_validation_mask, weighted_policy_ce
from unbiased_escher.policy_sampling import PolicyGroupSampler
from unbiased_escher.replay_grouping import grouped_policy_data, exact_group_indices
from .bottleneck_refit import metrics, model_hash_from_state
from .checkpointing import LoadedHULHEPolicy, sha256_file
from .game import load_hulhe_game, serialisable_game_definition
from .p104_protocol import POLICY_SEEDS, POLICY_BUDGETS, LEGACY_BUDGETS, ITERATION
from .policy_replay_artifacts import load_policy_replay
from .policy_sampling_experiment import cells_for_groups, model_hash
from .position_investigation.common import deadline_after
from .train import write_json


def fit_path(initial, data, train, *, seed, lr, batch, budgets, check, valid=None, legacy=()):
    """No validation access during the final all-replay fit. Budget fixed by caller."""
    model = deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = PolicyGroupSampler(train, data[3], batch, mode='objective_mass', seed=seed)
    history, best, old_best = [], None, None
    entropy = None
    if valid is not None:
        y, w = data[1][valid].double(), data[3][valid].double()
        entropy = float((-(y*y.clamp_min(1e-30).log()).sum(-1)*w).sum()/w.sum())
    started = time.monotonic()
    for step in range(1, max(budgets)+1):
        check(); model.train()
        idx, factors = sampler.sample()
        logits = model(data[0][idx]).masked_fill(data[2][idx] != 1, -1e20)
        loss = (-(data[1][idx]*torch.log_softmax(logits, -1)).sum(-1)*factors).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite policy loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if valid is not None and step in budgets:
            model.eval()
            kl = weighted_policy_ce(model, *data, valid, batch)-entropy
            if not np.isfinite(kl): raise FloatingPointError('Nonfinite validation KL')
            history.append(dict(update=step, weighted_kl=kl))
            if best is None or kl < best[0]: best = (kl, step, deepcopy(model.state_dict()))
            if step in legacy and (old_best is None or kl < old_best[0]):
                old_best = (kl, step, deepcopy(model.state_dict()))
    if best is not None: model.load_state_dict(best[2])
    model.eval()
    return model, dict(updates_completed=max(budgets), selected_update=best[1] if best else max(budgets),
        validation_history=history, fit_seconds=time.monotonic()-started,
        group_sampling='objective_mass', sampling_seed=seed,
        training_groups=len(train), validation_groups=0 if valid is None else len(valid)), old_best


def run(state, policy, output, *, threads=8, smoke=False, max_seconds=1800):
    check = deadline_after(max_seconds); torch.set_num_threads(threads)
    if threads < 1: raise ValueError('Positive thread count required')
    buffer, metadata, _ = load_policy_replay(state)
    loaded = LoadedHULHEPolicy(load_hulhe_game(), policy)
    config, payload = metadata['config'], loaded.checkpoint
    iteration = metadata['solver']['num_iteration']
    if (metadata['game'] != serialisable_game_definition() or metadata['encoder'] != payload['feature_encoder']
        or config != payload['training_config'] or iteration != payload['outer_iteration']
        or model_hash(loaded.model) != model_hash_from_state(metadata['average_policy_trainer']['model'], loaded.model)):
        raise ValueError('Source policy and recovery identities differ')
    if not config['average_policy_reset_each_fit']: raise ValueError('Requires reset-each-fit source')
    if not smoke and iteration != ITERATION: raise ValueError('Production protocol requires P104')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FITTING', protocol='p104_policy_export_v1', smoke=smoke, results=[],
        source_policy_sha256=sha256_file(policy), source_manifest_sha256=sha256_file(Path(state)/'manifest.json'),
        source_iteration=iteration, fit_seeds=list(POLICY_SEEDS),
        selection_rule='Per-seed minimum historical-validation weighted KL; ties prefer fewer updates. Same budget for matched control/refit.',
        limitations=['All-replay metrics, including former validation groups, are in-sample fitting diagnostics.',
            'Fresh deal-group intervals condition on these two fits and this replay; they do not measure training-seed uncertainty.'])
    started = time.monotonic()
    write_json(output/'experiment.json', report)
    try:
        data, coverage = grouped_policy_data(buffer, iteration, config['gamma'], 'cpu',
            context_columns=loaded.feature_encoder.replay_context_columns())
        held = policy_validation_mask(data[0], config['policy_validation_fraction'])
        train, valid, all_ids = np.flatnonzero(~held), np.flatnonzero(held), np.arange(len(held))
        if not len(train) or not len(valid): raise ValueError('Empty fitting/validation partition')
        first, inv = exact_group_indices(buffer.infostate_buf)
        if not np.array_equal(first, data[0].rows): raise ValueError('Group ordering mismatch')
        counts = np.bincount(inv, minlength=len(first)); del inv
        cells = cells_for_groups(data[0], loaded.feature_encoder)
        budgets = (2,4,8) if smoke else POLICY_BUDGETS
        legacy = (4,8) if smoke else LEGACY_BUDGETS
        report.update(coverage=coverage, budgets=list(budgets), legacy_budgets=list(legacy),
            partition_sizes=dict(training=len(train), historical_validation=len(valid)),
            partition_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in
                              [('training',train),('historical_validation',valid)]})
        def export(model, name, details):
            cp = deepcopy(payload)
            cp.update(policy_model=model.checkpoint_metadata(),
                policy_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                checkpoint_kind='p104_policy_export_diagnostic',
                policy_refit=dict(**details, source_sha256=report['source_policy_sha256']))
            path = output/f'{name}.pkl'
            with path.open('xb') as handle: pickle.dump(cp, handle, pickle.HIGHEST_PROTOCOL)
            row = dict(**details, name=name, checkpoint=path.name, sha256=sha256_file(path),
                metrics={key:metrics(model,data,ids,cells,counts) for key,ids in
                    [('historical_training',train),('historical_validation',valid)]})
            report['results'].append(row)
            write_json(output/'experiment.json', report)
        for seed in POLICY_SEEDS:
            check(); torch.manual_seed(seed)
            initial = deepcopy(loaded.model); initial.reset_parameters()
            initial_hash = model_hash(initial)
            args = dict(seed=seed, lr=config['average_policy_learning_rate'],
                        batch=config['ave_policy_batch_size'], check=check)
            control, fit, old = fit_path(initial,data,train,budgets=budgets,valid=valid,legacy=legacy,**args)
            selected = fit['selected_update']
            # Publish selection BEFORE starting the all-replay fit or inspecting its metrics.
            write_json(output/f'budget_seed{seed}.json',dict(seed=seed,selected_update=selected,
                legacy_selected_update=old[1], initial_model_sha256=initial_hash,
                validation_history=fit['validation_history']))
            common = dict(fit_seed=seed, initial_model_sha256=initial_hash, learning_rate=args['lr'])
            export(control, f'control_{seed}', dict(**common,arm='control',fit=fit))
            legacy_model = deepcopy(initial); legacy_model.load_state_dict(old[2]); legacy_model.eval()
            export(legacy_model,f'legacy_{seed}',dict(**common,arm='legacy_reference',
                fit=dict(selected_update=old[1],note='Original-grid choice from the same training path; descriptive reference.')))
            refit, final_fit, _ = fit_path(initial,data,all_ids,budgets=(selected,),**args)
            export(refit,f'all_replay_{seed}',dict(**common,arm='all_replay',fit=final_fit))
        report['source_metrics'] = {key:metrics(loaded.model,data,ids,cells,counts) for key,ids in
            [('historical_training',train),('historical_validation',valid)]}
        report['status'] = 'COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(output/'experiment.json', report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','policy','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-seconds',type=float,default=1800)
    run(**vars(p.parse_args()))


if __name__=='__main__': main()
