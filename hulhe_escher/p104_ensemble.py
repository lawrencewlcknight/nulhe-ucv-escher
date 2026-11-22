"""Twenty independent P104 policy fits and a portable probability ensemble."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from open_spiel.python import policy as spiel_policy

from unbiased_escher.holdem_fitting import policy_validation_mask
from unbiased_escher.replay_grouping import grouped_policy_data
from .bottleneck_refit import model_hash_from_state
from .checkpointing import LoadedHULHEPolicy, sha256_file
from .game import load_hulhe_game, serialisable_game_definition
from .p104_policy_refit import fit_path
from .p104_protocol import ITERATION, POLICY_SHA256, POLICY_BUDGETS
from .policy_replay_artifacts import load_policy_replay, safe_path
from .policy_sampling_experiment import model_hash
from .position_investigation.common import deadline_after
from .train import write_json


PROTOCOL = 'p104_ensemble20_v1'
FIT_SEEDS = tuple(range(1045001, 1045021))
SAMPLING_SEEDS = tuple(range(1046001, 1046021))
PILOT_SEED = 1049701
CONFIRMATION_SEED = 1049702


def fit(state, policy, output, *, threads=8, max_seconds=9000, smoke=False):
    if threads < 1:
        raise ValueError('Positive thread count required')
    torch.set_num_threads(threads)
    check = deadline_after(max_seconds)
    started = time.monotonic()
    buffer, metadata, _ = load_policy_replay(state)
    loaded = LoadedHULHEPolicy(load_hulhe_game(), policy)
    payload, config = loaded.checkpoint, metadata['config']
    iteration = metadata['solver']['num_iteration']
    if (metadata['game'] != serialisable_game_definition()
            or metadata['encoder'] != payload['feature_encoder']
            or config != payload['training_config'] or iteration != payload['outer_iteration']
            or model_hash(loaded.model) != model_hash_from_state(
                metadata['average_policy_trainer']['model'], loaded.model)):
        raise ValueError('Source policy and replay identities differ')
    if not config['average_policy_reset_each_fit']:
        raise ValueError('Requires reset-each-fit source')
    if not smoke and (iteration != ITERATION or sha256_file(policy) != POLICY_SHA256):
        raise ValueError('Production experiment requires pinned P104')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    budgets = (2, 4, 8) if smoke else POLICY_BUDGETS
    report = dict(status='FITTING', protocol=PROTOCOL, smoke=smoke,
        source_policy_sha256=sha256_file(policy), source_iteration=iteration,
        source_manifest_sha256=sha256_file(Path(state)/'manifest.json'),
        fit_seeds=list(FIT_SEEDS), sampling_seeds=list(SAMPLING_SEEDS),
        budgets=list(budgets), members=[], architecture=payload['policy_model'],
        learning_rate=config['average_policy_learning_rate'],
        batch_size=config['ave_policy_batch_size'], threads=threads,
        selection='Minimum historical-validation weighted KL per seed; ties prefer fewer updates.',
        aggregation='Equal arithmetic mean of masked action probabilities at every decision.',
        limitations=['All members share one replay and upstream learner.',
            'All-replay refits have no independent replay holdout; fresh deals test playing strength.',
            'Comparison with saved P104 combines ensemble and export-method effects; it does not isolate ensembling.'])
    write_json(output/'experiment.json', report)
    try:
        data, coverage = grouped_policy_data(buffer, iteration, config['gamma'], 'cpu',
            context_columns=loaded.feature_encoder.replay_context_columns())
        held = policy_validation_mask(data[0], config['policy_validation_fraction'])
        train, valid, all_ids = np.flatnonzero(~held), np.flatnonzero(held), np.arange(len(held))
        if not len(train) or not len(valid):
            raise ValueError('Empty fitting/validation partition')
        report.update(coverage=coverage, partition_sizes=dict(training=len(train), historical_validation=len(valid)),
            partition_sha256={name:hashlib.sha256(ids.tobytes()).hexdigest()
                for name, ids in [('training', train), ('historical_validation', valid)]})
        for seed, sampling_seed in zip(FIT_SEEDS, SAMPLING_SEEDS):
            check()
            torch.manual_seed(seed)
            initial = deepcopy(loaded.model)
            initial.reset_parameters()
            initial_hash = model_hash(initial)
            args = dict(seed=sampling_seed, lr=report['learning_rate'], batch=report['batch_size'], check=check)
            control, selection, _ = fit_path(initial, data, train, budgets=budgets, valid=valid, **args)
            del control
            receipt = dict(seed=seed, sampling_seed=sampling_seed, initial_model_sha256=initial_hash,
                selected_update=selection['selected_update'], selection=selection)
            # Persist the budget before touching the final all-replay fit.
            receipt_path = output/f'budget_{seed}.json'
            write_json(receipt_path, receipt)
            model, final_fit, _ = fit_path(initial, data, all_ids,
                budgets=(receipt['selected_update'],), **args)
            details = dict(fit_seed=seed, sampling_seed=sampling_seed, initial_model_sha256=initial_hash,
                selected_update=receipt['selected_update'], source_sha256=report['source_policy_sha256'],
                group_sampling='objective_mass', learning_rate=report['learning_rate'])
            cp = deepcopy(payload)
            cp.update(policy_model=model.checkpoint_metadata(),
                policy_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                checkpoint_kind=PROTOCOL, policy_refit=details)
            path = output/f'member_{seed}.pkl'
            with path.open('xb') as handle:
                pickle.dump(cp, handle, pickle.HIGHEST_PROTOCOL)
            report['members'].append(dict(**details, checkpoint=path.name, sha256=sha256_file(path),
                budget=receipt_path.name, budget_sha256=sha256_file(receipt_path),
                selection=selection, fit=final_fit))
            report['elapsed_seconds'] = time.monotonic()-started
            write_json(output/'experiment.json', report)
            print(json.dumps(dict(member=len(report['members']), total=20,
                selected_update=receipt['selected_update'], elapsed_seconds=report['elapsed_seconds'])), flush=True)
        report['status'] = 'COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(output/'experiment.json', report)
    # Portable relative-path bundle; the evaluator additionally checks fit receipts.
    write_json(output/'ensemble.json', dict(protocol=PROTOCOL, smoke=smoke,
        source_policy_sha256=report['source_policy_sha256'], aggregation='mean_probabilities',
        experiment_sha256=sha256_file(output/'experiment.json'),
        members=[{k:r[k] for k in ('fit_seed','checkpoint','sha256')} for r in report['members']]))
    return report


def verified_members(manifest, *, policy=None, smoke=None):
    manifest = Path(manifest)
    root = manifest.parent
    bundle = json.loads(manifest.read_text())
    if (bundle['protocol'] != PROTOCOL or bundle['aggregation'] != 'mean_probabilities'
            or bundle['experiment_sha256'] != sha256_file(root/'experiment.json')):
        raise ValueError('Invalid ensemble manifest or fit checksum')
    report = json.loads((root/'experiment.json').read_text())
    if (report['status'] != 'COMPLETE' or report['protocol'] != PROTOCOL
            or report['fit_seeds'] != list(FIT_SEEDS) or report['sampling_seeds'] != list(SAMPLING_SEEDS)
            or bundle['smoke'] != report['smoke'] or (smoke is not None and report['smoke'] != smoke)
            or report['source_policy_sha256'] != bundle['source_policy_sha256']):
        raise ValueError('Requires completed twenty-member experiment with matching smoke mode')
    if policy is not None and sha256_file(policy) != report['source_policy_sha256']:
        raise ValueError('Source policy mismatch')
    if not report['smoke'] and (report['source_policy_sha256'] != POLICY_SHA256 or report['source_iteration'] != ITERATION):
        raise ValueError('Production ensemble requires pinned P104')
    budgets = [2,4,8] if report['smoke'] else list(POLICY_BUDGETS)
    if report['budgets'] != budgets or len(report['members']) != 20 or len(bundle['members']) != 20:
        raise ValueError('Wrong budget grid or ensemble size')
    paths, initial_hashes = [], []
    for seed, sampling_seed, row, spec in zip(FIT_SEEDS, SAMPLING_SEEDS, report['members'], bundle['members']):
        path = safe_path(root, row['checkpoint'])
        receipt_path = safe_path(root, row['budget'])
        receipt = json.loads(receipt_path.read_text())
        history = row['selection']['validation_history']
        if (row['fit_seed'] != seed or row['sampling_seed'] != sampling_seed
                or spec != {k:row[k] for k in ('fit_seed','checkpoint','sha256')}
                or sha256_file(path) != row['sha256'] or sha256_file(receipt_path) != row['budget_sha256']
                or receipt['seed'] != seed or receipt['sampling_seed'] != sampling_seed
                or receipt['initial_model_sha256'] != row['initial_model_sha256']
                or receipt['selection'] != row['selection']
                or [h['update'] for h in history] != budgets
                or not all(np.isfinite(h['weighted_kl']) for h in history)):
            raise ValueError('Member identity, selection or checksum mismatch')
        selected = min(history, key=lambda h:(h['weighted_kl'],h['update']))['update']
        if (selected != receipt['selected_update'] or selected != row['selected_update']
                or row['fit']['selected_update'] != selected or row['fit']['updates_completed'] != selected
                or row['fit']['validation_history'] or row['fit']['validation_groups'] != 0
                or row['fit']['training_groups'] != sum(report['partition_sizes'].values())
                or row['fit']['group_sampling'] != 'objective_mass'
                or row['fit']['sampling_seed'] != sampling_seed
                or row['learning_rate'] != report['learning_rate']):
            raise ValueError('Member did not refit complete replay at its frozen budget')
        paths.append(path)
        initial_hashes.append(row['initial_model_sha256'])
    if len(set(paths)) != 20 or len(set(initial_hashes)) != 20:
        raise ValueError('Ensemble members must have distinct files and initial weights')
    return paths, report


class EnsemblePolicy(spiel_policy.Policy):
    """One shared encoding, vectorized networks, mean AFTER masked softmax."""
    def __init__(self, game, manifest):
        paths, self.fit_report = verified_members(manifest)
        super().__init__(game, list(range(game.num_players())))
        members = [LoadedHULHEPolicy(game, path) for path in paths]
        first = members[0]
        keys = ('game','feature_encoder','policy_model','input_size','num_actions','training_config')
        for member, row in zip(members, self.fit_report['members']):
            if any(member.checkpoint[k] != first.checkpoint[k] for k in keys):
                raise ValueError('Ensemble models have incompatible encoders or architectures')
            details = member.checkpoint.get('policy_refit', {})
            if (member.checkpoint['checkpoint_kind'] != PROTOCOL
                    or any(details.get(k) != row[k] for k in
                        ('fit_seed','sampling_seed','selected_update','initial_model_sha256','learning_rate'))
                    or details.get('source_sha256') != self.fit_report['source_policy_sha256']):
                raise ValueError('Member payload differs from manifest')
        self.feature_encoder = first.feature_encoder
        self.template = first.model
        self.params, self.buffers = torch.func.stack_module_state([m.model for m in members])
        for p in self.params.values():
            p.requires_grad_(False)
        self._forward = torch.vmap(lambda p,b,x: torch.func.functional_call(self.template,(p,b),(x,)),
                                  in_dims=(0,0,None), randomness='error')

    def probabilities(self, encoded, mask):
        with torch.no_grad():
            logits = self._forward(self.params, self.buffers, encoded)
            members = torch.softmax(logits.masked_fill(~mask, -1e21), dim=-1)
            result = members.mean(dim=0)
        if not torch.isfinite(result).all():
            raise RuntimeError('Nonfinite ensemble probabilities')
        return result

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        actions = state.legal_actions(player)
        if not actions:
            return {}
        encoded = (state.information_state_tensor(player) if self.feature_encoder is None
                   else self.feature_encoder.information_state(state, player))
        probabilities = self.probabilities(torch.as_tensor(encoded, dtype=torch.float32),
            torch.as_tensor(state.legal_actions_mask(player), dtype=torch.bool)).cpu().numpy()
        return {a:float(probabilities[a]) for a in actions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state','policy','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--max-seconds', type=float, default=9000)
    p.add_argument('--smoke', action='store_true')
    fit(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
