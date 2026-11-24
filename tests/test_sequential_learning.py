"""Production protected routing, objective, checkpoint/actor and Batch contracts."""
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import torch

from hulhe_escher.config import make_solver, smoke_config, validate_config
from hulhe_escher.sequential_model import ProtectedSequentialNetwork
from hulhe_escher.sequential_training import sequential_config, StreetSampler, ProtectedFit
from hulhe_escher.checkpointing import save_policy_checkpoint, LoadedHULHEPolicy
from hulhe_escher.game import load_hulhe_game


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_real_learning_and_export(tmp_path):
    solver = make_solver(sequential_config(small=True), seed=7)
    original = make_solver(smoke_config(), seed=7)
    for trainer in [*solver.regret_trainers, solver.ave_policy_trainer]:
        assert isinstance(trainer.model, ProtectedSequentialNetwork)
        assert sum(p.numel() for p in trainer.model.parameters()) == 1496067
    for a, b in zip(solver.q_value_trainer.members, original.q_value_trainer.members):
        assert a.model.checkpoint_metadata() == b.model.checkpoint_metadata()
    assert solver.calibration_trainer.model.checkpoint_metadata() == original.calibration_trainer.model.checkpoint_metadata()
    solver.iteration(); solver.train_average_policy()
    for trainer in solver.regret_trainers:
        assert trainer.last_fit_stats['protected_training']['coefficient'] == .3
        assert sum(trainer.last_fit_stats['protected_training']['specialist_updates_by_street']) == 1
        assert all(p.requires_grad for p in trainer.model.parameters())
    fit = solver.ave_policy_trainer.last_fit_diagnostics
    assert fit['group_sampling'] == 'objective_mass' and fit['policy_sampling_seed'] == [7, 1]
    assert fit['protected_training']['policy_preservation'] == 'refreshed_forward_KL'
    path = tmp_path/'policy.pkl'
    save_policy_checkpoint(solver, path, seed=7, config=sequential_config(small=True), checkpoint_row={})
    loaded = LoadedHULHEPolicy(load_hulhe_game(), path)
    x = torch.randn(3, solver.infostate_size)
    torch.testing.assert_close(loaded.model(x), solver.ave_policy_trainer.model(x), rtol=0, atol=0)
    state = load_hulhe_game().new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    assert sum(loaded.action_probabilities(state).values()) == pytest.approx(1.)
    with path.open('rb') as f:
        payload = pickle.load(f)
    payload['policy_model']['specialist_inner'] += 1
    with path.open('wb') as f:
        pickle.dump(payload, f)
    with pytest.raises(ValueError, match='metadata mismatch'):
        LoadedHULHEPolicy(load_hulhe_game(), path)


@pytest.mark.parametrize('kind', ['regret', 'policy'])
def test_adam_freezing_with_real_protection_and_stale_momentum(kind):
    solver = make_solver(sequential_config(small=True), seed=4)
    trainer = solver.regret_trainers[0] if kind == 'regret' else solver.ave_policy_trainer
    trainer.train_steps = 5
    trainer.last_fit_stats = dict(max_gradient_norm=0., clipped_updates=0)
    n = 32
    x = torch.randn(n, trainer.input_size)
    columns = trainer.feature_encoder.replay_context_columns()[1]
    x[:, columns] = torch.eye(4)[torch.arange(n) % 4]
    masks = torch.ones(n, 3)
    targets = torch.softmax(torch.randn(n, 3), -1) if kind == 'policy' else torch.randn(n, 3)
    fit = ProtectedFit(trainer, x, targets, masks, np.arange(n),
        rng=np.random.default_rng(4), kind=kind, weights=torch.ones(n) if kind == 'policy' else None)
    fit.warmup = 2
    fit.update(1); fit.update(2)
    before = [{k:v.clone() for k,v in block.state_dict().items()} for block in trainer.model.specialists]
    shared = {k:v.clone() for k,v in trainer.model.trunk.state_dict().items()}
    fit.sampler.street = lambda exclude=None: 0
    fit.update(3); fit.update(4)
    for block, old in zip(trainer.model.specialists[1:], before[1:]):
        assert all(torch.equal(v, old[k]) for k,v in block.state_dict().items())
        assert all(p.grad is None for p in block.parameters())
    assert any(not torch.equal(v, before[0][k]) for k,v in trainer.model.specialists[0].state_dict().items())
    assert any(not torch.equal(v, shared[k]) for k,v in trainer.model.trunk.state_dict().items())
    assert fit.aux_updates == 2
    if kind == 'policy':
        assert fit.aux_sum > 0 and all(not p.requires_grad for p in fit.teacher.parameters())
    fit.finish()
    assert all(p.requires_grad for p in trainer.model.parameters())


def test_mass_sampling_preserves_primary_objective_and_excludes_heldout():
    x = torch.eye(4).repeat_interleave(2, 0)
    weights = torch.tensor([1., 2., 3., 4., 5., 6., 7., 999.])
    ids = np.arange(7)
    sampler = StreetSampler(x, ids, slice(0,4), weights, np.random.default_rng(0))
    np.testing.assert_array_equal(sampler.mass, [3, 7, 11, 7])
    assert sampler.factor == 4
    samples = sampler.draw(100000)
    assert 7 not in samples
    np.testing.assert_allclose(np.bincount(samples, minlength=7)/len(samples), weights[:7]/28, atol=.005)
    aux = sampler.draw(1000, exclude=2)
    assert not np.isin(aux, [4, 5, 7]).any()
    # E[factor * loss(draw)] = sum(weight * loss)/number_of_groups.
    loss = np.arange(7)
    assert (loss[samples]*sampler.factor).mean() == pytest.approx(float((weights[:7]*torch.arange(7)).sum()/7), rel=.01)


def test_single_street_and_invalid_streets():
    x = torch.tensor([[1., 0, 0, 0]] * 4)
    sampler = StreetSampler(x, np.arange(4), slice(0,4), None, np.random.default_rng(1))
    assert sampler.street() == 0 and len(sampler.draw(4, exclude=0)) == 0
    with pytest.raises(ValueError, match='one-hot'):
        StreetSampler(x*0, np.arange(4), slice(0,4), None, np.random.default_rng(1))


def test_actor_reconstructs_and_collects_sequential_network():
    from unbiased_escher.grouped_parallel_solver import StructuredGroupedTraversalWorker
    from unbiased_escher.parallel_solver import ParallelUnbiasedControlVariateEscher
    config = sequential_config(small=True)
    solver = make_solver(config)
    kwargs = dict(config)
    for key in ('cache_frozen_critic_targets', 'max_num_iterations', 'preserve_evaluation_rng',
                'evaluate_initial_policy', 'early_evaluation_node_thresholds'):
        kwargs.pop(key)
    actor = StructuredGroupedTraversalWorker('HULHE', kwargs, 17, 16)
    assert isinstance(actor._solver.regret_trainers[0].model, ProtectedSequentialNetwork)
    result = actor.collect(16, 0, 0, ParallelUnbiasedControlVariateEscher._inference_snapshot(solver), 1, snapshot_token=1)
    assert len(result['average_policy']['infostates']) > 0


def test_batch_bundle_is_offline_and_preserves_original_schedule(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_sequential_learning as launcher
    plan = launcher.build_bundle(tmp_path/'bundle', run_name='hulhe-sequential-4plus4-test')
    assert plan['checkpoint_hours'] == [4, 8] and plan['active_training_hours'] == 8
    assert plan['training_threads'] == 8 and plan['collection_workers'] == 8
    assert plan['config']['ave_policy_buffer_size'] == 4000000
    assert plan['config']['average_policy_train_steps'] == 20000
    for role, cpus, machine in [('train', 16000, 'c4-standard-16'), ('external', 4000, 'c4-standard-4')]:
        job = json.loads((tmp_path/'bundle'/f'{role}-job.json').read_text())
        assert job['taskGroups'][0]['taskSpec']['computeResource']['cpuMilli'] == cpus
        assert job['taskGroups'][0]['taskSpec']['maxRetryCount'] == 0
        assert job['allocationPolicy']['instances'][0]['policy']['machineType'] == machine


@pytest.mark.parametrize('key,value', [('structured_branch_width', 96), ('policy_group_sampling', 'uniform'),
    ('cache_frozen_regret_targets', False), ('num_hiddens', 256)])
def test_invalid_sequential_contract(key, value):
    config = sequential_config(); config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)
