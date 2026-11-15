from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from hulhe_escher.config import baseline_config, smoke_config, make_solver
from hulhe_escher.checkpoint_schedule import CheckpointSchedule
from unbiased_escher.holdem_fitting import policy_validation_mask, weighted_policy_ce
from test_hulhe import state_at_street, assert_models_equal, model_states


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_production_spec_and_limit_capacity_bound():
    c = baseline_config()
    assert (c['num_traversals'], c['advantage_network_train_steps'],
            c['baseline_network_train_steps'], c['calibration_train_steps']) == (10000,750,2500,2000)
    assert c['advantage_buffer_size'] == 250000
    assert c['advantage_buffer_size'] >= c['num_traversals'] * 23
    assert c['ave_policy_buffer_size'] == 2000000
    assert c['cache_frozen_regret_targets'] and c['uniform_regret_ties']


@pytest.mark.parametrize('values,legal,expected', [
    ([0,0,0], [0,1,2], [1/3,1/3,1/3]),
    ([-1,-1,-2], [0,1,2], [.5,.5,0]),
    ([-3,-1,-2], [0,1,2], [0,1,0]),
    ([10,-1,-1], [1,2], [0,.5,.5]),
    ([1,2,-1], [0,1,2], [1/3,2/3,0]),
])
def test_uniform_best_ties_only(values, legal, expected):
    trainer = make_solver(smoke_config()).regret_trainers[0]
    np.testing.assert_allclose(trainer.regret_matching(np.array(values,dtype=float),legal),expected)


def test_critic_batched_strategies_match_collection(monkeypatch):
    solver = make_solver(smoke_config())
    states = torch.zeros(5, solver.infostate_size)
    states[:,:3] = torch.tensor([[0.,0,0],[-1,-1,-3],[-3,-1,-2],[10,-1,-1],[1,2,-1]])
    masks = torch.tensor([[1,1,1],[1,1,1],[1,1,1],[0,1,1],[1,1,1]])
    players = torch.tensor([0,1,0,1,0])
    for trainer in solver.regret_trainers:
        monkeypatch.setattr(trainer, 'predict', lambda model,x,mask: x[:,:3]*mask)
    member = solver.q_value_trainer.members[0]
    for flag in (True,False):
        for trainer in solver.regret_trainers: trainer.use_regret_matching_argmax = flag
        actual = member._batched_predictive_strategies(states,masks,players,2)
        for i,p in enumerate(players):
            expected = solver.regret_trainers[p].regret_matching(states[i,:3].numpy(),np.flatnonzero(masks[i]))
            np.testing.assert_allclose(actual[i].numpy(),expected)
    assert not member._batched_predictive_strategies(states,torch.zeros_like(masks),players,2).any()


@pytest.mark.parametrize('rows', [5,19,270])
def test_cached_regret_fit_matches_uncached_and_rebuilds(rows):
    results = []
    for cached in (False,True):
        c=smoke_config(); c['cache_frozen_regret_targets']=cached
        solver=make_solver(c,seed=33)
        trainer=solver.regret_trainers[0]
        rng=np.random.default_rng(21)
        with torch.no_grad():
            for p in trainer.model.parameters(): p.uniform_(-.03,.03)
            trainer.target_model.load_state_dict(trainer.model.state_dict())
        for i in range(rows):
            trainer.buffer.add(rng.random(trainer.input_size,dtype=np.float32),
                rng.normal(size=3).astype(np.float32), [1,1,i%2], 2)
        trainer.train_model(2)
        first=deepcopy(trainer.model.state_dict())
        # New labels and a new target/iteration must invalidate the previous cache.
        trainer.buffer.q_value_buf[:min(rows,256)] *= -2
        trainer.train_model(3)
        results.append((first,deepcopy(trainer.model.state_dict()),
                        deepcopy(trainer.optimizer.state_dict()),deepcopy(trainer.buffer.rng.bit_generator.state)))
    assert_models_equal(results[0][:2],results[1][:2],exact=False)
    assert results[0][3] == results[1][3]
    for key, state in results[0][2]['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(value,results[1][2]['state'][key][field],atol=1e-6,rtol=1e-5)


def test_policy_split_is_stable_and_groups_duplicates():
    trainer=make_solver(smoke_config()).ave_policy_trainer
    rng=np.random.default_rng(8)
    for i in range(100):
        feature=rng.random(trainer.input_size,dtype=np.float32)
        for iteration in (1,2): trainer.buffer.add(feature,[.2,.3,.5],[1,1,1],iteration)
    features,targets,masks,weights=trainer._grouped_training_data(2)
    assert len(features)==100
    split=policy_validation_mask(features,.2)
    assert 0<sum(split)<100
    perm=rng.permutation(100)
    np.testing.assert_array_equal(policy_validation_mask(features[perm],.2),split[perm])
    # Later fits retain the same membership for existing information sets.
    more=torch.cat([features,torch.ones(1,trainer.input_size)])
    np.testing.assert_array_equal(policy_validation_mask(more,.2)[:100],split)


def test_policy_keeps_best_validation_weights_and_optimizer(monkeypatch):
    import unbiased_escher.holdem_fitting as fitting
    trainer=make_solver(smoke_config()).ave_policy_trainer
    rng=np.random.default_rng(10)
    for _ in range(100):
        trainer.buffer.add(rng.random(trainer.input_size,dtype=np.float32),[.2,.3,.5],[1,1,1],1)
    trainer.validation_fraction=.2
    snapshots=[]
    def score(model,*args):
        snapshots.append((deepcopy(model.state_dict()),deepcopy(trainer.optimizer.state_dict())))
        return len(snapshots) # First fit is deliberately the best.
    monkeypatch.setattr(fitting,'weighted_policy_ce',score)
    trainer.train_model(1)
    assert trainer.last_fit_diagnostics['selected_update']==1
    assert trainer.last_fit_diagnostics['updates_completed']==2
    assert_models_equal([trainer.model.state_dict()],[snapshots[0][0]])
    for state in trainer.optimizer.state_dict()['state'].values(): assert state['step']==1


def test_validation_ce_uses_iteration_masses():
    model=torch.nn.Identity()
    features=torch.tensor([[2.,0.],[0.,1.],[0.,0.]])
    targets=torch.tensor([[0.,1.],[1.,0.],[.5,.5]])
    weights=torch.tensor([1.,9.,2.]); masks=torch.ones_like(features)
    actual=weighted_policy_ce(model,features,targets,masks,weights,np.arange(3),2)
    expected=(-(targets*torch.log_softmax(features,-1)).sum(-1)*weights).sum()/weights.sum()
    assert actual==pytest.approx(expected.item())


def test_cli_recovery_without_policy_and_rng_isolation(tmp_path, monkeypatch):
    from hulhe_escher import train
    original=train.CheckpointSchedule.due
    def forced(self,**kwargs):
        if not kwargs['final']: return True,False
        return original(self,**kwargs)
    monkeypatch.setattr(train.CheckpointSchedule,'due',forced)
    saved = {}
    original_save = train.save_training_state
    def capture_first(path, payload):
        if path.name == 'state_000001':
            saved.update(deepcopy(payload))
        return original_save(path, payload)
    monkeypatch.setattr(train, 'save_training_state', capture_first)
    train.main(['--smoke','--iterations','2','--output',str(tmp_path)])
    assert not (tmp_path/'state_000001').exists()
    assert not (tmp_path/'policy_000001.pkl').exists()
    assert (tmp_path/'policy_000002.pkl').is_file()
    from hulhe_escher.training_state import read_training_state,restore_training_state
    config=smoke_config()
    payload=saved
    assert payload['checkpoint_schedule']['last_policy_seconds']==0
    kwargs=dict(seed=0,repository_commit=train.source_fingerprint(),config=config)
    restored=make_solver(config)
    restore_training_state(restored,payload,**kwargs)
    restored.iteration()
    expected=read_training_state(tmp_path/'state_000002')
    other=make_solver(config)
    restore_training_state(other,expected,**kwargs)
    rng = other._capture_rng_state()
    # The policy fit at iteration 2 must not alter the self-play random streams.
    restored.iteration(); a=model_states(restored)[:4]
    other._restore_rng_state(rng)
    other.iteration(); b=model_states(other)[:4]
    assert_models_equal(a,b)
