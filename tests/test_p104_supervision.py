"""Frozen estimator parity, independent references and diagnostic/Batch contracts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from hulhe_escher.config import smoke_config, make_solver
from hulhe_escher.game import serialisable_game_definition
from hulhe_escher.checkpointing import save_policy_checkpoint, sha256_file
from hulhe_escher.state_storage import save_sharded_training_state, STATE_TYPE
from hulhe_escher.supervision_runtime import FrozenRuntime, load_runtime, conditional_ranges, exact_river_q
from hulhe_escher.supervision_positions import encounters, select_positions, analyse_position, references
from hulhe_escher.supervision_statistics import decomposition, measurements, summarise_position, select, aggregate
from hulhe_escher.position_investigation.common import visible_position
from hulhe_escher.diagnostic_probes import positions
from hulhe_escher.strategic_audit.library import digest


@pytest.fixture
def frozen(tmp_path):
    torch.set_num_threads(1)
    c = smoke_config()
    c.update(num_hiddens=16,num_layers=1,structured_branch_width=4,critic_branch_width=4,
        critic_network_layers=[16],calibration_network_layers=[8],calibration_branch_width=4,
        average_policy_network_layers=[16])
    solver = make_solver(c,seed=19,workers=1)
    solver.num_iteration = 7
    # Uniform regret policies provide full support and avoid fixture-specific unreachable histories.
    for trainer in solver.regret_trainers:
        for model in (trainer.model,trainer.target_model):
            for p in model.parameters():
                p.data.zero_()
    def models(trainer):
        return dict(model=deepcopy(trainer.model.state_dict()),target_model=deepcopy(trainer.target_model.state_dict()),
                    target_version=0)
    metadata = dict(type=STATE_TYPE, schema_version=1, config=c, game=serialisable_game_definition(),
        encoder=solver.feature_encoder.metadata(),repository_commit='test',solver=dict(num_iteration=7),
        regret_trainers=[models(t) for t in solver.regret_trainers],
        q_ensemble=dict(members=[models(t) for t in solver.q_value_trainer.members]),
        calibration=models(solver.calibration_trainer),
        average_policy_trainer=dict(model=deepcopy(solver.ave_policy_trainer.model.state_dict())))
    state,policy = tmp_path/'state',tmp_path/'policy.pkl'
    save_sharded_training_state(state,metadata)
    save_policy_checkpoint(solver,policy,seed=19,config=c,checkpoint_row={})
    runtime = load_runtime(state,policy,smoke=True)
    return runtime,solver,state,policy


def test_restore_native_batched_inference_and_read_only(frozen):
    runtime,solver,state,policy = frozen
    hashes = {p:sha256_file(p) for p in state.rglob('*') if p.is_file()}
    s = positions()[0]['state']
    for fold in (0,1):
        solver.q_value_trainer.active_fold = fold
        np.testing.assert_allclose(runtime.controls(s,s.current_player(),fold)[0],
            solver.q_value_trainer.get_baseline_and_disagreement(s,s.current_player())[0])
        np.testing.assert_allclose(runtime.proposal_batch([s,s],s.current_player(),fold),
            np.tile(runtime.controls(s,s.current_player(),fold)[2],(2,1)),rtol=1e-6)
    np.testing.assert_allclose(runtime.policy_batch([s,s]),np.tile(runtime.policy(s),(2,1)))
    with pytest.raises(RuntimeError,match='forbidden'):
        runtime.solver.regret_trainers[0].train_model()
    with pytest.raises(ValueError,match='pinned'):
        load_runtime(state,policy)
    assert all(sha256_file(p)==h for p,h in hashes.items())


@pytest.mark.parametrize('fold',[0,1])
def test_recursive_current_estimator_matches_production(frozen,monkeypatch,fold):
    runtime,solver,_,_ = frozen
    # Exercise nonuniform policies/proposals and nonzero, fold-specific controls,
    # not just the zero-initialised networks used by the range fixtures.
    generator = torch.Generator().manual_seed(714)
    pairs = [(a.model,b.model) for a,b in zip(runtime.solver.regret_trainers,solver.regret_trainers)]
    pairs += [(a.target_model,b.target_model) for a,b in zip(
        runtime.solver.q_value_trainer.members,solver.q_value_trainer.members)]
    pairs += [(runtime.solver.calibration_trainer.target_model,solver.calibration_trainer.target_model)]
    with torch.no_grad():
        for index,(a,b) in enumerate(pairs):
            for p in a.parameters():
                p.copy_(torch.randn(p.shape,generator=generator)*.03)
            if index < 2:
                list(a.parameters())[-1].copy_(torch.tensor([.1,.3,.6]))
            b.load_state_dict(a.state_dict())
    solver.num_iteration = runtime.solver.num_iteration
    solver.q_value_trainer.active_fold = fold
    for t in [*solver.regret_trainers,solver.ave_policy_trainer,solver.q_value_trainer]:
        monkeypatch.setattr(t,'add_data',lambda *a,**kw:None)
    monkeypatch.setattr(solver.calibration_trainer,'add',lambda *a,**kw:None)
    import unbiased_escher.solver as native
    original = native.control_variate_advantage
    captured = []
    def capture(*a,**kw):
        result = original(*a,**kw)
        captured.append(result)
        return result
    monkeypatch.setattr(native,'control_variate_advantage',capture)
    for state in [p['state'] for p in positions()]:
        np.testing.assert_allclose(runtime.policy_batch([state])[0],runtime.policy(state),rtol=1e-6)
        np.testing.assert_allclose(runtime.proposal_batch([state],state.current_player(),fold)[0],
            runtime.controls(state,state.current_player(),fold)[2],rtol=1e-6)
        for seed in range(4):
            hero = state.current_player()
            np.random.seed(seed)
            expected = solver.dfs(state.clone(),hero)
            actual,draw = runtime.estimate(state.clone(),hero,fold,np.random.RandomState(seed))
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8)
            np.testing.assert_allclose(draw['advantage'],captured[-1].advantages,rtol=1e-6,atol=1e-8)


def test_exact_river_and_all_action_reference(frozen,monkeypatch):
    runtime,_,_,_ = frozen
    state = positions()[0]['state']  # Capped river, fold/call both terminal.
    position = visible_position(state)
    hands,roots,weights,report = conditional_ranges(runtime,position)
    assert len(hands)==990 and np.allclose(weights.sum(axis=1),1)
    assert max(report['production_total_variation']) < 1e-6
    exact = exact_river_q(runtime,roots,position['hero'])
    for action in state.legal_actions():
        np.testing.assert_allclose(exact[:,action],[s.child(action).returns()[position['hero']]/runtime.solver.max_utility for s in roots])
    def prohibited(*a,**kw):
        raise AssertionError('Reference queried control network')
    monkeypatch.setattr(runtime,'controls',prohibited)
    ref,_,_ = references(runtime,roots,weights,position,samples=8)
    assert ref['exact'] and np.allclose(ref['q'],weights@exact)
    _,draw = runtime.estimate(state.clone(),position['hero'],0,np.random.default_rng(5),arm='all_actions')
    expected = [state.child(a).returns()[position['hero']]/runtime.solver.max_utility if a in state.legal_actions() else 0 for a in range(3)]
    np.testing.assert_allclose(draw['q'],expected)
    assert runtime.direct(state.child(1),position['hero'],np.random.default_rng(3))==expected[1]


def test_direct_continuations_never_query_critic(frozen,monkeypatch):
    runtime,_,_,_ = frozen
    def prohibited(*a,**kw):
        raise AssertionError('Direct reference queried critic/calibration')
    monkeypatch.setattr(runtime,'controls',prohibited)
    state = positions()[-1]['state']
    a = runtime.direct(state.clone(),state.current_player(),np.random.default_rng(42))
    b = runtime.direct(state.clone(),state.current_player(),np.random.default_rng(42))
    assert a==b and np.isfinite(a)


def test_range_audit_includes_hidden_dependent_own_proposals(frozen,monkeypatch):
    runtime,_,_,_ = frozen
    def proposal(states,hero,fold):
        masks = np.array([s.legal_actions_mask() for s in states],float)
        # Deliberate full-state-dependent proposal: self reach no longer cancels.
        masks[:,1] *= np.linspace(.05,1.,len(states))
        return masks/masks.sum(axis=1,keepdims=True)
    monkeypatch.setattr(runtime,'proposal_batch',proposal)
    _,_,weights,report = conditional_ranges(runtime,visible_position(positions()[0]['state']))
    assert min(report['production_total_variation']) > .01
    np.testing.assert_allclose(weights.sum(axis=1),1)
    assert np.ptp(weights[0]) < 1e-12 and np.ptp(weights[1]) > 0


def test_real_nested_sampling_stats_and_average_four(frozen):
    runtime,_,_,_ = frozen
    position = dict(visible_position(positions()[0]['state']),id='river-test',priority=digest('test'),
        partition='validation',component='a',encounter_weight=1.,family='river')
    report,arrays = analyse_position(runtime,position,hidden=8,repeats=2,samples=8)
    for f in (0,1):
        a,b = arrays[f'fold{f}_current'],arrays[f'fold{f}_average4']
        assert a.shape==(8,2,3) and b.shape==(2,2,3)
        np.testing.assert_allclose(a.mean((0,1)),b.mean((0,1)))
        costs = report['folds'][f]['cpu_seconds_per_target']
        assert costs['average4']==4*costs['current']
    stats,replicas = summarise_position(report,arrays,bootstrap=20)
    for fold in stats['folds']:
        np.testing.assert_array_equal(fold['arms']['current']['equal_cpu_mse'],
                                      fold['arms']['average4']['equal_cpu_mse'])
    np.testing.assert_array_equal(replicas['current'],replicas['average4'])
    report['statistics'] = stats
    decision = select([report])
    assert not decision['assessment_used']
    summary = aggregate([report],{report['id']:replicas},decision['selected'],bootstrap=40)
    assert not summary['evidence_for_short_continuation']  # One board is insufficient.
    assert report['beta_one_alias']
    report['position']['partition']='assessment'
    with pytest.raises(ValueError,match='validation'):
        select([report])


def test_nested_variance_not_pseudoreplicated():
    data = np.repeat(np.arange(8)[:,None,None],4,axis=1)*np.ones((8,4,3))
    d = decomposition(data)
    assert np.all(d['within']==0)
    np.testing.assert_allclose(d['mean_variance'],np.var(np.arange(8),ddof=1)/8)
    m = measurements(data,np.ones(3)*3.5,np.zeros((3,3)),1.,256.)
    np.testing.assert_allclose(m['equal_cpu_mse'],d['variance']/256)


def test_real_spawned_smoke_pipeline_preserves_source(frozen,tmp_path):
    _,_,state,policy = frozen
    before = {p:sha256_file(p) for p in (*state.rglob('*'),policy) if p.is_file()}
    output = tmp_path/'audit'
    subprocess.run([sys.executable,'-m','hulhe_escher.p104_supervision','--state',str(state),
        '--policy',str(policy),'--output',str(output),'--workers','2','--smoke',
        '--max-seconds','180'],check=True,timeout=200,capture_output=True,text=True)
    report = json.loads((output/'summary.json').read_text())
    assert report['status']=='COMPLETE' and report['source_preserved']
    assert len(report['positions'])==18 and not report['assessment']['evidence_for_short_continuation']
    assert json.loads((output/'selection.json').read_text())==report['selection']
    assessment_files = [output/'positions'/f"{r['id']}.json" for r in report['positions']
                        if r['position']['partition']=='assessment']
    assert all(p.stat().st_mtime >= (output/'selection.json').stat().st_mtime for p in assessment_files)
    assert all(sha256_file(p)==h for p,h in before.items())
    assert all('NaN' not in p.read_text() for p in output.rglob('*.json'))


def test_encounters_split_is_outcome_blind_deterministic(frozen):
    runtime,_,_,_ = frozen
    rows = encounters(runtime,0,80)
    assert rows and all(len(r['board'])>=3 for r in rows)
    a,b = select_positions(rows,smoke=True),select_positions(list(reversed(rows)),smoke=True)
    assert a['positions']==b['positions']
    partitions = {}
    for p in a['positions']:
        partitions.setdefault(p['component'],set()).add(p['partition'])
    assert all(len(v)==1 for v in partitions.values())


def test_launcher_limits_archive_and_no_automatic_submission(tmp_path,monkeypatch,capsys):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_p104_supervision as launcher
    calls = []
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:calls.append(a))
    launcher.main(['--run-name','hulhe-p104-supervision-test','--output',str(tmp_path/'bundle')])
    assert not calls
    plan = json.loads((tmp_path/'bundle/launch.json').read_text())
    job = json.loads((tmp_path/'bundle/job.json').read_text())
    assert plan['positions']==96 and plan['metadata_only_download'] and plan['training_iterations']==0
    assert job['taskGroups'][0]['taskSpec']['maxRunDuration']=='28800s'
    assert job['taskGroups'][0]['taskSpec']['maxRetryCount']==0
    assert plan['max_task_hours']==8
    capsys.readouterr()


def test_cloud_failure_preserves_status_and_uses_metadata_only(tmp_path,monkeypatch):
    from hulhe_escher import p104_supervision_cloud as cloud
    calls = []
    def prepare(uri,state,**kw):
        calls.append(('prepare',kw))
        raise ValueError('fixture source rejection')
    monkeypatch.setattr(cloud,'prepare',prepare)
    monkeypatch.setattr(cloud,'storage',lambda *a:calls.append(a))
    root = tmp_path/'cloud'
    with pytest.raises(ValueError,match='fixture'):
        cloud.run_cloud(root,'gs://test-bucket/hulhe-supervision-test')
    assert ('prepare',{'metadata_only':True}) in calls
    status = json.loads((root/'results/cloud_status.json').read_text())
    assert status['status']=='FAILED'
    assert any(call[0]=='rsync' for call in calls)
    assert calls[-1][0]=='cp' and str(calls[-1][1]).endswith('cloud_status.json')
    with pytest.raises(ValueError,match='overlap'):
        cloud.run_cloud(tmp_path/'bad',cloud.SOURCE_URI)
    assert not (tmp_path/'bad').exists()


def test_pilot_forecast_and_runtime_guard():
    from hulhe_escher.p104_supervision import forecast
    rows = [dict(position=dict(board=list(range(s))),stage_cpu_seconds=dict(ranges=1.,reference=2.,sampling=3.))
            for s in (3,4,5)]
    result = forecast(rows,[r['position'] for r in rows],workers=2,hidden=32,repeats=8,samples=4096)
    assert result['estimated_cpu_seconds']==549.
    assert result['conservative_remaining_wall_seconds']==711.75
    with pytest.raises(ValueError,match='every street'):
        forecast(rows[:2],[],workers=2,hidden=32,repeats=8,samples=4096)
