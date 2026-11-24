"""Hybrid learning, both gradient paths, immutable exports and cloud boundaries."""
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

from hulhe_escher.config import make_solver, validate_config
from hulhe_escher.scaling_config import hybrid_scaling_config, deep_scaling_config, CHECKPOINT_HOURS
from hulhe_escher.hybrid_model import HybridResidualHULHEMLP, WideResidualBlock, HYBRID_LAYERS
from hulhe_escher.checkpointing import save_policy_checkpoint, LoadedHULHEPolicy, sha256_file
from hulhe_escher.game import load_hulhe_game
from hulhe_escher import hybrid_scaling_experiment as experiment, hybrid_evaluate as evaluation
from test_scaling_corrected import pinned
from test_policy_sampling_experiment import frozen_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_hybrid_learning_capacity_reset_and_unchanged_critics():
    config = hybrid_scaling_config(small=True)
    reference = deep_scaling_config(small=True)
    assert {k for k in config if config[k] != reference[k]} == {
        'num_layers', 'average_policy_network_layers', 'regret_policy_architecture'}
    solver, baseline = make_solver(config), make_solver(reference)
    for trainer in [*solver.regret_trainers, solver.ave_policy_trainer]:
        assert isinstance(trainer.model, HybridResidualHULHEMLP)
        assert sum(p.numel() for p in trainer.model.parameters()) == 1245699
        assert trainer.model.hidden_layers == HYBRID_LAYERS
    for a, b in zip(solver.q_value_trainer.members, baseline.q_value_trainer.members):
        assert a.model.checkpoint_metadata() == b.model.checkpoint_metadata()
    assert solver.calibration_trainer.model.checkpoint_metadata() == baseline.calibration_trainer.model.checkpoint_metadata()
    before = solver.regret_trainers[0].model.wide[1].path[2].weight.detach().clone()
    solver.iteration(); solver.train_average_policy()
    assert not torch.equal(before, solver.regret_trainers[0].model.wide[1].path[2].weight)
    fit = solver.ave_policy_trainer.last_fit_diagnostics
    assert fit['group_sampling'] == 'objective_mass' and fit['policy_sampling_seed'] == [0, 1]
    model = solver.ave_policy_trainer.model
    x = torch.randn(3, model.input_size)
    expected = model(x).detach()
    model.eval()
    torch.testing.assert_close(expected, torch.stack([model(row) for row in x]), atol=2e-6, rtol=2e-5)
    with torch.no_grad():
        model.wide[1].path[0].weight.fill_(7)
    model.reset_parameters()
    assert torch.equal(model.wide[1].path[0].weight, torch.ones(512))
    assert torch.count_nonzero(model(x)) == 0


def test_identity_skip_and_direct_output_gradient():
    block = WideResidualBlock()
    with torch.no_grad():
        block.path[-1].weight.zero_(); block.path[-1].bias.zero_()
    x = torch.randn(2, 512, requires_grad=True)
    torch.testing.assert_close(block(x), x, atol=0, rtol=0)
    block(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)
    model = make_solver(hybrid_scaling_config(small=True)).ave_policy_trainer.model
    wide, deep = torch.randn(2, 512, requires_grad=True), torch.randn(2, 256, requires_grad=True)
    for first, last in ((0, 512), (512, 768)):
        with torch.no_grad():
            model.output.weight.zero_(); model.output.weight[:, first:last].fill_(1)
        wide.grad = deep.grad = None
        model.output(torch.cat([wide, deep], dim=-1)).sum().backward()
        assert torch.count_nonzero(wide.grad) == (1024 if first == 0 else 0)
        assert torch.count_nonzero(deep.grad) == (512 if first == 512 else 0)


def save_hybrid(path):
    config = hybrid_scaling_config(small=True)
    solver = make_solver(config)
    solver.iteration(); solver.train_average_policy()
    save_policy_checkpoint(solver, path, seed=0, config=config, checkpoint_row={})
    return solver


def test_export_reload_semantics_and_legal_probabilities(tmp_path):
    path = tmp_path/'policy.pkl'
    solver = save_hybrid(path)
    loaded = LoadedHULHEPolicy(load_hulhe_game(), path)
    x = torch.randn(3, solver.infostate_size)
    torch.testing.assert_close(loaded.model(x), solver.ave_policy_trainer.model(x), rtol=0, atol=0)
    state = load_hulhe_game().new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    probabilities = loaded.action_probabilities(state)
    assert set(probabilities) == set(state.legal_actions())
    assert sum(probabilities.values()) == pytest.approx(1.)
    with path.open('rb') as f:
        payload = pickle.load(f)
    payload['policy_model']['output_features'].reverse()
    with path.open('wb') as f:
        pickle.dump(payload, f)
    with pytest.raises(ValueError, match='metadata mismatch'):
        LoadedHULHEPolicy(load_hulhe_game(), path)


@pytest.mark.parametrize('key,value', [('num_layers', 10), ('num_hiddens', 512),
    ('structured_branch_width', 96), ('average_policy_network_layers', [256]*13)])
def test_invalid_hybrid_descriptor(key, value):
    config = hybrid_scaling_config(small=True); config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_worker_snapshot_uses_hybrid():
    from unbiased_escher.grouped_parallel_solver import StructuredGroupedTraversalWorker
    from unbiased_escher.parallel_solver import ParallelUnbiasedControlVariateEscher
    config = hybrid_scaling_config(small=True)
    solver = make_solver(config)
    kwargs = dict(config)
    for key in ('cache_frozen_critic_targets', 'max_num_iterations', 'preserve_evaluation_rng',
                'evaluate_initial_policy', 'early_evaluation_node_thresholds'):
        kwargs.pop(key)
    actor = StructuredGroupedTraversalWorker('HULHE', kwargs, 17, 16)
    assert isinstance(actor._solver.regret_trainers[0].model, HybridResidualHULHEMLP)
    snapshot = ParallelUnbiasedControlVariateEscher._inference_snapshot(solver)
    # A two-trajectory fixture can terminate before recording an opponent turn.
    result = actor.collect(16, 0, 0, snapshot, 1, snapshot_token=1)
    assert len(result['average_policy']['infostates']) > 0
    for a, b in zip(solver.regret_trainers, actor._solver.regret_trainers):
        for key, tensor in a.model.state_dict().items():
            torch.testing.assert_close(tensor, b.model.state_dict()[key], rtol=0, atol=0)


def rows(values):
    return [{name: dict(mean_bb=v+i) for i, (name, *_) in enumerate(evaluation.MATCHUPS)} for v in values]


def test_grouped_refits_multiplicity_and_runtime_planning():
    data = rows([1., 2., 3., 4.])
    obs = evaluation.observations(data)
    np.testing.assert_array_equal(obs['hybrid_vs_corrected'], [2.5, 3.5, 4.5, 5.5])
    plan = evaluation.plan_from_pilot(data, 10000., 10000.)
    assert plan['selected'] == max(p['selected'] for p in plan['tests'].values())
    assert [p['alpha'] for p in plan['tests'].values()] == [.05, .025, .025]
    assert plan['pilot_excluded'] and plan['sample_not_reduced_for_runtime']
    slow = evaluation.plan_from_pilot(data, 1., 1.)
    assert slow['selected'] == plan['selected'] and not slow['full_selected_sample_fits']


def test_real_spawned_mixed_architecture_evaluation(pinned, tmp_path):
    policy = tmp_path/'hybrid.pkl'; save_hybrid(policy)
    before = {p: sha256_file(p) for p in pinned.iterdir()}
    report = evaluation.evaluate(policy, pinned, tmp_path/'eval', workers=2,
        max_seconds=120, smoke=True, opponent='check_call')
    assert report['status'] == 'COMPLETE' and report['completed_hands'] == 64
    assert report['completed_groups'] == 8
    assert [r['confidence'] for r in report['results'].values()] == [.95, .975, .975]
    assert all(not r['confirmatory'] for r in report['results'].values())
    assert all(sha256_file(p) == digest for p, digest in before.items())


def test_training_and_evaluation_only_batch_plans(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_hybrid_scaling_batch as launcher
    root = tmp_path/'train'
    plan = launcher.build_bundle(root, run_name='hulhe-hybrid32-test')
    assert plan['active_training_hours'] == 32 and plan['checkpoint_hours'] == list(CHECKPOINT_HOURS)
    assert plan['per_network_parameters'] == 1245699 and plan['fell_omen_2']
    assert plan['config'] == hybrid_scaling_config()
    job = json.loads((root/'job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRetryCount'] == 0 and task['maxRunDuration'] == '201600s'
    assert job['allocationPolicy']['instances'][0]['policy']['bootDisk']['sizeGb'] == 200
    assert '--training-uri' not in (root/'batch.sh').read_text()
    other = tmp_path/'eval'
    plan = launcher.build_bundle(other, run_name='hulhe-hybrid32-eval', training_plan=root/'launch.json')
    assert plan['evaluation_only'] and plan['active_training_hours'] == 0 and plan['max_job_hours'] == 14
    assert '--training-uri' in (other/'batch.sh').read_text()
    for folder in (root, other):
        subprocess.run(['bash', '-n', str(folder/'batch.sh')], check=True)
    with pytest.raises(ValueError, match='independent'):
        launcher.build_bundle(tmp_path/'bad', run_name='hulhe-hybrid32-test', training_plan=root/'launch.json')
    with pytest.raises(ValueError, match='original'):
        launcher.build_bundle(tmp_path/'bad2', run_name='hulhe-hybrid32-other', training_plan=other/'launch.json')


def test_evaluation_only_never_trains_and_rejects_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'prepare_inputs', lambda *a: None)
    monkeypatch.setattr(experiment, 'memory_gate', lambda *a, **k: pytest.fail('No preflight'))
    monkeypatch.setattr(experiment, 'train_stages', lambda *a, **k: pytest.fail('No training'))
    monkeypatch.setattr(experiment, 'sync_analysis', lambda *a: None)
    calls = []
    def fetch(*a, **k):
        calls.append(k['expected_config']); return dict(status='READY')
    monkeypatch.setattr(experiment, 'fetch_policies', fetch)
    monkeypatch.setattr(experiment, 'evaluate_hybrid', lambda *a, **k: dict(status='COMPLETE'))
    experiment.run(tmp_path/'eval', upload_uri='gs://test/eval',
                   training_uri='gs://test/train', training_source_sha256='a'*64)
    assert calls == [hybrid_scaling_config()]
    with pytest.raises(ValueError, match='independent'):
        experiment.run(tmp_path/'bad', upload_uri='gs://test/train',
                       training_uri='gs://test/train', training_source_sha256='a'*64)


def test_all_evaluation_stages_and_incomplete_propagation(tmp_path, monkeypatch):
    commands = []
    def child(command, log, **kwargs):
        commands.append(command)
        folder = Path(command[command.index('--output')+1]); folder.mkdir(parents=True)
        name = 'summary.json' if 'hulhe_escher.strategic_audit' in command else 'report.json'
        (folder/name).write_text(json.dumps(dict(status='COMPLETE')))
        return dict(returncode=0, timed_out=False)
    monkeypatch.setattr(experiment, 'run_child', child)
    calls = []
    def progress(*a, **kw):
        calls.append(kw); return dict(status='INCOMPLETE')
    monkeypatch.setattr(experiment, 'evaluate', progress)
    report = experiment.evaluate_hybrid(tmp_path)
    assert report['status'] == 'INCOMPLETE'
    assert len(commands) == 2 and len(calls[0]['matches']) == 5
    assert calls[0]['seed_base'] == 743001
    assert all(row[3] == .01 for row in calls[0]['matches'])
    assert 'Hybrid-P32='+str(tmp_path/'P32.pkl') in commands[-1]
    assert 'Deep-P32='+str(tmp_path/'inputs/Deep-P32.pkl') in commands[-1]


class FakePool:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


@pytest.mark.parametrize('mode', ['complete', 'partial', 'error', 'pilot', 'budget'])
def test_pilot_excluded_and_partial_results_never_confirmatory(pinned, tmp_path, monkeypatch, mode):
    policy = tmp_path/'hybrid.pkl'; save_hybrid(policy)
    # Use production inference semantics with tiny, entirely simulated observations.
    with policy.open('rb') as f:
        payload = pickle.load(f)
    payload.update(training_config=hybrid_scaling_config(), seed=20261115,
                   training_elapsed_seconds=32*3600)
    with policy.open('wb') as f:
        pickle.dump(payload, f)
    monkeypatch.setattr(evaluation, 'verify_build', lambda _: {})
    monkeypatch.setattr(evaluation.mp, 'get_context', lambda _: type('Context', (), {'Pool': FakePool}))
    monkeypatch.setattr(evaluation, 'plan_from_pilot', lambda *a, **k:
                        dict(selected=4, full_selected_sample_fits=mode != 'budget'))
    seeds = []
    def collect(pool, *, seed, count, rows=None, **kw):
        seeds.append(seed)
        if rows is None:
            return globals()['rows']([999.]*(1 if mode == 'pilot' else count))
        rows.extend(globals()['rows']([1., 2., 3., 4.][:2 if mode in ('partial', 'error') else 4]))
        if mode == 'error':
            raise RuntimeError('worker failed')
        return rows
    monkeypatch.setattr(evaluation, 'collect', collect)
    report = evaluation.evaluate(policy, pinned, tmp_path/'eval', assets=tmp_path)
    if mode in ('pilot', 'budget'):
        assert not report['results'] and len(seeds) == 1
    else:
        assert seeds == [evaluation.PILOT_SEED, evaluation.PILOT_SEED+1]
        result = report['results']['hybrid_vs_deep']
        assert result['bb_per_100'] == (250. if mode == 'complete' else 150.)
        assert result['confirmatory'] == (mode == 'complete')


def test_inputs_and_fo2_fail_before_training(pinned, tmp_path, monkeypatch):
    import shutil
    from hulhe_escher import scaling_corrected_comparison
    monkeypatch.setattr(experiment, 'SOURCES', scaling_corrected_comparison.SOURCES)
    monkeypatch.setattr(experiment, 'transfer', lambda *a: shutil.copyfile(pinned/Path(a[-2]).name, a[-1]))
    monkeypatch.setattr(experiment, 'run_child', lambda *a, **k: dict(returncode=1, timed_out=False))
    root = tmp_path/'setup'; root.mkdir()
    with pytest.raises(RuntimeError, match='before training'):
        experiment.prepare_inputs(root)
    monkeypatch.setattr(experiment, 'prepare_inputs', lambda *a: (_ for _ in ()).throw(ValueError('bad inputs')))
    monkeypatch.setattr(experiment, 'sync_analysis', lambda *a: None)
    monkeypatch.setattr(experiment, 'train_stages', lambda *a, **k: pytest.fail('Do not train'))
    with pytest.raises(ValueError, match='bad inputs'):
        experiment.run(tmp_path/'run', upload_uri='gs://test/new')
    assert json.loads((tmp_path/'run/failure.json').read_text())['error_type'] == 'ValueError'
