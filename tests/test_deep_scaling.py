"""Residual training, inference, fair comparison provenance and Batch boundaries."""
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import pytest
import torch

from hulhe_escher.config import make_solver, validate_config
from hulhe_escher.scaling_config import scaling_config, deep_scaling_config, CHECKPOINT_HOURS
from hulhe_escher.deep_model import DeepResidualHULHEMLP, PreActivationResidualBlock
from hulhe_escher.checkpointing import save_policy_checkpoint, LoadedHULHEPolicy, sha256_file
from hulhe_escher.game import load_hulhe_game
from hulhe_escher import deep_scaling_experiment as experiment


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_deep_recipe_capacity_and_learning():
    wide, deep = scaling_config(), deep_scaling_config()
    assert {k for k in wide if wide[k] != deep[k]} == {
        'num_layers', 'num_hiddens', 'average_policy_network_layers'}
    assert set(deep)-set(wide) == {'regret_policy_architecture'}
    baseline = make_solver(scaling_config(small=True))
    solver = make_solver(deep_scaling_config(small=True))
    for trainer in [*solver.regret_trainers, solver.ave_policy_trainer]:
        assert isinstance(trainer.model, DeepResidualHULHEMLP)
        assert sum(p.numel() for p in trainer.model.parameters()) == 749827
        assert sum(isinstance(m, torch.nn.LayerNorm) for m in trainer.model.modules()) == 9
    for a, b in zip(baseline.q_value_trainer.members, solver.q_value_trainer.members):
        assert a.model.checkpoint_metadata() == b.model.checkpoint_metadata()
    assert baseline.calibration_trainer.model.checkpoint_metadata() == solver.calibration_trainer.model.checkpoint_metadata()
    before = solver.regret_trainers[0].model.trunk[1].path[2].weight.detach().clone()
    solver.iteration()
    solver.train_average_policy()
    assert not torch.equal(before, solver.regret_trainers[0].model.trunk[1].path[2].weight)
    fit = solver.ave_policy_trainer.last_fit_diagnostics
    assert fit['group_sampling'] == 'objective_mass' and fit['policy_sampling_seed'] == [0, 1]
    for trainer in [*solver.regret_trainers, solver.ave_policy_trainer]:
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    # Train/eval and batch size must not change a state's representation.
    model = solver.ave_policy_trainer.model
    x = torch.randn(4, model.input_size)
    batched = model(x).detach()
    model.eval()
    torch.testing.assert_close(batched, torch.stack([model(row) for row in x]), atol=2e-6, rtol=2e-5)
    # A fresh policy fit must reset normalization parameters as well as weights.
    with torch.no_grad():
        model.trunk[1].path[0].weight.fill_(7)
    model.reset_parameters()
    assert torch.equal(model.trunk[1].path[0].weight, torch.ones(256))
    assert torch.count_nonzero(model(x)) == 0


def test_residual_skip_preserves_signed_identity():
    block = PreActivationResidualBlock()
    with torch.no_grad():
        block.path[-1].weight.zero_()
        block.path[-1].bias.zero_()
    x = torch.randn(2, 256, requires_grad=True)
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)
    block(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


def test_policy_export_reloads_and_rejects_semantic_drift(tmp_path):
    config = deep_scaling_config(small=True)
    solver = make_solver(config)
    solver.iteration(); solver.train_average_policy()
    policy = tmp_path/'policy.pkl'
    save_policy_checkpoint(solver, policy, seed=0, config=config, checkpoint_row={})
    loaded = LoadedHULHEPolicy(load_hulhe_game(), policy)
    x = torch.randn(3, solver.infostate_size)
    torch.testing.assert_close(loaded.model(x), solver.ave_policy_trainer.model(x), rtol=0, atol=0)
    with policy.open('rb') as handle:
        payload = pickle.load(handle)
    payload['policy_model']['residual_scale'] = 1.0
    with policy.open('wb') as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match='metadata mismatch'):
        LoadedHULHEPolicy(load_hulhe_game(), policy)


@pytest.mark.parametrize('key,value', [('num_layers', 9), ('structured_branch_width', 96),
                                      ('regret_policy_architecture', 'typo')])
def test_invalid_architecture_rejected(key, value):
    config = deep_scaling_config(small=True); config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def reference_plan():
    return dict(config=scaling_config(), seed=20261115, collection_workers=8,
                training_threads=8, machine='c4-standard-16', checkpoint_hours=list(CHECKPOINT_HOURS),
                source_sha256='a'*64, uri='gs://test/wide')


def launcher():
    path = Path(__file__).resolve().parents[1]/'tools'
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location('deep_launcher', path/'launch_deep_scaling_batch.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_training_and_deferred_evaluation_bundles(tmp_path):
    wide = tmp_path/'wide.json'; wide.write_text(json.dumps(reference_plan()))
    launch = launcher()
    out = tmp_path/'training'
    plan = launch.build_bundle(out, run_name='hulhe-deep32-test', wide_plan=wide)
    assert plan['active_training_hours'] == 32 and plan['wide_source_sha256'] == 'a'*64
    assert plan['config'] == deep_scaling_config()
    job = json.loads((out/'job.json').read_text())
    assert job['taskGroups'][0]['taskSpec']['maxRunDuration'] == '201600s'
    assert job['taskGroups'][0]['taskSpec']['maxRetryCount'] == 0
    assert '--deep-uri' not in (out/'batch.sh').read_text()
    evaluation = tmp_path/'evaluation'
    plan = launch.build_bundle(evaluation, run_name='hulhe-deep32-eval-test', wide_plan=wide,
                               training_plan=out/'launch.json')
    assert plan['active_training_hours'] == 0 and plan['max_job_hours'] == 14
    assert '--deep-uri' in (evaluation/'batch.sh').read_text()
    for root in (out, evaluation):
        subprocess.run(['bash', '-n', str(root/'batch.sh')], check=True)
    changed = reference_plan(); changed['source_sha256'] = 'b'*64
    wide.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='originally selected'):
        launch.build_bundle(tmp_path/'bad', run_name='hulhe-deep32-bad', wide_plan=wide,
                            training_plan=out/'launch.json')


def test_missing_control_defers_but_auth_errors_fail(tmp_path, monkeypatch):
    def transfer(*args):
        if args[-2].endswith('/launch.json'):
            Path(args[-1]).write_text(json.dumps(reference_plan()))
        else:
            raise subprocess.CalledProcessError(1, 'gcloud', stderr='No URLs matched')
    monkeypatch.setattr(experiment, 'transfer', transfer)
    report = experiment.fetch_policies(tmp_path/'missing', 'gs://test/wide', 'a'*64)
    assert report['status'] == 'WAITING_FOR_POLICIES'
    def denied(*args):
        raise subprocess.CalledProcessError(1, 'gcloud', stderr='403 Permission denied')
    monkeypatch.setattr(experiment, 'transfer', denied)
    with pytest.raises(subprocess.CalledProcessError):
        experiment.fetch_policies(tmp_path/'denied', 'gs://test/wide', 'a'*64)


def test_policy_checksums_and_config_are_verified(tmp_path, monkeypatch):
    cloud = tmp_path/'cloud'; cloud.mkdir()
    (cloud/'launch.json').write_text(json.dumps(reference_plan()))
    solver = make_solver(scaling_config(small=True))
    # Small allocation, production metadata; fake cloud fixture only, never uploaded.
    for hour in CHECKPOINT_HOURS:
        label = f'P{hour}'
        save_policy_checkpoint(solver, cloud/f'{label}.pkl', seed=20261115,
                               config=scaling_config(), checkpoint_row={})
        record = dict(sha256=sha256_file(cloud/f'{label}.pkl'), active_seconds=hour*3600,
            outer_iteration=solver.num_iteration, metrics=dict(policy_validation=dict(
                group_sampling='objective_mass', policy_sampling_seed=[20261115, solver.num_iteration])))
        (cloud/f'{label}.json').write_text(json.dumps(record))
    monkeypatch.setattr(experiment, 'transfer', lambda *a: shutil.copyfile(cloud/Path(a[-2]).name, a[-1]))
    result = experiment.fetch_policies(tmp_path/'valid', 'gs://test/wide', 'a'*64)
    assert result['status'] == 'READY' and len(result['policies']) == 5
    with pytest.raises(ValueError, match='pinned'):
        experiment.fetch_policies(tmp_path/'source-mismatch', 'gs://test/wide', 'b'*64)
    with (cloud/'P32.pkl').open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        experiment.fetch_policies(tmp_path/'bad-hash', 'gs://test/wide', 'a'*64)


def test_match_protocol_and_audit(tmp_path, monkeypatch):
    commands = []
    def child(command, log, **kwargs):
        commands.append(command)
        destination = Path(command[command.index('--output')+1]); destination.mkdir(parents=True)
        filename = 'summary.json' if 'hulhe_escher.strategic_audit' in command else 'report.json'
        (destination/filename).write_text(json.dumps({'status':'COMPLETE'}))
        return dict(returncode=0, timed_out=False)
    from hulhe_escher import scaling_experiment
    monkeypatch.setattr(scaling_experiment, 'run_child', child)
    monkeypatch.setattr(experiment, 'run_child', child)
    report = experiment.evaluate_deep(tmp_path)
    assert report['status'] == 'COMPLETE' and report['primary'] == 'Deep-P32-vs-Wide-P32'
    matches = commands[:-1]
    assert len(matches) == 6
    assert [c[c.index('--alpha')+1] for c in matches] == ['0.05']+['0.01']*5
    assert all(c[c.index('--pilot-pairs')+1] == '10000' for c in matches)
    assert 'Wide-P32='+str(tmp_path/'wide/P32.pkl') in commands[-1]


def test_deferred_run_publishes_and_exits_without_evaluation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, 'transfer', lambda *a: calls.append('transfer'))
    monkeypatch.setattr(experiment, 'memory_gate', lambda *a, **k: calls.append('memory'))
    monkeypatch.setattr(experiment, 'train_stages', lambda *a, **k: calls.append('train'))
    monkeypatch.setattr(experiment, 'fetch_policies', lambda *a, **k: dict(status='WAITING_FOR_POLICIES'))
    monkeypatch.setattr(experiment, 'sync_analysis', lambda *a: calls.append('publish'))
    monkeypatch.setattr(experiment, 'evaluate_deep', lambda *a, **k: pytest.fail('Must defer'))
    root = tmp_path/'run'
    experiment.run(root, upload_uri='gs://test/deep', wide_uri='gs://test/wide', wide_source_sha256='a'*64)
    assert calls.count('train') == 1 and calls[-1] == 'publish'
    assert json.loads((root/'evaluation.json').read_text())['status'] == 'DEFERRED_WIDE_NOT_READY'
    with pytest.raises(ValueError, match='independent'):
        experiment.run(tmp_path/'unsafe', upload_uri='gs://test/wide', wide_uri='gs://test/wide',
                       wide_source_sha256='a'*64)


def test_traversal_actor_uses_residual_snapshot():
    from unbiased_escher.grouped_parallel_solver import StructuredGroupedTraversalWorker
    from unbiased_escher.parallel_solver import ParallelUnbiasedControlVariateEscher
    config = deep_scaling_config(small=True)
    solver = make_solver(config)
    kwargs = dict(config)
    for key in ('cache_frozen_critic_targets', 'max_num_iterations', 'preserve_evaluation_rng',
                'evaluate_initial_policy', 'early_evaluation_node_thresholds'):
        kwargs.pop(key)
    actor = StructuredGroupedTraversalWorker('HULHE', kwargs, 17, 2)
    assert isinstance(actor._solver.regret_trainers[0].model, DeepResidualHULHEMLP)
    snapshot = ParallelUnbiasedControlVariateEscher._inference_snapshot(solver)
    result = actor.collect(2, 0, 0, snapshot, 1, snapshot_token=1)
    assert len(result['average_policy']['infostates']) > 0
    for left, right in zip(solver.regret_trainers, actor._solver.regret_trainers):
        for name, tensor in left.model.state_dict().items():
            torch.testing.assert_close(tensor, right.model.state_dict()[name], rtol=0, atol=0)


def test_evaluation_only_cannot_start_training(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(experiment, 'memory_gate', lambda *a, **k: pytest.fail('No training preflight'))
    monkeypatch.setattr(experiment, 'train_stages', lambda *a, **k: pytest.fail('No training'))
    monkeypatch.setattr(experiment, 'sync_analysis', lambda *a: None)
    def fetch(root, uri, source, **kwargs):
        fetched.append((uri, kwargs.get('deep', False)))
        return dict(status='READY')
    monkeypatch.setattr(experiment, 'fetch_policies', fetch)
    monkeypatch.setattr(experiment, 'evaluate_deep', lambda *a, **k: dict(status='COMPLETE'))
    experiment.run(tmp_path/'eval', upload_uri='gs://test/eval',
                   wide_uri='gs://test/wide', wide_source_sha256='a'*64,
                   deep_uri='gs://test/deep', deep_source_sha256='b'*64)
    assert fetched == [('gs://test/deep', True), ('gs://test/wide', False)]
