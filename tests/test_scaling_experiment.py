"""Capacity isolation, corrected sampling, exact resume and bounded Batch launch."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch

from hulhe_escher.config import baseline_config, smoke_config, make_solver, validate_config
from hulhe_escher.scaling_config import scaling_config, CHECKPOINT_HOURS
from hulhe_escher.training_state import read_training_state
from test_optimization import assert_tree_equal


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_scaled_recipe_preserves_critics_and_learning_schedule():
    old, new = baseline_config(), scaling_config()
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {'num_layers', 'num_hiddens', 'average_policy_network_layers',
                       'structured_branch_width', 'ave_policy_buffer_size', 'policy_validation_steps'}
    assert new['ave_policy_buffer_size'] == 12_000_000
    assert new['policy_validation_steps'] == [500, 1000, 2500, 5000, 10000, 20000]
    assert new['policy_group_sampling'] == 'objective_mass'
    baseline, scaled = make_solver(smoke_config()), make_solver(scaling_config(small=True))
    params = lambda m: sum(p.numel() for p in m.parameters())
    assert params(scaled.ave_policy_trainer.model) == 777731
    assert all(params(t.model) == 777731 for t in scaled.regret_trainers)
    assert params(baseline.ave_policy_trainer.model) == 180867
    for a, b in zip(baseline.q_value_trainer.members, scaled.q_value_trainer.members):
        assert a.model.checkpoint_metadata() == b.model.checkpoint_metadata()
        assert params(a.model) == params(b.model)
    assert baseline.calibration_trainer.model.checkpoint_metadata() == scaled.calibration_trainer.model.checkpoint_metadata()
    scaled.iteration()
    scaled.train_average_policy()
    fit = scaled.ave_policy_trainer.last_fit_diagnostics
    assert fit['group_sampling'] == 'objective_mass'
    assert fit['policy_sampling_seed'] == [0, 1]
    assert fit['training_groups'] > 8  # Exercises actual draws, not exact full batch.


@pytest.mark.parametrize('key,value', [('critic_branch_width', 0), ('policy_group_sampling', 'typo')])
def test_invalid_experiment_override_rejected(key, value):
    c = scaling_config(small=True)
    c[key] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_objective_mass_scaled_resume_matches_uninterrupted(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(scaling_config(small=True)))
    common = [sys.executable, '-m', 'hulhe_escher.train', '--config', str(config), '--seed', '43',
              '--threads', '1', '--policy-every', '1', '--checkpoint-every', '1', '--recovery-seconds', '0']
    for name, iterations, resume in [('control', 3, False), ('continued', 2, False), ('continued', 3, True)]:
        outcome = subprocess.run([*common, '--output', str(tmp_path / name), '--iterations', str(iterations),
                                  *(['--resume'] if resume else [])], capture_output=True, text=True, timeout=180)
        assert outcome.returncode == 0, outcome.stderr + outcome.stdout[-3000:]
    expected = read_training_state(tmp_path / 'control/state_000003')
    actual = read_training_state(tmp_path / 'continued/state_000003')
    for key in ('regret_trainers', 'average_policy_trainer', 'q_ensemble',
                'calibration', 'gate_controller', 'rng', 'replay_rng', 'actors'):
        assert_tree_equal(expected[key], actual[key])
    fit = json.loads((tmp_path / 'continued/metrics_000003.json').read_text())['policy_validation']
    assert fit['group_sampling'] == 'objective_mass' and fit['policy_sampling_seed'] == [43, 3]


def load_launcher():
    tools = Path(__file__).resolve().parents[1] / 'tools'
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location('launch_scaling_batch', tools / 'launch_scaling_batch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('machine,cpus,workers', [('c4-standard-16', 16000, 8), ('c4-highmem-8', 8000, 4)])
def test_batch_bundle_limits_and_explicit_submission(tmp_path, machine, cpus, workers):
    launcher = load_launcher()
    out = tmp_path / 'bundle'
    plan = launcher.build_bundle(out, run_name='hulhe-wide32-test', machine=machine)
    job = json.loads((out / 'job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRetryCount'] == 0 and task['maxRunDuration'] == '201600s'
    assert task['computeResource']['cpuMilli'] == cpus
    assert plan['collection_workers'] == plan['training_threads'] == workers
    assert plan['active_training_hours'] == 32 and plan['checkpoint_hours'] == list(CHECKPOINT_HOURS)
    assert plan['config']['policy_group_sampling'] == 'objective_mass'
    assert job['allocationPolicy']['instances'][0]['policy']['machineType'] == machine
    assert not plan['slumbot'] and not plan['gtow']
    subprocess.run(['bash', '-n', str(out / 'batch.sh')], check=True)
    with pytest.raises(FileExistsError):
        launcher.build_bundle(out, run_name='hulhe-wide32-test')


def test_analysis_publisher_excludes_state_and_weights(tmp_path, monkeypatch):
    from hulhe_escher import scaling_experiment as experiment
    calls = []
    monkeypatch.setattr(experiment, 'transfer', lambda *args: calls.append(args))
    experiment.sync_analysis(tmp_path, 'gs://test/run')
    pattern = calls[0][calls[0].index('--exclude')+1]
    for name in ['training/state_000003/manifest.json', 'training/state_000003/data.npy',
                 'training/latest.json', 'training/manifest.json', 'P32.json', 'P32.pkl',
                 'source.tar.gz', 'evaluation/test/confirmatory.jsonl', 'memory-probe-ab/state/manifest.json']:
        assert re.search(pattern, name), name
    for name in ['scaling.json', 'memory-preflight.json', 'reload-P32.json',
                 'training/metrics_000003.json', 'evaluation/test/report.json']:
        assert not re.search(pattern, name), name


def test_memory_gate_stops_training_without_headroom(tmp_path, monkeypatch):
    from hulhe_escher import scaling_experiment as experiment
    (tmp_path / 'memory-preflight.json').write_text(json.dumps(dict(
        policy_group_sampling='objective_mass', checkpoint_roundtrip=True)))
    monkeypatch.setattr(experiment, 'run_child', lambda *a, **k: dict(returncode=0,
        peak_host_unavailable_bytes=10**15, monitoring_available=True))
    with pytest.raises(RuntimeError, match='headroom'):
        experiment.memory_gate(tmp_path, tmp_path / 'config.json', workers=8, threads=8)


def test_cumulative_targets_and_sampler_guard(tmp_path, monkeypatch):
    from hulhe_escher import scaling_experiment as experiment
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(scaling_config()))
    commands = []
    monkeypatch.setattr(experiment, 'run_child', lambda command, *a, **kw:
                        commands.append(command) or dict(returncode=0))
    def publish(training, root, label):
        hour = int(label[1:])
        training.mkdir(exist_ok=True)
        (training / f'metrics_{hour:06d}.json').write_text(json.dumps(dict(
            iteration=hour, nodes=100, trajectories=10, learner_seconds=60,
            policy_fit_seconds=1, persistence_seconds=1,
            collection_and_overhead_seconds=30, policy_replay={})))
        return dict(active_seconds=hour*3600+60, outer_iteration=hour,
                    metrics=dict(policy_validation=dict(group_sampling='objective_mass', policy_sampling_seed=[20261115, hour])))
    monkeypatch.setattr(experiment, 'publish_policy', publish)
    report = experiment.train_stages(tmp_path, config)
    assert report['final_recovery_verified']
    train = [c for c in commands if '--additional-training-seconds' in c]
    assert float(train[0][train[0].index('--additional-training-seconds')+1]) == 8*3600
    assert all(float(c[c.index('--additional-training-seconds')+1]) == 6*3600-60 for c in train[1:])
    assert '--resume' not in train[0] and all('--resume' in c for c in train[1:])
    monkeypatch.setattr(experiment, 'publish_policy', lambda *a: dict(metrics=dict(policy_validation={'group_sampling': 'uniform'})))
    with pytest.raises(RuntimeError, match='correction'):
        experiment.train_stages(tmp_path, config)
