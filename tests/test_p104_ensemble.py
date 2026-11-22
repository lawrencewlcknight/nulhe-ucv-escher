"""Twenty real tiny fits, probability semantics, paired statistics and cloud contracts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_policy_sampling_experiment import frozen_source
from hulhe_escher import p104_ensemble as ensemble
from hulhe_escher import p104_ensemble_evaluate as evaluate
from hulhe_escher.checkpointing import LoadedHULHEPolicy, sha256_file
from hulhe_escher.game import load_hulhe_game
from hulhe_escher.learning_evaluate import load_policy


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp('ensemble20')
    state, policy = frozen_source.__wrapped__(root)
    hashes = {p:sha256_file(p) for p in root.rglob('*') if p.is_file()}
    report = ensemble.fit(state, policy, root/'fit', threads=1, max_seconds=120, smoke=True)
    assert all(sha256_file(p) == h for p,h in hashes.items())
    return root/'fit', policy, report


def test_twenty_independent_all_replay_fits(fitted):
    root, policy, report = fitted
    paths, checked = ensemble.verified_members(root/'ensemble.json', policy=policy, smoke=True)
    assert report == checked and report['status'] == 'COMPLETE'
    assert len(paths) == 20 and len({r['initial_model_sha256'] for r in report['members']}) == 20
    assert len({r['sampling_seed'] for r in report['members']}) == 20
    for row in report['members']:
        assert row['fit']['validation_history'] == []
        assert row['fit']['training_groups'] == sum(report['partition_sizes'].values())
        assert row['fit']['updates_completed'] == row['selected_update']
        assert row['selection']['updates_completed'] == 8
    with pytest.raises(ValueError, match='smoke'):
        ensemble.verified_members(root/'ensemble.json', smoke=False)


def test_shared_encoding_vectorization_matches_individual_probabilities(fitted, monkeypatch):
    root, _, _ = fitted
    game = load_hulhe_game()
    combined = load_policy(dict(kind='ensemble', manifest=root/'ensemble.json'), game)
    paths, _ = ensemble.verified_members(root/'ensemble.json')
    members = [LoadedHULHEPolicy(game, p) for p in paths]
    rng = np.random.default_rng(72)
    states = []
    for _ in range(3):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(int(rng.choice([a for a,_ in state.chance_outcomes()])))
            else:
                states.append(state.clone())
                state.apply_action(1)  # Across all streets, including free-check/fold masks.
    for state in states:
        expected = np.mean([[m.action_probabilities(state)[a] for a in state.legal_actions()]
                            for m in members], axis=0)
        actual = combined.action_probabilities(state)
        assert set(actual) == set(state.legal_actions())
        np.testing.assert_allclose(list(actual.values()), expected, rtol=2e-6, atol=1e-7)
        assert sum(actual.values()) == pytest.approx(1., abs=1e-6)
    # Instrument encoding: exactly one call for the entire ensemble.
    calls = []
    original = combined.feature_encoder.information_state
    monkeypatch.setattr(type(combined.feature_encoder), 'information_state',
        lambda self,*args: (calls.append(1), original(*args))[1])
    combined.action_probabilities(states[0])
    assert len(calls) == 1


def test_average_is_after_softmax_not_before(fitted):
    root, _, _ = fitted
    combined = ensemble.EnsemblePolicy(load_hulhe_game(), root/'ensemble.json')
    logits = torch.tensor([[12.,0.,99.]]*10 + [[0.,1.,-99.]]*10)
    combined._forward = lambda *args:logits
    mask = torch.tensor([True,True,False])
    result = combined.probabilities(torch.zeros(1), mask)
    expected = logits.masked_fill(~mask,-1e21).softmax(-1).mean(0)
    torch.testing.assert_close(result, expected)
    assert result[2] == 0
    assert not torch.allclose(result, logits.mean(0).masked_fill(~mask,-1e21).softmax(-1))


def test_vectorization_at_production_widths():
    from hulhe_escher.config import baseline_config
    from hulhe_escher.features import make_feature_encoder, StructuredHULHEMLP
    torch.set_num_threads(1)
    config = baseline_config()
    encoder = make_feature_encoder(config['feature_encoder_id'])
    models = []
    for seed in ensemble.FIT_SEEDS:
        torch.manual_seed(seed)
        model = StructuredHULHEMLP(encoder.policy_layout, config['average_policy_network_layers'],
            3, branch_width=config['structured_branch_width']).eval()
        # Production initialization has a zero head; exercise nontrivial predictions.
        with torch.no_grad():
            for p in model.output.parameters():
                p.normal_(0,.03)
        models.append(model)
    params, buffers = torch.func.stack_module_state(models)
    x = torch.randn(5, encoder.policy_layout.total_size)
    forward = torch.vmap(lambda p,b,x:torch.func.functional_call(models[0],(p,b),(x,)),
                         in_dims=(0,0,None))
    with torch.no_grad():
        vectorized = forward(params,buffers,x)
        individual = torch.stack([m(x) for m in models])
    torch.testing.assert_close(vectorized, individual, rtol=2e-5, atol=2e-6)


def test_portable_bundle_and_tampering(fitted, tmp_path):
    import shutil
    root, policy, _ = fitted
    copied = tmp_path/'copy'
    shutil.copytree(root, copied)
    ensemble.verified_members(copied/'ensemble.json', policy=policy)
    member = copied/f'member_{ensemble.FIT_SEEDS[0]}.pkl'
    with member.open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        ensemble.verified_members(copied/'ensemble.json', policy=policy)


def test_actual_spawned_paired_evaluation(fitted, tmp_path):
    root, policy, _ = fitted
    r = evaluate.evaluate(root/'ensemble.json', policy, tmp_path/'evaluation',
        workers=2, max_seconds=120, smoke=True, opponent='check_call')
    assert r['status'] == 'COMPLETE' and r['completed_groups'] == 8 and r['completed_hands'] == 48
    assert set(r['results']) == set(evaluate.TESTS)
    assert all(not row['confirmatory'] for row in r['results'].values())
    assert r['plan']['sample_not_reduced_for_runtime']
    for row in r['results'].values():
        assert row['confidence'] == pytest.approx(1-.05/3)
        assert row['observations'] == [8]
    assert len((tmp_path/'evaluation/pilot.jsonl').read_text().splitlines()) == 4
    # Render the final summary using actual reports, including selected budgets.
    from hulhe_escher.p104_ensemble_cloud import summary
    import shutil
    shutil.copytree(root, tmp_path/'fit')
    summary(tmp_path)
    assert json.loads((tmp_path/'summary.json').read_text())['fit']['status'] == 'COMPLETE'


def test_group_statistics_and_runtime_do_not_change_sample():
    rows = [dict(head=dict(mean_bb=x), ensemble_fo2=dict(mean_bb=x+2), P104_fo2=dict(mean_bb=x))
            for x in (-2., 0., 1., 4.)]
    observations = evaluate.observations(rows)
    np.testing.assert_array_equal(observations['fo2_improvement'], [2.]*4)
    cheap = evaluate.plan_from_pilot(rows, 10000., 1e6)
    costly = evaluate.plan_from_pilot(rows, 1., 1.)
    assert cheap['selected'] == costly['selected'] and costly['selected'] >= 100000
    assert cheap['full_selected_sample_fits'] and not costly['full_selected_sample_fits']
    assert all(r['alpha'] == pytest.approx(.05/3) for r in cheap['tests'].values())


def test_insufficient_budget_never_starts_confirmation(fitted, tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    root, policy, _ = fitted
    monkeypatch.setattr(evaluate.mp, 'get_context', lambda _:SimpleNamespace(
        Pool=lambda *a,**k:nullcontext(None)))
    clock = [0.]
    monkeypatch.setattr(evaluate.time, 'monotonic', lambda:clock[0])
    def collect(pool, **kw):
        assert kw['seed'] == ensemble.PILOT_SEED and 'rows' not in kw
        clock[0] += 9.
        return [dict(head=dict(mean_bb=x), ensemble_fo2=dict(mean_bb=x), P104_fo2=dict(mean_bb=0.))
                for x in (-1.,1.,-1.,1.)]
    monkeypatch.setattr(evaluate, 'collect', collect)
    r = evaluate.evaluate(root/'ensemble.json', policy, tmp_path/'eval',
        max_seconds=10, smoke=True, opponent='check_call')
    assert r['status'] == 'INSUFFICIENT_TIME_FOR_SELECTED_SAMPLE'
    assert r['completed_groups'] == 0 and not (tmp_path/'eval/confirmatory.jsonl').exists()


def test_launch_contract_no_submit_and_shell(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_p104_ensemble as launch
    calls = []
    monkeypatch.setattr(launch.subprocess, 'run', lambda cmd,**kw:calls.append(cmd))
    launch.main(['--run-name','hulhe-p104-ensemble-test','--output',str(tmp_path/'bundle')])
    assert not calls
    plan = json.loads((tmp_path/'bundle/launch.json').read_text())
    assert plan['fit_seeds'] == list(ensemble.FIT_SEEDS)
    assert plan['sampling_seeds'] == list(ensemble.SAMPLING_SEEDS)
    assert plan['evaluation']['pilot_seed'] == ensemble.PILOT_SEED
    assert plan['evaluation']['confirmation_seed'] == ensemble.CONFIRMATION_SEED
    assert plan['evaluation']['tests'] == list(evaluate.TESTS)
    job = json.loads((tmp_path/'bundle/job.json').read_text())
    spec = job['taskGroups'][0]['taskSpec']
    assert spec['maxRunDuration'] == '43200s' and spec['maxRetryCount'] == 0
    assert plan['collection_iterations'] == 0 and plan['members'] == 20
    proc = subprocess.Popen(['bash','-n',str(tmp_path/'bundle/batch.sh')])
    assert proc.wait() == 0
    launch.main(['--run-name','hulhe-p104-ensemble-test2','--output',str(tmp_path/'submit'),'--execute'])
    assert len(calls) == 4 and calls[-1][:4] == ['gcloud','batch','jobs','submit']


@pytest.mark.parametrize('fail', [False,True])
def test_cloud_stage_guards_publication_and_source_preservation(tmp_path, monkeypatch, fail):
    from hulhe_escher import p104_ensemble_cloud as cloud
    transfers, stages = [], []
    monkeypatch.setattr(cloud, 'storage', lambda *args:transfers.append(args))
    monkeypatch.setattr(cloud, 'prepare_inputs', lambda root:(root/'source/state',root/'source/P104.pkl',{}))
    monkeypatch.setattr(cloud, 'prepare_fo2', lambda _:None)
    monkeypatch.setattr(cloud, 'summary', lambda root: (root/'summary.json').write_text('{}'))
    def child(cmd, log, **kw):
        output = Path(cmd[cmd.index('--output')+1])
        stages.append((output.name, cmd, kw))
        output.mkdir()
        (output/('experiment.json' if output.name == 'fit' else 'report.json')).write_text('{"status":"COMPLETE"}')
        return dict(returncode=int(fail), timed_out=False)
    monkeypatch.setattr(cloud, 'run_child', child)
    if fail:
        with pytest.raises(RuntimeError):
            cloud.run_cloud(tmp_path/'cloud','gs://test/ensemble')
    else:
        assert cloud.run_cloud(tmp_path/'cloud','gs://test/ensemble')['status'] == 'COMPLETE'
        assert [s[0] for s in stages] == ['fit','evaluation']
    assert all('--max-seconds' in cmd and kw['timeout'] <= 32400 for _,cmd,kw in stages)
    assert transfers[-1][0] == 'cp' and str(transfers[-1][1]).endswith('cloud_status.json')
    assert all('/source/' not in str(t) for t in transfers)
    status = json.loads((tmp_path/'cloud/results/cloud_status.json').read_text())
    assert status['status'] == ('FAILED' if fail else 'COMPLETE')


def test_source_prefix_rejected_before_writes(tmp_path):
    from hulhe_escher.p104_ensemble_cloud import run_cloud
    from hulhe_escher.p104_protocol import SOURCE_URI
    with pytest.raises(ValueError, match='overlap'):
        run_cloud(tmp_path/'cloud', SOURCE_URI+'/child')
    assert not (tmp_path/'cloud').exists()
