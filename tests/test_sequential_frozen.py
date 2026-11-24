"""Real full-width fits, gradient routing, source protection and Batch contracts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_policy_sampling_experiment import frozen_source
from test_regret_diagnostic import diagnostic_inputs
from hulhe_escher import sequential_frozen as diag
from hulhe_escher import sequential_protocol as protocol
from hulhe_escher.features import make_feature_encoder, ENCODER_ID
from hulhe_escher.sequential_model import SequentialStreetNetwork
from hulhe_escher.checkpointing import sha256_file


@pytest.fixture
def small_data():
    torch.set_num_threads(1)
    encoder = make_feature_encoder(ENCODER_ID)
    rng = np.random.default_rng(8)
    x = rng.normal(size=(200, encoder.policy_layout.total_size)).astype(np.float32)
    x[:, encoder.replay_context_columns()[1]] = np.eye(4)[np.arange(len(x)) % 4]
    y = rng.normal(size=(len(x), 3)).astype(np.float32)
    mask = np.ones_like(y); mask[:, 0] = 0
    return encoder, diag.dataset(x, y, mask, np.ones(len(x)), encoder, 'regret')


def test_exact_parameter_count_forward_reload_and_batch_independence(small_data):
    encoder, d = small_data
    model = SequentialStreetNetwork(encoder.policy_layout)
    assert sum(p.numel() for p in model.parameters()) == 1496067
    assert sum(p.numel() for p in model.specialists[0].parameters()) == 173856
    x = torch.from_numpy(d['features'][:5])
    # Nonzero output makes this a meaningful forward/gradient check.
    with torch.no_grad():
        model.output.weight.fill_(.01)
    clone = SequentialStreetNetwork(encoder.policy_layout)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
    torch.testing.assert_close(model(x), torch.cat([model(row[None]) for row in x]), atol=1e-5, rtol=1e-5)
    assert model.checkpoint_metadata()['diagnostic_only']


def test_inactive_specialists_and_adam_momentum_remain_unchanged(small_data):
    encoder, d = small_data
    model = SequentialStreetNetwork(encoder.policy_layout)
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    ids = np.arange(16)
    for _ in range(3):
        diag.update(model, opt, d, ids)  # Seed optimizer momentum in EVERY block.
    frozen = [deepcopy(m.state_dict()) for m in model.specialists]
    shared = deepcopy(model.shared_stages[0].state_dict())
    stats = diag.update(model, opt, d, ids, street=0)
    assert stats['specialist_gradient_norms'][0] > 0
    assert stats['specialist_gradient_norms'][1:] == [0, 0, 0]
    for s in range(1, 4):
        for name, value in model.specialists[s].state_dict().items():
            torch.testing.assert_close(value, frozen[s][name], atol=0, rtol=0)
    assert any(not torch.equal(v, shared[k]) for k, v in model.shared_stages[0].state_dict().items())
    assert any(not torch.equal(v, frozen[0][k]) for k, v in model.specialists[0].state_dict().items())


def test_current_target_auxiliary_does_not_require_teacher(small_data):
    encoder, d = small_data
    model = SequentialStreetNetwork(encoder.policy_layout)
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    stats = diag.update(model, opt, d, np.arange(12), street=3,
                        aux_ids=np.arange(12, 24), coefficient=.3)
    assert stats['auxiliary'] > 0
    assert np.isfinite(stats['gradient_norm'])


def test_masked_losses_and_policy_preservation_have_correct_direction():
    pred = torch.tensor([[1., 5., -1.]], requires_grad=True)
    ref = torch.tensor([[0., -100., 1.]])
    mask = torch.tensor([[1., 0., 1.]])
    target = torch.tensor([[.5, 0., .5]])
    kl = diag.policy_kl(pred, ref, mask)
    assert kl.item() > 0
    kl.backward()
    assert pred.grad[0, 1] == 0
    assert pred.grad[0, 0] > 0 and pred.grad[0, 2] < 0
    assert diag.policy_kl(ref, ref, mask).item() == 0
    changed = pred.detach().clone(); changed[0, 1] = 999
    torch.testing.assert_close(diag.losses(pred, target, mask, 'policy'),
                               diag.losses(changed, target, mask, 'policy'))
    torch.testing.assert_close(diag.losses(pred, target, mask, 'regret'),
                               diag.losses(changed, target, mask, 'regret'))


def test_sampling_preserves_mass_without_double_weighting(small_data):
    _, d = small_data
    d['weight'][:4] = [1, 9, 1, 9]
    draws = diag.sample(d, np.arange(4), 20000, np.random.default_rng(7))
    np.testing.assert_allclose(np.bincount(draws)/len(draws), [.05, .45, .05, .45], atol=.01)
    # Street-mass selection then conditional sampling gives the same row objective.
    w = np.array([1., 9., 2., 8.]); street = np.array([0, 0, 1, 1])
    p = [w[street == s].sum()/w.sum() * w[i]/w[street == s].sum() for i, s in enumerate(street)]
    np.testing.assert_allclose(p, w/w.sum())


def test_duplicates_cannot_cross_partitions_and_inputs_validated(small_data):
    encoder, d = small_data
    d['features'][1] = d['features'][0]
    again = diag.dataset(d['features'], d['target'], d['mask'], d['weight'], encoder, 'regret')
    assert all((0 in ids) == (1 in ids) for ids in again['partitions'])
    bad = d['features'].copy(); bad[:, encoder.replay_context_columns()[1]] = 0
    with pytest.raises(ValueError, match='street'):
        diag.dataset(bad, d['target'], d['mask'], d['weight'], encoder, 'regret')


def test_subsampling_preserves_full_street_mass(diagnostic_inputs, monkeypatch):
    captures, state, policy, _ = diagnostic_inputs
    monkeypatch.setattr(protocol, 'POLICY_GROUPS_PER_STREET', 12)
    data, _, provenance = diag.prepare_data(state, policy, captures, smoke=True)
    d = data['policy']
    for s, row in enumerate(provenance['policy_coverage']['diagnostic_subset']):
        assert row['retained_groups'] == 12
        assert d['weight'][d['street'] == s].sum() == pytest.approx(row['full_mass'])


@pytest.mark.parametrize('kind', ['policy', 'regret'])
def test_full_size_network_learns_known_targets(small_data, kind):
    encoder, d = small_data
    torch.manual_seed(55)
    d['kind'] = kind
    d['target'][:] = [0., .2, .8] if kind == 'policy' else [0., -1., 2.]
    model = SequentialStreetNetwork(encoder.policy_layout)
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    train, valid, _ = d['partitions']
    before = diag.scores(model, d, valid)['overall']['error']
    rng = np.random.default_rng(19)
    for _ in range(30):
        diag.update(model, opt, d, diag.sample(d, train, 32, rng))
    after = diag.scores(model, d, valid)['overall']['error']
    assert after < before*.2


def test_real_full_network_smoke_pipeline_preserves_source(diagnostic_inputs, tmp_path, monkeypatch):
    captures, state, policy, _ = diagnostic_inputs
    inputs = [p for root in (captures, state) for p in root.rglob('*') if p.is_file()] + [policy]
    hashes = {p:sha256_file(p) for p in inputs}
    out = tmp_path/'checks'
    original_scores = diag.scores
    def enforce_audit_lock(model, data, ids, **kwargs):
        if np.array_equal(ids, data['partitions'][2]):
            assert (out/'selection.json').exists(), 'Audit inspected before selections committed'
        return original_scores(model, data, ids, **kwargs)
    monkeypatch.setattr(diag, 'scores', enforce_audit_lock)
    report = diag.run(state, policy, captures, out, threads=1, max_seconds=240, smoke=True)
    assert report['status'] == 'COMPLETE'
    assert len(report['results']) == 24 and len(report['warmups']) == 6
    assert all(sha256_file(p) == h for p, h in hashes.items())
    selection = json.loads((out/'selection.json').read_text())
    assert len(selection['selections']) == 24
    for row in report['results']:
        scores = [(r['validation']['overall']['error'], r['update']) for r in row['validation_history']]
        assert row['selected_update'] == min(scores)[1]
        assert np.isfinite(row['audit']['overall']['error'])
        assert len(row['interference']['error_changes']) == 4
        assert row['interference']['intervention_updates'] == 5
    for name in ('policy', 'regret_0', 'regret_1'):
        for seed in protocol.SEEDS:
            rows = [r for r in report['results'] if r['dataset'] == name and r['seed'] == seed]
            assert len({r['initial_sha256'] for r in rows}) == 1
    assert not list(out.glob('*.pt')) and not list(out.glob('*.pkl'))


def test_invalid_inputs_fail_closed(diagnostic_inputs, tmp_path):
    captures, state, policy, _ = diagnostic_inputs
    with pytest.raises(ValueError, match='overlaps'):
        diag.run(state, policy, captures, state/'bad', smoke=True)
    with (captures/'regret_0_before.npz').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        diag.run(state, policy, captures, tmp_path/'bad', smoke=True)
    assert json.loads((tmp_path/'bad/report.json').read_text())['status'] == 'FAILED'


def test_launch_bundle_is_bounded_nonlaunching_and_dependency_free(tmp_path):
    launcher = Path(__file__).resolve().parents[1]/'tools/launch_sequential_frozen.py'
    root = tmp_path/'bundle'
    subprocess.run([sys.executable, str(launcher), '--run-name', 'hulhe-sequential-frozen-test',
                    '--output', str(root)], check=True, capture_output=True)
    plan = json.loads((root/'launch.json').read_text())
    job = json.loads((root/'job.json').read_text())
    assert plan['collection_iterations'] == 0 and plan['architecture_parameters'] == 1496067
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRetryCount'] == 0 and task['maxRunDuration'] == '10800s'
    assert job['allocationPolicy']['location']['allowedLocations'] == ['regions/europe-west1']
    subprocess.run(['bash', '-n', str(root/'batch.sh')], check=True)


def test_cloud_overlap_protection():
    from hulhe_escher.sequential_frozen_cloud import reject_overlap
    for uri in (protocol.SOURCE_URI, protocol.CAPTURE_URI, protocol.SOURCE_URI+'/nested'):
        with pytest.raises(ValueError, match='overlap'):
            reject_overlap(uri)


def test_cloud_rejects_metadata_before_large_download(tmp_path, monkeypatch):
    from hulhe_escher import sequential_frozen_cloud as cloud
    calls = []
    monkeypatch.setattr(cloud, 'prepare', lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(cloud, 'read_metadata', lambda *a: ({}, {}))
    monkeypatch.setattr(cloud, 'sha256_file', lambda p: 'wrong')
    with pytest.raises(ValueError, match='metadata mismatch'):
        cloud.prepare_inputs(tmp_path)
    assert calls == [{'metadata_only': True}]


def test_cloud_failed_child_is_not_reported_complete(tmp_path, monkeypatch):
    from hulhe_escher import sequential_frozen_cloud as cloud
    monkeypatch.setattr(cloud, 'prepare_inputs', lambda root: (root/'state', root/'policy', root/'captures'))
    monkeypatch.setattr(cloud, 'run_child', lambda *a, **kw: {'timed_out': True, 'returncode': -15})
    uploads = []
    monkeypatch.setattr(cloud, 'storage', lambda *a: uploads.append(a))
    with pytest.raises(RuntimeError, match='90-minute'):
        cloud.run_cloud(tmp_path/'cloud', 'gs://test/sequential-new')
    status = json.loads((tmp_path/'cloud/results/cloud_status.json').read_text())
    assert status['status'] == 'FAILED'
    assert uploads[-1][-1] == 'gs://test/sequential-new/cloud_status.json'
    assert all('/source/' not in str(a) for a in uploads)
