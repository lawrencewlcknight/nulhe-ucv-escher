"""Identity guards and four paired, multiplicity-adjusted P32 comparisons."""
from copy import deepcopy
import json
import pickle

import numpy as np
import pytest
import torch

from test_policy_sampling_experiment import frozen_source
from hulhe_escher import sequential_p32_comparison as comparison
from hulhe_escher.checkpointing import load_checkpoint_payload, sha256_file
from hulhe_escher.features import make_feature_encoder, StructuredHULHEMLP
from hulhe_escher.deep_model import DeepResidualHULHEMLP
from hulhe_escher.hybrid_model import HybridResidualHULHEMLP
from hulhe_escher.scaling_config import scaling_config, deep_scaling_config, hybrid_scaling_config
from hulhe_escher.sequential_model import ProtectedSequentialNetwork
from hulhe_escher.sequential_training import sequential_config


@pytest.fixture
def pinned(frozen_source, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    _, policy = frozen_source
    base = load_checkpoint_payload(policy)
    root = tmp_path/'inputs'
    root.mkdir()
    fit = dict(status='COMPLETE', smoke=False, source_iteration=1132, results=[])
    files, policies = {}, {}
    iterations = dict(zip(comparison.MODEL_TYPES, (659, 953, 868, 777)))
    factories = {'Wide-P32': (scaling_config, StructuredHULHEMLP),
                 'Deep-P32': (deep_scaling_config, DeepResidualHULHEMLP),
                 'Hybrid-P32': (hybrid_scaling_config, HybridResidualHULHEMLP)}
    for label in (*comparison.MODEL_TYPES, *(f'P32-corrected-{s}' for s in comparison.SEEDS)):
        payload = deepcopy(base)
        payload.update(outer_iteration=iterations.get(label, 1132), training_elapsed_seconds=115600., seed=20261115)
        if label in comparison.MODEL_TYPES:
            if label == comparison.CANDIDATE:
                config = sequential_config()
                encoder = make_feature_encoder(config['feature_encoder_id'])
                model = ProtectedSequentialNetwork(encoder.policy_layout)
            else:
                factory, cls = factories[label]
                config = factory(small=True)
                encoder = make_feature_encoder(config['feature_encoder_id'])
                model = cls(encoder.policy_layout, config['average_policy_network_layers'], 3,
                            branch_width=config['structured_branch_width'])
            payload.update(training_config=config, policy_state_dict=model.state_dict(),
                policy_model=model.checkpoint_metadata(), policy_network_layers=config['average_policy_network_layers'])
        with (root/(label+'.pkl')).open('wb') as handle:
            pickle.dump(payload, handle)
        sha = sha256_file(root/(label+'.pkl'))
        policies[label] = dict(iteration=payload['outer_iteration'], active_seconds=115600.)
        if label.startswith('P32-corrected'):
            fit['results'].append(dict(arm='objective_mass', fit_seed=int(label.rsplit('-', 1)[-1]),
                fit=dict(group_sampling='objective_mass'), sha256=sha))
            continue
        marker = dict(label='P32', outer_iteration=payload['outer_iteration'], active_seconds=115600.,
            smoke=False, sha256=sha, metrics=dict(policy_validation=dict(group_sampling='objective_mass')))
        if label == comparison.CANDIDATE:
            marker['metrics']['policy_validation'].update(policy_sampling_seed=[20261115, 659],
                protected_training=dict(protocol='protected_selective_v1', coefficient=.3))
            (root/'Sequential-P32.json').write_text(json.dumps(marker))
            receipt = dict(status='VERIFIED', architecture='protected_sequential_v1',
                policy_group_sampling='objective_mass', protection_coefficient=.3,
                source_modified=False, config_changed=False, training_source_fingerprint='test-source')
            provenance = dict(sequential_source=True, policy_group_sampling='objective_mass',
                              training_source_fingerprint='test-source')
            (root/'Sequential-sampler-verification.json').write_text(json.dumps(receipt))
            (root/'Sequential-provenance.json').write_text(json.dumps(provenance))
        else:
            plan = dict(config=payload['training_config'], seed=payload['seed'])
            training = dict(config=plan['config'], seed=plan['seed'], status='TRAINING_COMPLETE',
                smoke=False, fresh_initialization=True, final_recovery_verified=True, checkpoints=dict(P32=marker))
            for suffix, data in [('launch', plan), ('scaling', training)]:
                (root/(label+'-'+suffix+'.json')).write_text(json.dumps(data))
            if label == 'Hybrid-P32':
                (root/'Hybrid-P32.json').write_text(json.dumps(marker))
    (root/'P32-fit.json').write_text(json.dumps(fit))
    for path in root.iterdir():
        files[path.name] = dict(uri='gs://test/source/'+path.name, sha256=sha256_file(path))
    source = tmp_path/'sources.json'
    source.write_text(json.dumps(dict(files=files, policies=policies, fit_seeds=list(comparison.SEEDS))))
    monkeypatch.setattr(comparison, 'SOURCES', source)
    return root


def repin(path):
    sources = json.loads(comparison.SOURCES.read_text())
    sources['files'][path.name]['sha256'] = sha256_file(path)
    comparison.SOURCES.write_text(json.dumps(sources))


def test_all_inputs_hashed_before_unpickling(pinned, monkeypatch):
    specs, sources = comparison.verify_inputs(pinned)
    assert len(specs) == 6 and len(sources['files']) == 17
    (pinned/'Hybrid-P32.pkl').write_bytes(b'wrong')
    monkeypatch.setattr(comparison, 'load_checkpoint_payload', lambda _: pytest.fail('hash first'))
    with pytest.raises(ValueError, match='checksum'):
        comparison.verify_inputs(pinned)


@pytest.mark.parametrize('field,value', [('outer_iteration', 658), ('training_elapsed_seconds', 28800.),
    ('feature_encoder', {'wrong': True}), ('policy_model', {'type': 'wrong'})])
def test_mismatched_checkpoint_rejected(pinned, monkeypatch, field, value):
    original = comparison.load_checkpoint_payload
    def changed(path):
        payload = original(path)
        if path.name == 'Hybrid-P32.pkl':
            payload[field] = value
        return payload
    monkeypatch.setattr(comparison, 'load_checkpoint_payload', changed)
    with pytest.raises(ValueError, match='mismatch|matched'):
        comparison.verify_inputs(pinned)


@pytest.mark.parametrize('change', ['sampler', 'protection', 'receipt', 'refit', 'hybrid', 'config'])
def test_invalid_provenance_even_with_recomputed_hash(pinned, change):
    if change in ('sampler', 'protection'):
        path = pinned/'Sequential-P32.json'
        record = json.loads(path.read_text())
        fit = record['metrics']['policy_validation']
        if change == 'sampler': fit['group_sampling'] = 'uniform'
        else: fit['protected_training']['coefficient'] = 0.
    elif change == 'receipt':
        path = pinned/'Sequential-sampler-verification.json'
        record = json.loads(path.read_text()); record['source_modified'] = True
    elif change == 'refit':
        path = pinned/'P32-fit.json'
        record = json.loads(path.read_text()); record['results'].pop()
    elif change == 'hybrid':
        path = pinned/'Hybrid-P32.json'
        record = json.loads(path.read_text()); record['smoke'] = True
    else:
        path = pinned/'Deep-P32-launch.json'
        record = json.loads(path.read_text()); record['config']['epsilon'] = .123
    path.write_text(json.dumps(record)); repin(path)
    with pytest.raises(ValueError, match='mismatch'):
        comparison.verify_inputs(pinned)


def rows(values):
    return [{name: dict(mean_bb=v+i, legs_bb=[v+i-1, v+i+1])
             for i, (name, *_) in enumerate(comparison.matchups())} for v in values]


def test_deals_not_hands_or_refits_are_independent_and_no_runtime_reduction():
    data = rows([1., 2., 3., 4.])
    np.testing.assert_array_equal(comparison.observations(data, 'Small-P32-corrected'), [1.5, 2.5, 3.5, 4.5])
    np.testing.assert_array_equal(comparison.observations(data, 'Hybrid-P32'), [5., 6., 7., 8.])
    plan = comparison.common_plan(data, 1000., 10000., minimum=100000, maximum=1000000)
    assert plan['selected'] == max(p['selected'] for p in plan['by_comparator'].values())
    assert plan['alpha_per_test'] == .0125 and plan['familywise_alpha'] == .05
    assert all(p['alpha'] == .0125 and p['pilot_excluded'] for p in plan['by_comparator'].values())
    short = comparison.common_plan(data, 1., 1., minimum=100000, maximum=1000000)
    assert short['selected'] == plan['selected'] and not short['full_selected_sample_fits']
    capped = comparison.common_plan(rows([-100., 100.]), 1e9, 1e9, minimum=8, maximum=10)
    assert all(p['capped'] for p in capped['by_comparator'].values())
    pairs = comparison.matchups()
    assert len(pairs) == 5 and all(a == comparison.CANDIDATE for _, a, *_ in pairs)
    assert len({stream for *_, stream in pairs}) == 5


def test_real_spawned_six_policy_smoke(pinned, tmp_path):
    before = {p: sha256_file(p) for p in pinned.iterdir()}
    report = comparison.run_evaluation(pinned, tmp_path/'evaluation', workers=2, max_seconds=180, smoke=True)
    assert report['status'] == 'COMPLETE', report
    assert report['completed_groups'] == 8 and report['completed_hands'] == 80
    assert set(report['results']) == set(comparison.COMPARATORS)
    for result in report['results'].values():
        assert result['observations'] == [8] and not result['confirmatory']
        assert result['conclusion'] == 'smoke_only' and result['confidence'] == .9875
    assert set(report['by_fit_seed']) == {'1701', '1702'}
    assert all(not r['confirmatory'] for r in report['by_fit_seed'].values())
    assert all(sha256_file(p) == sha for p, sha in before.items())


class FakePool:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


@pytest.mark.parametrize('mode', ['complete', 'partial', 'error', 'pilot', 'budget', 'replaced'])
def test_pilot_exclusion_no_partial_claims_and_final_identity_check(tmp_path, monkeypatch, mode):
    verifications = []
    def verify(path):
        verifications.append(path)
        if mode == 'replaced' and len(verifications) > 1: raise ValueError('replaced input')
        return {}, {}
    monkeypatch.setattr(comparison, 'verify_inputs', verify)
    monkeypatch.setattr(comparison.mp, 'get_context', lambda _: type('Context', (), {'Pool': FakePool}))
    monkeypatch.setattr(comparison, 'common_plan', lambda *a, **kw: dict(selected=4,
        by_comparator={}, full_selected_sample_fits=mode != 'budget'))
    seeds = []
    def collect(pool, *, seed, count, rows=None, **kw):
        seeds.append(seed)
        if rows is None: return globals()['rows']([999.]*(1 if mode == 'pilot' else count))
        rows.extend(globals()['rows']([1., 2., 3., 4.][:2 if mode in ('error', 'partial') else 4]))
        if mode == 'error': raise RuntimeError('failure')
        return rows
    monkeypatch.setattr(comparison, 'collect', collect)
    report = comparison.run_evaluation(tmp_path, tmp_path/'out')
    if mode in ('pilot', 'budget'):
        assert not report['results'] and len(seeds) == 1
    else:
        assert seeds == [comparison.DEAL_SEED, comparison.DEAL_SEED+1]
        result = report['results']['Small-P32-corrected']
        assert result['bb_per_100'] == (200 if mode in ('partial', 'error') else 300)
        assert result['confirmatory'] == (mode == 'complete')
        assert result['observations'] == ([2] if mode in ('partial', 'error') else [4])
        assert all(not r['confirmatory'] for r in report['by_fit_seed'].values())
    if mode == 'replaced': assert report['status'] == 'EVALUATION_ERROR'
