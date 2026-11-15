"""Execution optimisations must preserve the previous training trajectory."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_escher import frozen_target_cache as targets
from hulhe_escher import features, card_features as cards
from hulhe_escher.config import make_solver, smoke_config
from hulhe_escher.training_state import build_training_state
from unbiased_escher import holdem_fitting
from vr_deep_cfr.solver import RegretTrainer
from test_hulhe import state_at_street


def assert_tree_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, (torch.Tensor, np.ndarray)):
        if isinstance(left, torch.Tensor):
            left, right = left.detach().cpu().numpy(), right.detach().cpu().numpy()
        assert left.shape == right.shape and left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()
    else:
        np.testing.assert_equal(left, right)  # Includes matching NaN diagnostics.


def reference_top_ranks(mask, k=5):
    return np.sort(np.where(mask, np.arange(2, 15), 0), axis=1)[:, ::-1][:, :k]


def clear_card_caches():
    # Reference and optimised runs must both compute features, including preflop.
    for value in vars(cards).values():
        if hasattr(value, 'cache_clear'):
            value.cache_clear()


def test_rank_lookup_exhausts_all_masks_and_preserves_owned_rows():
    masks = (np.arange(8192)[:, None] & (1 << np.arange(13))) != 0
    for k in range(1, 6):
        actual = cards._top_ranks(masks, k)
        expected = reference_top_ranks(masks, k)
        assert_tree_equal(actual, expected)
        actual[:] = -1
        assert_tree_equal(cards._top_ranks(masks, k), expected)
    assert not cards._TOP_RANKS.flags.writeable
    assert cards._TOP_RANKS.nbytes < 1024 * 1024


@pytest.mark.parametrize('size', [5, 6, 7])
def test_rank_lookup_matches_sort_for_random_hands_and_empty_batches(monkeypatch, size):
    rng = np.random.default_rng(2026)
    hands = np.array([rng.choice(52, size, replace=False) for _ in range(10000)])
    for batch in (hands, hands[:0]):
        actual = cards.rank_cards(batch)
        with monkeypatch.context() as patch:
            patch.setattr(cards, '_top_ranks', reference_top_ranks)
            expected = cards.rank_cards(batch)
        assert_tree_equal(actual, expected)


def reference_suits(*channels):
    signatures = [tuple(float(x) for channel in channels for x in channel[suit])
                  for suit in range(4)]
    order = sorted(range(4), key=lambda suit: signatures[suit], reverse=True)
    return tuple(channel[order].reshape(-1) for channel in channels)


def reference_forward(self, model, x, mask):
    with torch.device(self.device):
        x = torch.as_tensor(x, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        with torch.no_grad():
            return self.predict(model, x, mask).cpu().numpy()


def reference_batch(buffer, indices, device):
    floats = ('history', 'next_history', 'next_state', 'reward')
    integers = ('next_legal_actions_mask', 'next_player', 'action', 'done')
    return tuple(torch.as_tensor(getattr(buffer, name + '_buf')[indices],
                                 dtype=dtype, device=device)
                 for names, dtype in ((floats, torch.float32), (integers, torch.int64))
                 for name in names)


def install_reference(monkeypatch):
    """The pre-optimisation operations; no alternate learning configuration."""
    monkeypatch.setattr(cards, '_top_ranks', reference_top_ranks)
    monkeypatch.setattr(features, '_canonicalise_suits', reference_suits)
    monkeypatch.setattr(features.LegacyHULHEFeatureEncoder, '_cached_state_features',
                        lambda self, state, key, build: build())
    monkeypatch.setattr(RegretTrainer, 'forward', reference_forward)
    monkeypatch.setattr(targets, 'transition_batch', reference_batch)
    for module in (targets, holdem_fitting):
        monkeypatch.setattr(module, 'target_batch_indices',
                            lambda start, stop, batch: np.resize(np.arange(start, stop), batch))


@pytest.mark.parametrize('channels_count', [4, 5])
def test_suit_order_matches_float_tuples_including_ties(channels_count):
    rng = np.random.default_rng(27)
    for _ in range(1000):
        channels = [rng.integers(0, 2, (4, 13)).astype(np.float32)
                    for _ in range(channels_count)]
        # Exercise stable ties and signed zero as well as dense binary patterns.
        for channel in channels:
            channel[1] = channel[0]
            channel[channel == 0] = -0.0
        actual = features._canonicalise_suits(*channels)
        expected = reference_suits(*channels)
        assert all(a.tobytes() == b.tobytes() for a, b in zip(actual, expected))


def test_target_batches_share_full_rows_and_preserve_tail_padding():
    class Unused:
        def __getitem__(self, index):
            raise AssertionError('Unused current-state fields must not be gathered')
    data = np.arange(11 * 7, dtype=np.float32).reshape(11, 7)
    buffer = SimpleNamespace(history_buf=Unused(), action_buf=Unused(),
        next_history_buf=data, next_state_buf=data, reward_buf=data[:, 0],
        next_legal_actions_mask_buf=np.ones((11, 7), dtype=np.int64),
        next_player_buf=np.zeros(11, dtype=np.int64), done_buf=np.zeros(11, dtype=np.int64))
    for start, stop in ((0, 4), (4, 8), (8, 11)):
        indices = targets.target_batch_indices(start, stop, 4)
        batch = targets.transition_batch(buffer, indices, 'cpu')
        assert batch[0] is None and batch[6] is None
        expected = data[np.resize(np.arange(start, stop), 4)]
        np.testing.assert_array_equal(batch[1].numpy(), expected)
        assert np.shares_memory(batch[1].numpy(), data) == (stop - start == 4)


def training_snapshot(config):
    clear_card_caches()
    solver = make_solver(config, seed=771)
    snapshots = []
    for _ in range(3):
        solver.iteration()
        solver.train_average_policy()
        payload = build_training_state(solver, seed=771, checkpoint_id='equivalence',
            repository_commit='test', config=config, captured_checkpoints=[])
        for key in ('training_elapsed_seconds', 'checkpoint_rows'):
            payload['solver'].pop(key)
        # Recovery omits fresh-regret replay; compare it explicitly as well.
        payload['regret_replay'] = [{key: value[:len(t.buffer)].copy()
            for key, value in vars(t.buffer).items() if isinstance(value, np.ndarray)}
            for t in solver.regret_trainers]
        snapshots.append(deepcopy(payload))
    return snapshots


@pytest.mark.parametrize('threads', [1, 4])
def test_training_matches_reference_including_replay_rng_and_adam(monkeypatch, threads):
    torch.set_num_threads(threads)
    try:
        config = smoke_config()
        # Nondivisible capacities exercise padded tails and ring/reservoir
        # replacement; repeated fits exercise learned (nonzero) network weights.
        for key in ('advantage_buffer_size', 'ave_policy_buffer_size',
                    'baseline_buffer_size', 'calibration_buffer_size'):
            config[key] = 37
        config['num_traversals'] = 64
        actual = training_snapshot(config)
        with monkeypatch.context() as patch:
            install_reference(patch)
            expected = training_snapshot(config)
        assert_tree_equal(actual, expected)
    finally:
        torch.set_num_threads(1)


def test_full_size_minibatches_and_padded_labels_match_reference(monkeypatch):
    """Exercise the production GEMM shape, not just small smoke minibatches."""
    torch.set_num_threads(4)
    try:
        config = smoke_config()
        for key in ('advantage_buffer_size', 'ave_policy_buffer_size', 'calibration_buffer_size'):
            config[key] = 2051
        config['baseline_buffer_size'] = 4102  # 2051 per critic, with a 3-row tail.
        for key in ('advantage_batch_size', 'ave_policy_batch_size',
                    'baseline_batch_size', 'calibration_batch_size'):
            config[key] = 2048

        def fit():
            solver = make_solver(config, seed=731)
            rng = np.random.default_rng(998)
            for trainer in solver.regret_trainers:
                b = trainer.buffer
                b.cur_id = b.buffer_size
                b.infostate_buf[:] = rng.random(b.infostate_buf.shape, dtype=np.float32)
                b.q_value_buf[:] = rng.normal(size=b.q_value_buf.shape)
                b.q_value_mask_buf[:] = 1
                b.iteration_buf[:] = 1
            for member in solver.q_value_trainer.members:
                b = member.buffer
                b.size = b.buffer_size
                for field in ('history_buf', 'next_history_buf', 'next_state_buf', 'reward_buf'):
                    array = getattr(b, field)
                    array[:] = rng.random(array.shape, dtype=np.float32)
                b.action_buf[:] = np.arange(b.size) % solver.action_size
                b.next_legal_actions_mask_buf[:] = 1
                b.next_player_buf[:] = np.arange(b.size) % 2
                b.done_buf[:] = np.arange(b.size) % 3 == 0
            results = []
            for iteration in (1, 2):
                for trainer in solver.regret_trainers:
                    trainer.train_model(iteration)
                for member in solver.q_value_trainer.members:
                    results.append(targets.build_target_cache(member, iteration).clone())
                    member.train_model(iteration)
                results.extend(deepcopy((trainer.model.state_dict(), trainer.optimizer.state_dict()))
                    for trainer in [*solver.regret_trainers, *solver.q_value_trainer.members])
            results.append(solver._capture_rng_state())
            results.extend(deepcopy(trainer.buffer.rng.bit_generator.state)
                for trainer in [*solver.regret_trainers, *solver.q_value_trainer.members])
            return results

        actual = fit()
        with monkeypatch.context() as patch:
            install_reference(patch)
            expected = fit()
        assert_tree_equal(actual, expected)
    finally:
        torch.set_num_threads(1)


@pytest.mark.parametrize('encoder_type', [features.LegacyHULHEFeatureEncoder,
    features.StrategicV3HULHEFeatureEncoder, features.StrategicV4HULHEFeatureEncoder,
    features.HULHEFeatureEncoder])
def test_native_state_cache_observers_mutations_clones_and_ownership(encoder_type):
    encoder = encoder_type()
    for street in range(4):
        state = state_at_street(street)
        for player in (0, 1):
            expected = encoder._encode_information_state(state, player)
            assert_tree_equal(encoder.information_state(state, player), expected)
            encoder.information_state(state, player)[:] = -999
            assert_tree_equal(encoder.information_state(state, player), expected)
        full = encoder._encode_full_state(state)
        encoder.full_state(state)[:] = -999
        assert_tree_equal(encoder.full_state(state), full)
        token, _ = encoder._state_feature_cache[state]
        clone = state.clone()
        assert clone not in encoder._state_feature_cache
        clone.apply_action(2)
        assert_tree_equal(encoder.information_state(clone), encoder._encode_information_state(clone))
        assert encoder._state_feature_cache[state][0] == token
        assert_tree_equal(encoder.full_state(state), full)
        state.apply_action(1)
        for player in (0, 1):
            assert_tree_equal(encoder.information_state(state, player),
                              encoder._encode_information_state(state, player))
        assert encoder._state_feature_cache[state][0] != token
        state.apply_action(1)
        while state.is_chance_node():
            with pytest.raises(ValueError, match='decision states'):
                encoder.information_state(state, 0)
            state.apply_action(state.chance_outcomes()[0][0])
        assert_tree_equal(encoder.full_state(state), encoder._encode_full_state(state))
        if state.is_terminal():
            assert not encoder.full_state(state).any()
            with pytest.raises(ValueError, match='decision states'):
                encoder.information_state(state, 0)


def test_native_state_cache_lifetime_instances_and_serialization(monkeypatch):
    import gc
    import pickle
    import weakref
    state = state_at_street()
    encoders = [features.HULHEFeatureEncoder(), features.HULHEFeatureEncoder()]
    for encoder in encoders:
        expected = encoder.information_state(state)
        # A repeated call must not rebuild features.
        with monkeypatch.context() as patch:
            patch.setattr(encoder, '_encode_information_state',
                lambda *args: pytest.fail('Unchanged state was re-encoded'))
            assert_tree_equal(encoder.information_state(state), expected)
        for restored in (deepcopy(encoder), pickle.loads(pickle.dumps(encoder))):
            assert len(restored._state_feature_cache) == 0
            assert_tree_equal(restored.information_state(state), expected)
    assert encoders[0]._state_feature_cache is not encoders[1]._state_feature_cache
    ref = weakref.ref(state)
    del state
    gc.collect()
    assert ref() is None
    assert all(len(e._state_feature_cache) == 0 for e in encoders)


def test_native_state_cache_history_token_distinguishes_equal_depth_branches():
    # Universal poker currently lacks undo_action. Exercise its future-safe
    # invalidation contract with a mutable history of the same length.
    class State:
        actions = [0, 5, 10, 15, 1]
        def history(self):
            return self.actions
    encoder = features.HULHEFeatureEncoder()
    state = State()
    first = encoder._cached_state_features(state, ('info', 0), lambda: np.array([1]))
    state.actions = [0, 5, 10, 15, 2]
    second = encoder._cached_state_features(state, ('info', 0), lambda: np.array([2]))
    assert first[0] == 1 and second[0] == 2
