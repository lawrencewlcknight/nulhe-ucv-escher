"""Independent blocker/draw checks and preservation of complete v4 features."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pytest

from hulhe_escher import card_features as cf, v4_card_features as old
from hulhe_escher.features import StrategicV4HULHEFeatureEncoder, encoder_from_metadata
from test_strategic_features import code_reference


def cards(text):
    return tuple('23456789TJQKA'.index(c[0])*4+'cdhs'.index(c[1]) for c in text.split())


@pytest.mark.parametrize('hole,board', [
    ('Kc Qd', 'Ac 7c 2c'), ('As Ah', 'Ad Kc Qh'),
    ('5c 6c', '2c 3c 4d 7h'), ('2c 3d', 'As Ks Qs Js Ts'),
])
def test_general_blockers_against_independent_enumeration(hole, board):
    hole, board = cards(hole), cards(board)
    f = dict(zip(cf.CARD_NAMES, cf.card_features(hole, board)))
    hands = list(combinations([c for c in range(52) if c not in board], 2))
    ranks = [code_reference(h+board) for h in hands]
    nuts = max(ranks)
    total = sum(r == nuts for r in ranks)
    legal = [not set(h).intersection(hole) for h in hands]
    remaining = sum(r == nuts and ok for r, ok in zip(ranks, legal))
    assert f['board_nuts_removed_fraction'] == pytest.approx(1-remaining/total)
    assert f['board_nuts_legal_mass'] == pytest.approx(remaining/sum(legal))


def test_backdoor_draws_count_both_required_cards():
    windows = [set(range(i, i+5)) for i in range(9)] + [{12,0,1,2,3}]
    def straight(hand):
        return any(w <= {c//4 for c in hand} for w in windows)
    def flush(hand):
        return any(sum(c%4 == suit for c in hand) >= 5 for suit in range(4))
    for h, b in [('Ks Qs','Js 2d 3c'), ('As 2s','4d 8h Kc'), ('As Ks','Qs Js Ts')]:
        hole, board = cards(h), cards(b)
        visible = hole+board
        runs = list(combinations([c for c in range(52) if c not in visible], 2))
        f = dict(zip(cf.CARD_NAMES, cf.card_features(hole, board)))
        for name, predicate in [('straight',straight), ('flush',flush)]:
            completions = sum(predicate(visible+(a,b)) and not predicate(visible+(a,))
                              and not predicate(visible+(b,)) for a,b in runs)
            assert f['runner_runner_'+name] == pytest.approx(completions/1081)
    f = dict(zip(cf.CARD_NAMES,cf.card_features(cards('Ks Qs'),cards('Js 2d 3c'))))
    assert f['runner_runner_flush'] == pytest.approx(45/1081)


def test_old_values_preserved_and_preflop_cache_survives_eviction(monkeypatch):
    cf._preflop_features.cache_clear()
    for hole in combinations(range(52),2):
        new = cf.card_features(hole, ())
        np.testing.assert_array_equal(new[:len(old.CARD_NAMES)],old.card_features(hole,()))
        assert not new.flags.writeable
    assert cf._preflop_features.cache_info().currsize == 169
    expected = cf.card_features((48,49),()).copy()
    cf._card_features.cache_clear()
    def no_sampling(*args):
        raise AssertionError('Cached preflop vector must not resample')
    monkeypatch.setattr(cf,'_sample_completions',no_sampling)
    np.testing.assert_array_equal(cf.card_features((48,49),()),expected)


def test_v4_metadata_is_frozen():
    metadata = json.loads((Path(__file__).parent/'fixtures/encoder_v4_metadata.json').read_text())
    assert StrategicV4HULHEFeatureEncoder().metadata() == metadata
    assert encoder_from_metadata(metadata).metadata() == metadata
