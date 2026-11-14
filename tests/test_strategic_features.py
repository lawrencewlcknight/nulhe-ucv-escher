"""Independent ranking, information-boundary and fixed-limit arithmetic oracles."""
from collections import Counter
from itertools import combinations, permutations
import random

import numpy as np
import pytest
import torch

from hulhe_escher.card_features import (
    CARD_NAMES, CARD_INDEX, card_features, rank_cards, _card_features,
    canonical_cards, _sample_completions,
)
from hulhe_escher.betting_features import BETTING_INDEX, _public_economics, public_rounds
from hulhe_escher.features import HULHEFeatureEncoder, LegacyHULHEFeatureEncoder
from hulhe_escher.game import load_hulhe_game
from test_hulhe import state_at_street


def five_reference(cards):
    ranks = [c//4+2 for c in cards]
    count = Counter(ranks)
    groups = sorted(((n,r) for r,n in count.items()), reverse=True)
    distinct = sorted(count)
    straight = (5 if distinct == [2,3,4,5,14] else
                max(distinct) if len(distinct) == 5 and max(distinct)-min(distinct) == 4 else 0)
    flush = len({c % 4 for c in cards}) == 1
    if flush and straight: result = (8,straight)
    elif groups[0][0] == 4: result = (7,groups[0][1],groups[1][1])
    elif [n for n,r in groups] == [3,2]: result = (6,groups[0][1],groups[1][1])
    elif flush: result = (5,*sorted(ranks,reverse=True))
    elif straight: result = (4,straight)
    else:
        cat = { (3,1,1):3, (2,2,1):2, (2,1,1,1):1 }.get(tuple(n for n,r in groups),0)
        result = (cat,*(r for n,r in groups))
    return result + (0,)*(6-len(result))


def code_reference(cards):
    best = max(five_reference(c) for c in combinations(cards,5))
    return sum(value*15**(5-i) for i,value in enumerate(best))


def test_best_five_rank_matches_independent_exhaustive_oracle():
    rng = np.random.default_rng(191)
    for size in (5,6,7):
        cards = np.stack([rng.choice(52,size,replace=False) for _ in range(500)])
        codes, _, _ = rank_cards(cards)
        np.testing.assert_array_equal(codes,[code_reference(c) for c in cards])
    cases = [
        (48,49,50,44,45,46,0),  # two trips -> ace full of kings
        (48,49,44,45,40,41,0),  # three pairs -> top two, queen kicker
        (48,0,4,8,12,45,46),   # wheel straight flush
        (48,44,40,36,28,0,1),  # flush with six suited cards
        (0,1,2,3,48,49,50),    # quads override trips
    ]
    np.testing.assert_array_equal(rank_cards(cases)[0],[code_reference(c) for c in cases])


def test_rank_and_exact_river_strength_agree_with_open_spiel():
    rng = np.random.default_rng(349)
    game = load_hulhe_game()
    for _ in range(100):
        cards = rng.choice(52,9,replace=False).tolist()
        s = game.new_initial_state()
        deck = iter(cards)
        while not s.is_terminal(): s.apply_action(next(deck) if s.is_chance_node() else 1)
        ranks = rank_cards([cards[:2]+cards[4:],cards[2:4]+cards[4:]])[0]
        assert np.sign(s.returns()[0]) == np.sign(ranks[0]-ranks[1])
    hole, board = (0,5), (32,36,40,44,48)  # everyone plays the royal flush
    f = card_features(hole,board)
    assert f[CARD_INDEX['showdown_equity']] == .5
    assert f[CARD_INDEX['current_tie']] == 1
    assert f[CARD_INDEX['currently_unbeatable']] == 1
    assert f[CARD_INDEX['currently_strict_nuts']] == 0
    assert f[CARD_INDEX['plays_board']] == 1
    assert f[CARD_INDEX['showdown_exact']] == 1
    # Private royal flush cannot lose OR tie.
    f = card_features((48,44),(40,36,32,1,6))
    assert f[CARD_INDEX['currently_strict_nuts']] == 1
    assert f[CARD_INDEX['showdown_equity']] == 1


def test_draw_probability_and_nut_blocker_are_exact():
    # Ac Kc on 2c 7c 9d: nine remaining clubs complete a flush next card.
    f = card_features((48,44),(0,20,29))
    assert f[CARD_INDEX['next_card_flush']] == pytest.approx(9/47)
    assert f[CARD_INDEX['made_high_card']] == 1
    # Open-ended 6/7/8/9 completes on four fives or four tens.
    f = card_features((16,21),(26,31,1))
    assert f[CARD_INDEX['next_card_straight']] == pytest.approx(8/47)
    f = card_features((48,45),(0,20,28))
    assert f[CARD_INDEX['board_flush_suit_ace_blocker']] == 1


def test_sampler_is_legal_deterministic_and_does_not_touch_global_rng():
    py = random.getstate(); np_state = np.random.get_state(); th = torch.random.get_rng_state()
    for board in ((),(0,5,10),(0,5,10,15),(0,5,10,15,20)):
        hole = (48,49)
        f = card_features(hole,board)
        assert np.isfinite(f).all() and f.min() >= 0 and f.max() <= 1
        assert not f.flags.writeable
        assert sum(f[CARD_INDEX[n]] for n in ('showdown_win','showdown_tie','showdown_loss')) == pytest.approx(1)
        assert sum(f[CARD_INDEX['showdown_category_'+c]] for c in
                   ('high_card','pair','two_pair','trips','straight','flush','full_house','quads','straight_flush')) == pytest.approx(1)
        if len(board) in (3,4): assert f[CARD_INDEX['transition_loss_to_loss']:CARD_INDEX['transition_loss_to_loss']+9].sum() == pytest.approx(1)
        key = canonical_cards(hole,board)
        expected = f.copy()
        _card_features.cache_clear()
        np.testing.assert_array_equal(card_features(hole,board),expected)
        if len(board)<5:
            samples = _sample_completions(*key, count=500)
            assert not np.isin(samples,key[0]+key[1]).any()
            assert all(len(set(row)) == len(row) for row in samples)
    assert random.getstate() == py
    assert np_state[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(np_state[1],np.random.get_state()[1])
    assert np_state[2:] == np.random.get_state()[2:]
    assert torch.equal(th,torch.random.get_rng_state())


def test_all_suit_permutations_and_flop_order_keep_same_features():
    e=HULHEFeatureEncoder()
    s=state_at_street(3,holes=(48,45,2,7),board=(0,20,28,34,39))
    expected=e.information_state(s); critic=e.full_state(s)
    for perm in permutations(range(4)):
        t=load_hulhe_game().new_initial_state()
        for event in s.full_history():
            c=event.action
            t.apply_action(4*(c//4)+perm[c%4] if event.player == -1 else c)
        np.testing.assert_array_equal(e.information_state(t),expected)
        np.testing.assert_array_equal(e.full_state(t),critic)
    t=state_at_street(3,holes=(48,45,2,7),board=(28,0,20,34,39))
    np.testing.assert_array_equal(e.information_state(t),expected)
    # Critic's player-1 strategic block equals that player's isolated policy block.
    np.testing.assert_array_equal(critic[355:450],expected[208:303])


def test_exact_information_channels_are_preserved_and_history_matters():
    old,new=LegacyHULHEFeatureEncoder(),HULHEFeatureEncoder()
    for st in range(4):
        s=state_at_street(st)
        a,b=old.information_state(s),new.information_state(s)
        np.testing.assert_array_equal(a,np.concatenate((b[:208],b[303:377],b[-3:])))
        a,b=old.full_state(s),new.full_state(s)
        np.testing.assert_array_equal(a,np.concatenate((b[:260],b[454:528],b[-3:])))
    assert new.policy_size == 417 and new.full_state_size == 605


def test_economics_all_betting_paths_and_blind_option():
    for street in range(4):
        start=state_at_street(street)
        def walk(s):
            p=s.current_player()
            e=_public_economics(public_rounds(s))[p]
            spent=s.observation_tensor(p)[-2:]
            call=max(0,spent[1-p]-spent[p])
            assert e[BETTING_INDEX['pot_fraction']] == pytest.approx(sum(spent)/4800)
            assert e[BETTING_INDEX['pot_odds']] == pytest.approx(call/(sum(spent)+call))
            assert bool(e[BETTING_INDEX['raises_remaining_fraction']]) == (2 in s.legal_actions())
            closes=s.child(1).is_terminal() or s.child(1).is_chance_node()
            assert bool(e[BETTING_INDEX['call_closes_round']]) == closes
            assert bool(e[BETTING_INDEX['call_closes_hand']]) == s.child(1).is_terminal()
            for a in s.legal_actions():
                child=s.child(a)
                if not child.is_terminal() and not child.is_chance_node(): walk(child)
        walk(start)


def test_legacy_evaluation_and_versioned_feature_contract(tmp_path):
    from copy import deepcopy
    import pickle
    from hulhe_escher.features import encoder_from_metadata, StructuredHULHEMLP
    from hulhe_escher.checkpointing import LoadedHULHEPolicy, CHECKPOINT_TYPE, CHECKPOINT_VERSION
    from hulhe_escher.game import serialisable_game_definition
    from hulhe_escher.training_state import restore_training_state
    from hulhe_escher.config import make_solver, smoke_config
    old = LegacyHULHEFeatureEncoder()
    assert encoder_from_metadata(old.metadata()).metadata() == old.metadata()
    model = StructuredHULHEMLP(old.policy_layout,[16],3,branch_width=8)
    path = tmp_path/'old.pkl'
    path.write_bytes(pickle.dumps({'type':CHECKPOINT_TYPE,'version':CHECKPOINT_VERSION,
        'game':serialisable_game_definition(), 'feature_encoder':old.metadata(),
        'policy_model':model.checkpoint_metadata(), 'policy_state_dict':model.state_dict()}))
    loaded = LoadedHULHEPolicy(load_hulhe_game(),path)
    assert sum(loaded.action_probabilities(state_at_street()).values()) == pytest.approx(1)
    metadata = deepcopy(HULHEFeatureEncoder().metadata())
    metadata['strategic_cards']['postflop_samples'] += 1
    with pytest.raises(ValueError,match='encoder differs'): encoder_from_metadata(metadata)
    solver = make_solver(smoke_config())
    with pytest.raises(ValueError,match='encoder mismatch'):
        restore_training_state(solver,{'game':serialisable_game_definition(),'encoder':old.metadata()})
