"""Board-cache equivalence, blocker semantics and frozen policy compatibility."""
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from hulhe_escher import card_features as cf
from hulhe_escher import legacy_card_features as old_cf
from hulhe_escher.features import HULHEFeatureEncoder, StrategicV3HULHEFeatureEncoder, StrategicV4HULHEFeatureEncoder, StructuredHULHEMLP
from test_hulhe import state_at_street


def test_board_cache_preserves_all_original_features_including_samples():
    rng = np.random.default_rng(20261115)
    for n in (0,3,4,5):
        for _ in range(20):
            ids=tuple(map(int,rng.choice(52,n+2,replace=False)))
            a,b=cf.card_features(ids[:2],ids[2:]),old_cf.card_features(ids[:2],ids[2:])
            np.testing.assert_array_equal(a[:len(old_cf.CARD_NAMES)],b)
    board = (0,5,10)
    cf._board_table.cache_clear(); cf._card_features.cache_clear()
    for hole in ((48,45),(40,37),(32,29)):
        cf.card_features(hole,board)
    assert cf._board_table.cache_info().misses == 1
    assert cf._board_table.cache_info().hits == 2
    assert cf._board_table.cache_info().maxsize == 64
    for array in cf._board_table(cf._board_key((48,45),board)[1]):
        assert not array.flags.writeable


def test_general_nut_flush_blocker_is_distinct_from_ace_blocker():
    # Kc on Ac 7c 2c; the ace is already public.
    f = dict(zip(cf.CARD_NAMES,cf.card_features((44,41),(48,20,0))))
    assert f['nut_flush_blocker'] == 1 and f['board_flush_suit_ace_blocker'] == 0
    f = dict(zip(cf.CARD_NAMES,cf.card_features((40,37),(48,20,0))))
    assert f['nut_flush_blocker'] == 0


@pytest.mark.parametrize("encoder_class", [StrategicV3HULHEFeatureEncoder, StrategicV4HULHEFeatureEncoder])
def test_v3_policy_compatibility_and_continuation_rejection(tmp_path, encoder_class):
    from hulhe_escher.checkpointing import LoadedHULHEPolicy, CHECKPOINT_TYPE, CHECKPOINT_VERSION
    from hulhe_escher.game import load_hulhe_game, serialisable_game_definition
    from hulhe_escher.training_state import restore_training_state
    from hulhe_escher.config import make_solver,smoke_config
    old = encoder_class()
    model = StructuredHULHEMLP(old.policy_layout,[16],3,branch_width=8)
    with torch.no_grad():
        for p in model.parameters(): p.uniform_(-.1,.1)
    path=tmp_path/'v3.pkl'
    path.write_bytes(pickle.dumps({'type':CHECKPOINT_TYPE,'version':CHECKPOINT_VERSION,
        'game':serialisable_game_definition(),'feature_encoder':old.metadata(),
        'policy_model':model.checkpoint_metadata(),'policy_state_dict':model.state_dict()}))
    loaded=LoadedHULHEPolicy(load_hulhe_game(),path)
    for street in range(4):
        state=state_at_street(street)
        with torch.no_grad():
            logits=model(torch.tensor(old.information_state(state)))
            expected=torch.softmax(logits.masked_fill(~torch.tensor(state.legal_actions_mask(),dtype=torch.bool),-1e21),-1)
        for a,p in loaded.action_probabilities(state).items(): assert p == expected[a].item()
    solver=make_solver(smoke_config())
    with pytest.raises(ValueError,match='encoder mismatch'):
        restore_training_state(solver,{'game':serialisable_game_definition(),'encoder':old.metadata()})


@pytest.mark.parametrize("fixture_name", ["holdem_card_contract_v1.json", "holdem_card_contract_v2.json"])
def test_shared_holdem_card_contract(fixture_name):
    fixture=json.loads((Path(__file__).parent/'fixtures'/fixture_name).read_text())
    for row in fixture['cases']:
        features=dict(zip(cf.CARD_NAMES,cf.card_features(tuple(row['hole']),tuple(row['board']))))
        for field,value in row['values'].items():
            assert features[fixture['mapping'][field]['limit']] == value
