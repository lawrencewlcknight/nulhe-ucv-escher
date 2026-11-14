"""OpenSpiel heads-up fixed-limit Hold'em contract for Fell Omen 2 play.

Player 0 is the small blind/button, first preflop and last postflop. The big
blind counts toward the four-bet preflop cap. See docs/GAME_INTERFACE.md for
source evidence, chip units and the separate opponent-adapter boundary.
"""
from types import MappingProxyType

import pyspiel

HULHE_GAME_NAME = "HULHE"
GAME_CONTRACT_ID = "hulhe_fell_omen2_limit_v1"
OPEN_SPIEL_GAME_NAME = "universal_poker"
FOLD, CHECK_CALL, BET_RAISE = range(3)
NUM_ACTIONS = 3
SMALL_BET_CHIPS = 100
BIG_BET_CHIPS = 200
MAX_COMMITMENT_CHIPS = 2400
# In OpenSpiel 1.6.3 the fcpa dispatch uses the complete three-action set for
# limit games. Pot-sized raises/all-ins exist only in its no-limit branch.
# Do not use fullgame: that dispatch is for chip-sized no-limit actions.
HULHE_GAME_PARAMETERS = MappingProxyType({
    "betting": "limit", "bettingAbstraction": "fcpa",
    "blind": "50 100", "raiseSize": "100 100 200 200",
    "maxRaises": "3 4 4 4",
    "firstPlayer": "1 2 2 2", "numPlayers": 2, "numRounds": 4,
    "numSuits": 4, "numRanks": 13, "numHoleCards": 2,
    "numBoardCards": "0 3 1 1",
})


def load_hulhe_game():
    return pyspiel.load_game(OPEN_SPIEL_GAME_NAME, dict(HULHE_GAME_PARAMETERS))


def validate_hulhe_game(game):
    """Reject different rules even when their tensor dimensions happen to match."""
    if (game.get_type().short_name != OPEN_SPIEL_GAME_NAME
            or game.get_parameters() != load_hulhe_game().get_parameters()):
        raise ValueError("Game does not match the Fell Omen 2 HULHE contract")


def serialisable_game_definition():
    return {
        "contract_id": GAME_CONTRACT_ID,
        "benchmark_opponent": "Fell Omen 2",
        "name": HULHE_GAME_NAME, "open_spiel_game": OPEN_SPIEL_GAME_NAME,
        "parameters": dict(HULHE_GAME_PARAMETERS),
        "actions": {"0": "fold", "1": "check/call", "2": "bet/raise"},
        "action_abstraction": "none; complete fixed-limit action set",
        "small_blind_player": 0, "big_blind_player": 1,
        "bets_per_round": [4, 4, 4, 4],
        "preflop_big_blind_counts_as_bet": True,
        "stack_model": "nonbinding bankroll; independent hands",
        "payoff_unit": "chips", "small_bet_chips": SMALL_BET_CHIPS,
        "big_bet_chips": BIG_BET_CHIPS,
        "max_commitment_chips": MAX_COMMITMENT_CHIPS, "rake": 0,
    }
