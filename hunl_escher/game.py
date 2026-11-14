"""Versioned 200bb heads-up no-limit Hold'em training game."""
from types import MappingProxyType

import pyspiel

HUNL_GAME_NAME = "HUNL"
OPEN_SPIEL_GAME_NAME = "universal_poker"
# FCPA is an explicit initial action abstraction, not unrestricted bet sizing.
HUNL_GAME_PARAMETERS = MappingProxyType({
    "betting": "nolimit", "bettingAbstraction": "fcpa",
    "blind": "50 100", "stack": "20000 20000",
    "firstPlayer": "1 2 2 2", "numPlayers": 2, "numRounds": 4,
    "numSuits": 4, "numRanks": 13, "numHoleCards": 2,
    "numBoardCards": "0 3 1 1",
})


def load_hunl_game():
    return pyspiel.load_game(OPEN_SPIEL_GAME_NAME, dict(HUNL_GAME_PARAMETERS))


def serialisable_game_definition():
    return {
        "name": HUNL_GAME_NAME, "open_spiel_game": OPEN_SPIEL_GAME_NAME,
        "parameters": dict(HUNL_GAME_PARAMETERS),
        "action_abstraction": "fold, check/call, pot-sized raise, all-in",
        "stack_reset": "every hand", "rake": 0,
    }
