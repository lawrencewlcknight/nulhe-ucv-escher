"""Inherited no-limit game for the HULHE preparation baseline.

Fixed-limit rules have not been implemented yet. Keep the source game intact
until the game, encoder and rules tests can be converted together."""
from types import MappingProxyType

import pyspiel

HULHE_GAME_NAME = "HULHE"
OPEN_SPIEL_GAME_NAME = "universal_poker"
# FCPA is an explicit initial action abstraction, not unrestricted bet sizing.
HULHE_GAME_PARAMETERS = MappingProxyType({
    "betting": "nolimit", "bettingAbstraction": "fcpa",
    "blind": "50 100", "stack": "20000 20000",
    "firstPlayer": "1 2 2 2", "numPlayers": 2, "numRounds": 4,
    "numSuits": 4, "numRanks": 13, "numHoleCards": 2,
    "numBoardCards": "0 3 1 1",
})


def load_hulhe_game():
    return pyspiel.load_game(OPEN_SPIEL_GAME_NAME, dict(HULHE_GAME_PARAMETERS))


def serialisable_game_definition():
    return {
        "implementation_status": "inherited_no_limit_base",
        "name": HULHE_GAME_NAME, "open_spiel_game": OPEN_SPIEL_GAME_NAME,
        "parameters": dict(HULHE_GAME_PARAMETERS),
        "action_abstraction": "fold, check/call, pot-sized raise, all-in",
        "stack_reset": "every hand", "rake": 0,
    }
