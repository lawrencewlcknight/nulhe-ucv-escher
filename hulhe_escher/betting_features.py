"""Exact fixed-limit economics from public actions; all channels in [0,1]."""
from functools import lru_cache
import math

import numpy as np

from .card_features import CARD_INDEX

BETTING_NAMES = (
    'pot_fraction', 'log_pot', 'call_small_bets_fraction', 'call_current_bets',
    'pot_odds', 'current_bet_size', 'own_total_fraction', 'opponent_total_fraction',
    'own_round_fraction', 'opponent_round_fraction', 'raises_remaining_fraction',
    'button_small_blind', 'acts_last', 'next_to_act', 'facing_bet',
    'last_action_raise', 'last_action_opponent_check', 'call_closes_round',
    'call_closes_hand',
    *(f'street_{s}_raises' for s in range(4)),
    *(f'street_{s}_aggressor_{who}' for s in range(4) for who in ('self','opponent')),
    'equity_minus_pot_odds_shifted', 'current_strength_minus_pot_odds_shifted',
    'equity_times_call_closes_hand', 'equity_times_raise_available',
    'equity_times_position', 'uniform_showdown_call_value_shifted',
)
BETTING_INDEX = {name: i for i,name in enumerate(BETTING_NAMES)}


def public_rounds(state):
    rounds = [[], [], [], []]
    dealt = street = 0
    for event in state.full_history():
        if event.player == -1:
            dealt += 1
            if dealt > 4: street = min(3, max(1, dealt - 6))
        else:
            if event.action not in (1,2):
                raise ValueError('Decision history cannot contain a fold')
            rounds[street].append((event.player, event.action))
    return tuple(tuple(r) for r in rounds[:street+1])


@lru_cache(maxsize=4096)
def _public_economics(rounds):
    spent = [50,100]
    counts, aggressors = [0]*4, [-1]*4
    for street, actions in enumerate(rounds):
        start = [0,0] if street == 0 else spent.copy()
        actor = 0 if street == 0 else 1
        last_actor, last_raise, last_check = -1, False, False
        bet = 100 if street < 2 else 200
        for who, action in actions:
            if who != actor:
                raise ValueError('Invalid public action order')
            cost = max(spent) - spent[actor]
            last_actor, last_raise, last_check = actor, action == 2, action == 1 and cost == 0
            if action == 2:
                counts[street] += 1
                aggressors[street] = actor
                spent[actor] = max(spent) + bet
            else:
                spent[actor] += cost
            actor = 1-actor
    remaining = (3 if street == 0 else 4) - counts[street]
    if remaining < 0: raise ValueError('Raise cap exceeded')
    result = np.zeros((2,len(BETTING_NAMES)), dtype=np.float32)
    pot = sum(spent)
    for p in (0,1):
        call = max(0,spent[1-p]-spent[p])
        # Any check/call after the first voluntary action closes a heads-up
        # round, including the BB check after an SB blind completion.
        closes = p == actor and bool(rounds[-1])
        result[p,:19] = (pot/4800, math.log1p(pot/100)/math.log1p(48), call/200,
            call/bet, call/(pot+call), bet/200, spent[p]/2400, spent[1-p]/2400,
            (spent[p]-start[p])/800, (spent[1-p]-start[1-p])/800, remaining/4,
            p == 0, p == (1 if street == 0 else 0), p == actor, call > 0,
            last_raise, last_actor == 1-p and last_check, closes, closes and street == 3)
        result[p,19:23] = np.asarray(counts)/4
        result[p,23:31] = [a == who for a in aggressors for who in (p,1-p)]
    result.flags.writeable = False
    return result


def betting_features(rounds, player, cards, *, actor):
    result = _public_economics(rounds)[player].copy()
    equity = float(cards[CARD_INDEX['showdown_equity']])
    current = float(cards[CARD_INDEX['current_strength']])
    odds = result[BETTING_INDEX['pot_odds']]
    raises = result[BETTING_INDEX['raises_remaining_fraction']] > 0
    pot = result[BETTING_INDEX['pot_fraction']]*4800
    call = result[BETTING_INDEX['call_small_bets_fraction']]*200
    result[31:] = ((1+equity-odds)/2,
                  (1+current-odds)/2 if cards[CARD_INDEX['made_hand_available']] else 0,
                  equity*result[BETTING_INDEX['call_closes_hand']],
                  equity*raises*(actor == player), equity*result[BETTING_INDEX['acts_last']],
                  (1+(equity*(pot+call)-call)/4800)/2)
    return result
