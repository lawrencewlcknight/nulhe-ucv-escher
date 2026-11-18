"""Visible-state serialization and coarse, explicitly non-equivalent replay families."""
from collections import Counter
from functools import lru_cache
import time

import numpy as np

from ..card_features import CARD_INDEX, CATEGORIES
from ..strategic_audit.library import canonical_board, digest, build_library

STREETS = ('preflop', 'flop', 'turn', 'river')


def deadline_after(seconds):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive finite time budget required')
    end = time.monotonic() + seconds
    def check():
        if time.monotonic() >= end:
            raise TimeoutError('Diagnostic work budget exhausted')
    return check


def visible_position(state):
    if state.is_chance_node() or state.is_terminal():
        raise ValueError('Expected a player decision')
    replay = state.get_game().new_initial_state()
    private, board, actions = [], [], [[]]
    for action in state.history():
        if replay.is_chance_node():
            if len(private) < 4:
                private.append(int(action))
            else:
                board.append(int(action))
                street = 1 if len(board) <= 3 else len(board)-2
                while len(actions) <= street:
                    actions.append([])
        else:
            actions[-1].append(int(action))
        replay.apply_action(action)
    hero = state.current_player()
    return dict(hero=hero, hole=private[2*hero:2*hero+2], board=board, actions=actions)


def orbit(position):
    # Conservatively keep a whole flop orbit together, including its turn/river
    # extensions. The existing suite's full-board assessment orbits are protected too.
    board = position['board']
    return ('flop' if board else 'preflop', canonical_board(board[:3] if board else position['hole']))


@lru_cache(maxsize=1)
def protected_orbits():
    return {orbit(p) for p in build_library()['positions'] if p['partition']=='assessment'}


def partition(position):
    if orbit(position) in protected_orbits():
        return 'assessment'
    return 'assessment' if int(digest(orbit(position))[:8], 16) % 4 == 0 else 'development'


def lock_partitions(positions):
    """Also connect identical full-board orbits reached via different flops.

    If a connected component contains an assessment position, protect it all.
    Membership is frozen using visible cards only, never scores.
    """
    groups={}
    for i,p in enumerate(positions):
        p['partition']=partition(p)
        keys=[orbit(p)]
        if p['board']: keys.append(('full',canonical_board(p['board'])))
        for key in keys: groups.setdefault(key,[]).append(i)
    changed=True
    while changed:
        changed=False
        for ids in groups.values():
            if any(positions[i]['partition']=='assessment' for i in ids):
                for i in ids:
                    if positions[i]['partition']!='assessment':
                        positions[i]['partition']='assessment';changed=True
    return positions


def families(features, encoder):
    """Same function for encounters and replay; no nearest-neighbour target imputation.

    Family = street/seat/made category/facing a bet/raise availability. Exact
    cards, histories and ranges can differ substantially within one family.
    """
    x = np.atleast_2d(features)
    player, street = encoder.replay_context_columns()
    seats, streets = x[:, player].argmax(1), x[:, street].argmax(1)
    offset = encoder.policy_layout.card_size
    if encoder.policy_layout.strategic_size < len(CARD_INDEX):
        raise ValueError('Position investigation requires strategic card features')
    made = x[:, offset+CARD_INDEX['made_high_card']:offset+CARD_INDEX['made_straight_flush']+1].argmax(1)
    pocket = x[:, offset+CARD_INDEX['pocket_pair']] > .5
    return [f'{STREETS[s]}/p{p}/'+(('pocket_pair' if pair else 'unpaired') if s == 0 else CATEGORIES[c])+
            f'/facing{int(fold > .5)}/raise{int(raise_ > .5)}'
            for s,p,c,pair,fold,raise_ in zip(streets,seats,made,pocket,x[:,-3],x[:,-1])]


def priority(seed, source, hand, decision):
    return digest([seed, source, hand, decision])


def count_rows(rows, key):
    return dict(Counter(row[key] for row in rows))
