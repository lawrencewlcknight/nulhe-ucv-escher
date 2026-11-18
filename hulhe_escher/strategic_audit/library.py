"""Versioned, visible-information-only positions and exact OpenSpiel histories."""
from itertools import permutations
import hashlib
import json

from ..game import load_hulhe_game, serialisable_game_definition

VERSION = 'hulhe_strategic_audit_v1'
RANKS, SUITS = '23456789TJQKA', 'cdhs'


def cards(text):
    result = tuple(4 * RANKS.index(c[0]) + SUITS.index(c[1]) for c in text.split())
    if len(set(result)) != len(result):
        raise ValueError('Duplicate cards')
    return result


def card_text(ids):
    return ' '.join(RANKS[c // 4] + SUITS[c % 4] for c in ids)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def canonical_board(board):
    return min(tuple(sorted(4*(c//4) + p[c % 4] for c in board)) for p in permutations(range(4)))


def build_state(position, opponent=None):
    """Only deal the board visible at this decision; placeholders never determine EV."""
    hole, board = tuple(position['hole']), tuple(position['board'])
    hero = position['hero']
    if opponent is None:
        opponent = tuple(c for c in range(52) if c not in hole + board)[:2]
    if len(opponent) != 2 or len(set(hole + board + tuple(opponent))) != 4 + len(board):
        raise ValueError('Incompatible opponent hand')
    private = hole + tuple(opponent) if hero == 0 else tuple(opponent) + hole
    state = load_hulhe_game().new_initial_state()
    spent = [50, 100]
    for card in private:
        state.apply_action(card)
    board_offset = 0
    for street, actions in enumerate(position['actions']):
        if street:
            for _ in range(3 if street == 1 else 1):
                if not state.is_chance_node():
                    raise ValueError('Incomplete previous betting round')
                state.apply_action(board[board_offset]); board_offset += 1
        for action in actions:
            if state.is_chance_node() or state.is_terminal() or action not in state.legal_actions():
                raise ValueError('Illegal position history')
            actor = state.current_player()
            if action == 1:
                spent[actor] = max(spent)
            elif action == 2:
                spent[actor] = max(spent) + (100 if street < 2 else 200)
            state.apply_action(action)
    if (board_offset != len(board) or state.is_terminal() or state.is_chance_node()
            or state.current_player() != hero):
        raise ValueError('History does not end at specified hero decision')
    return state, spent


def replace_cards(state, *, opponent=None, suit_permutation=None, swap_hole=False):
    history = state.history()
    hero = state.current_player()
    if opponent is not None:
        start = 2*(1-hero)
        history[start:start+2] = list(opponent)
    if swap_hole:
        start = 2*hero
        history[start:start+2] = history[start:start+2][::-1]
    new = state.get_game().new_initial_state()
    for action in history:
        if new.is_chance_node() and suit_permutation is not None:
            action = 4*(action//4) + suit_permutation[action % 4]
        new.apply_action(action)
    return new


def validate_library(library):
    if library.get('version') != VERSION or library.get('game') != serialisable_game_definition():
        raise ValueError('Incompatible strategic library contract')
    positions = library['positions']
    if (not positions or any(not isinstance(p['id'],str) or not p['id'] for p in positions)
            or len({p['id'] for p in positions}) != len(positions)):
        raise ValueError('Position ids must be nonempty and unique')
    partitions = {}
    for p in positions:
        if p['partition'] not in ('development', 'assessment') or p['hero'] not in (0, 1):
            raise ValueError('Invalid partition or seat')
        hole, board = tuple(p['hole']), tuple(p['board'])
        if (len(hole) != 2 or len(board) not in (0, 3, 4, 5)
                or len(set(hole + board)) != len(hole + board)
                or any(type(c) is not int or not 0 <= c < 52 for c in hole + board)):
            raise ValueError('Invalid visible cards')
        if len(p['actions']) != {0:1, 3:2, 4:3, 5:4}[len(board)]:
            raise ValueError('Street/history mismatch')
        if any(type(a) is not int or a not in (0, 1, 2) for s in p['actions'] for a in s):
            raise ValueError('Invalid action')
        # Entire public-board suit orbit stays in one partition, across hero hands/history.
        # Preflop uses the private starting-hand orbit (one empty public board).
        key = ('board', canonical_board(board)) if board else ('preflop', canonical_board(hole))
        if key in partitions and partitions[key] != p['partition']:
            raise ValueError('Development/assessment card-orbit leakage')
        partitions[key] = p['partition']
        build_state(p)
    return library


def build_library():
    # Explicit texture/hand cases, not a distribution of natural game visitation.
    boards = [
        ('private_nuts', 'Qc Jc Tc 3d 7h', ['Ac Kc', 'Ah Qd'], 'development'),
        ('board_nuts', 'Tc Jc Qc Kc Ac', ['2d 3h', '9d 9h'], 'development'),
        ('dry_top_pair', 'Ks 7d 2c 4h 9c', ['Ac Kh', 'Kc 5d', 'Qh Jd'], 'development'),
        ('missed_draws', 'Jh 8h 3c 2s 6d', ['Ah Qh', 'Tc 9d', 'Js Jd'], 'development'),
        ('four_flush', 'Ah 9h 6h 2h Kc', ['Kh Qd', '3h Ad', 'Ac As'], 'development'),
        ('paired_board', 'Qs Qd 8c 4h 2s', ['Qh Ac', '8d 8h', 'Ac Kd'], 'development'),
        ('four_straight', '5c 6d 7h 8s Kc', ['9d Td', '4h As', 'Ac Ad'], 'development'),
        ('quads_board', '9c 9d 9h 9s 2c', ['Ac Kd', '3d 4h'], 'development'),
        ('low_full_house', '2c 2d 7h 7s Kc', ['2h Ad', '7c Qd', 'Ac Ah'], 'development'),
        ('high_card', 'Ah Kd 9c 5s 2h', ['Qh Jd', '9h 8h', 'Ac Ad'], 'development'),
        ('board_nuts_low_cards', 'Th Jh Qh Kh Ah', ['4d 5s'], 'development'),
        ('assessment_dry', 'Qh 6c 3d 8s Tc', ['Ac Qd', 'Qs 4h', 'Jc 9h'], 'assessment'),
        ('assessment_monotone', 'Ks Ts 7s 4d 2h', ['As Qs', 'Js 9s', 'Kc Kd'], 'assessment'),
        ('assessment_paired', 'Jc Jd 5h 8c As', ['Jh Kd', '5c 5d', 'Ac Qh'], 'assessment'),
        ('assessment_connected', '4h 5s 6c 7d Qh', ['8c 9d', '3c Ac', 'Qs Qd'], 'assessment'),
        ('assessment_quads', '6c 6d 6h 6s 3c', ['Ac Qd', '4d 5h'], 'assessment'),
    ]
    positions = []
    for family, board, holes, partition in boards:
        for h, hole in enumerate(holes):
            for hero in (0, 1):
                # Natural open/facing-bet decisions plus a capped call/fold stress case.
                for mode in ('open', 'facing_bet', 'capped'):
                    river = ([] if hero == 1 else [1]) if mode == 'open' else (
                        [2,2,2,2] if hero == 1 else [1,2,2,2,2])
                    if mode == 'facing_bet':
                        river = [1,2] if hero == 1 else [2]
                    prior = [[2,1],[1,1],[1,1]] if h % 2 == 0 else [[2,2,2,1],[2,1],[2,1]]
                    positions.append(dict(id=f'{family}-{h}-p{hero}-{mode}', partition=partition,
                        hero=hero, hole=list(cards(hole)), board=list(cards(board)),
                        actions=prior+[river], tags=[family, mode], family=family))
    # Earlier streets deliberately have no preferred-action answer or fictitious exact EV.
    contexts = [
        ('preflop_aces', 'Ac Ad', '', 'development'),
        ('preflop_weak', '7c 2d', '', 'development'),
        ('preflop_suited', 'Js Ts', '', 'assessment'),
        ('flop_nut_draw', 'Ah Kh', 'Qh 8h 2c', 'development'),
        ('flop_set', '7c 7d', 'As 7h 2c', 'development'),
        ('flop_wet_pair', 'Ac Kd', 'Kh Qh Jh', 'assessment'),
        ('turn_missed_draw', 'Ah Kh', 'Qh 8h 2c 5s', 'development'),
        ('turn_made_flush', 'Ah Kh', 'Qh 8h 2c 3h', 'development'),
        ('turn_paired', 'Jc Jd', 'Jh 8s 4c 8d', 'assessment'),
    ]
    for family, hole, board, partition in contexts:
        street = {0:0,3:1,4:2}[len(cards(board))]
        for hero in (0, 1):
            final = ([] if hero == 0 else [2]) if street == 0 else ([] if hero == 1 else [2])
            positions.append(dict(id=f'{family}-p{hero}', partition=partition, hero=hero,
                hole=list(cards(hole)), board=list(cards(board)),
                actions=[[1,1] for _ in range(street)] + [final], tags=[family, 'context_only'], family=family))
    return validate_library(dict(version=VERSION, game=serialisable_game_definition(), positions=positions,
        weighting='Equal root positions within each panel; overlapping tags are not additive',
        partition_note='Assessment is held out from audit-driven selection, not necessarily from training'))
