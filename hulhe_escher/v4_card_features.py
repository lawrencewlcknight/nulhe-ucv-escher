"""Own-plus-public Hold'em descriptors. No opponent cards or training RNG access.

Current strength and next-card category distributions are exact. Future showdown
statistics use a fixed, versioned local sampling rule, not a learned opponent range.
"""
from functools import lru_cache
from hashlib import blake2b
from itertools import combinations

import numpy as np

CATEGORIES = ('high_card', 'pair', 'two_pair', 'trips', 'straight', 'flush',
              'full_house', 'quads', 'straight_flush')
CARD_NAMES = (
    'hole_high_rank', 'hole_low_rank', 'hole_gap', 'pocket_pair', 'suited',
    'connected', 'hole_broadway_fraction', 'hole_ace',
    'made_hand_available', *(f'made_{c}' for c in CATEGORIES),
    *(f'tiebreak_{i}' for i in range(5)),
    *(f'board_rank_{i}' for i in range(5)),
    'board_pair_count', 'board_trips', 'board_quads',
    *(f'board_suit_count_{i}' for i in range(4)),
    'board_straight_window', 'combined_straight_window',
    'combined_max_suit', 'overpair', 'underpair', 'top_rank_match',
    'overcard_fraction', 'board_matching_fraction', 'pocket_set_or_better',
    'board_flush_suit_hole_high', 'board_flush_suit_hole_count',
    'board_flush_suit_ace_blocker', 'plays_board', 'future_cards_fraction',
    'current_win', 'current_tie', 'current_loss', 'current_strength',
    'currently_unbeatable', 'currently_strict_nuts',
    'next_card_available', *(f'next_card_{c}' for c in CATEGORIES),
    'showdown_win', 'showdown_tie', 'showdown_loss', 'showdown_equity',
    'equity_sampling_se', 'showdown_exact',
    *(f'showdown_category_{c}' for c in CATEGORIES),
    'transition_available',
    *(f'transition_{a}_to_{b}' for a in ('loss', 'tie', 'win')
      for b in ('loss', 'tie', 'win')),
    'nut_flush_blocker',
)
CARD_INDEX = {name: i for i, name in enumerate(CARD_NAMES)}
PREFLOP_SAMPLES = 4096
POSTFLOP_SAMPLES = 512
BOARD_CACHE_SIZE = 64
SAMPLING_ID = 'canonical_visible_cards_blake2b_pcg64_partial_fisher_yates_v1'
_PAIRS = np.asarray(list(combinations(range(52), 2)), dtype=np.int16)
_POWERS = np.asarray([15**i for i in range(4, -1, -1)], dtype=np.int32)
_RANKS = np.arange(2, 15)
_BITS = 1 << np.arange(13)
_STRAIGHT = np.zeros(8192, dtype=np.int8)
for _m in range(8192):
    if _m & 0x100F == 0x100F:
        _STRAIGHT[_m] = 5
    for _s in range(9):
        if _m & (31 << _s) == 31 << _s:
            _STRAIGHT[_m] = _s + 6
_WINDOWS = tuple(tuple(range(s, s + 5)) for s in range(9)) + ((12, 0, 1, 2, 3),)


def rank_cards(cards):
    """Vectorized exact best-five ranks for N x {5,6,7} cards (larger wins).

    Internal callers supply distinct, valid card ids. The explicit category and
    five tiebreak ranks are also useful features. Two trips, three pairs, wheel,
    board-only hands, and flush kickers are handled without five-card enumeration.
    """
    cards = np.asarray(cards, dtype=np.int16)
    if cards.ndim != 2 or cards.shape[1] not in (5, 6, 7):
        raise ValueError('Expected N by 5, 6 or 7 cards')
    ranks, suits = cards // 4, cards % 4
    counts = np.zeros((len(cards), 13), dtype=np.int8)
    rows = np.arange(len(cards))
    for col in range(cards.shape[1]):
        counts[rows, ranks[:, col]] += 1
    def top(mask, k):
        return np.sort(np.where(mask, _RANKS, 0), axis=1)[:, ::-1][:, :k]
    tie = top(counts > 0, 5).copy()
    category = np.zeros(len(cards), dtype=np.int8)
    pair = top(counts >= 2, 2)
    trip = top(counts >= 3, 1)[:, 0]
    quad = top(counts == 4, 1)[:, 0]
    def set_hand(mask, cat, values):
        category[mask] = cat
        tie[mask] = 0
        tie[mask, :values.shape[1]] = values[mask]
    p = pair[:, 0]
    set_hand(p > 0, 1, np.column_stack((p, top((counts > 0) & (_RANKS != p[:, None]), 3))))
    kicker = top((counts > 0) & (_RANKS != pair[:, :1]) & (_RANKS != pair[:, 1:2]), 1)
    set_hand(pair[:, 1] > 0, 2, np.column_stack((pair, kicker)))
    set_hand(trip > 0, 3, np.column_stack((trip, top((counts > 0) & (_RANKS != trip[:, None]), 2))))
    straight = _STRAIGHT[((counts > 0) * _BITS).sum(axis=1)]
    set_hand(straight > 0, 4, straight[:, None])
    straight_flush = np.zeros(len(cards), dtype=np.int8)
    for suit in range(4):
        present = np.zeros_like(counts, dtype=bool)
        for col in range(cards.shape[1]):
            present[rows, ranks[:, col]] |= suits[:, col] == suit
        set_hand(present.sum(axis=1) >= 5, 5, top(present, 5))
        straight_flush = np.maximum(straight_flush, _STRAIGHT[(present * _BITS).sum(axis=1)])
    other_pair = top((counts >= 2) & (_RANKS != trip[:, None]), 1)[:, 0]
    set_hand((trip > 0) & (other_pair > 0), 6, np.column_stack((trip, other_pair)))
    set_hand(quad > 0, 7, np.column_stack((quad, top((counts > 0) & (_RANKS != quad[:, None]), 1))))
    set_hand(straight_flush > 0, 8, straight_flush[:, None])
    return category.astype(np.int32) * 15**5 + tie @ _POWERS, category, tie


def canonical_cards(hole, board):
    """Canonical key depends solely on own cards and the currently visible board."""
    hole, board = tuple(sorted(hole)), tuple(sorted(board))
    if (len(hole) != 2 or len(board) not in (0, 3, 4, 5)
            or len(set(hole + board)) != len(hole + board)
            or any(c < 0 or c >= 52 for c in hole + board)):
        raise ValueError('Invalid visible cards')
    signatures = [(tuple(int(4*r+s in hole) for r in range(13)),
                   tuple(int(4*r+s in board) for r in range(13))) for s in range(4)]
    order = sorted(range(4), key=lambda s: signatures[s], reverse=True)
    mapping = [order.index(s) for s in range(4)]
    convert = lambda cards: tuple(sorted(4*(c//4) + mapping[c % 4] for c in cards))
    return convert(hole), convert(board)


def _sample_completions(hole, board, count):
    key = bytes(hole + board) + SAMPLING_ID.encode()
    seed = int.from_bytes(blake2b(key, digest_size=16).digest(), 'little')
    rng = np.random.Generator(np.random.PCG64(seed))
    available = np.asarray([c for c in range(52) if c not in hole + board], dtype=np.int16)
    decks = np.tile(available, (count, 1))
    rows = np.arange(count)
    needed = 7 - len(board)  # Two opponent cards and the remaining public cards.
    for i in range(needed):
        j = rng.integers(i, len(available), size=count)
        decks[rows, i], decks[rows, j] = decks[rows, j].copy(), decks[rows, i].copy()
    return decks[:, :needed]


def card_features(hole, board):
    return _card_features(*canonical_cards(hole, board))


def _board_key(hole, board):
    # Board-first cache ordering is deliberately separate from the original
    # own-first sampling key: existing sampled features must remain identical.
    signatures = [(sum(1 << (c//4) for c in board if c % 4 == s),
                   sum(1 << (c//4) for c in hole if c % 4 == s)) for s in range(4)]
    order = sorted(range(4), key=lambda s: signatures[s], reverse=True)
    return tuple(tuple(sorted(4*(c//4) + order.index(c % 4) for c in group))
                 for group in (hole, board))


@lru_cache(maxsize=BOARD_CACHE_SIZE)
def _board_table(board):
    hands = _PAIRS[~np.isin(_PAIRS, board).any(axis=1)]
    ranks, categories, ties = rank_cards(np.column_stack((hands, np.tile(board, (len(hands), 1)))))
    lookup = np.full((52,52), -1, dtype=np.int16)
    lookup[hands[:,0], hands[:,1]] = np.arange(len(hands))
    for value in (hands, ranks, categories, ties, lookup):
        value.flags.writeable = False
    return hands, ranks, categories, ties, lookup


@lru_cache(maxsize=8192)
def _card_features(hole, board):
    out = np.zeros(len(CARD_NAMES), dtype=np.float32)
    def put(name, value): out[CARD_INDEX[name]] = value
    hr = np.asarray(hole) // 4
    br = np.asarray(board, dtype=int) // 4
    hc = np.bincount(hr, minlength=13)
    bc = np.bincount(br, minlength=13)
    all_cards = hole + board
    counts = hc + bc
    high, low = max(hr), min(hr)
    for name, value in zip(CARD_NAMES[:8], ((high+2)/14, (low+2)/14, (high-low)/12,
            high == low, hole[0] % 4 == hole[1] % 4,
            high-low == 1 or (high == 12 and low == 0), np.mean(hr >= 8), high == 12)):
        put(name, value)
    put('future_cards_fraction', (5-len(board))/5)
    current = None
    if board:
        table_hole, table_board = _board_key(hole, board)
        hands, ranks, categories, ties, lookup = _board_table(table_board)
        index = int(lookup[table_hole])
        current, cat, tie = ranks[index:index+1], categories[index:index+1], ties[index:index+1]
        put('made_hand_available', 1)
        put('made_' + CATEGORIES[cat[0]], 1)
        for i, value in enumerate(tie[0]): put(f'tiebreak_{i}', value/14)
        for i, rank in enumerate(sorted(br, reverse=True)): put(f'board_rank_{i}', (rank+2)/14)
        put('board_pair_count', np.count_nonzero(bc >= 2)/2)
        put('board_trips', np.any(bc == 3))
        put('board_quads', np.any(bc == 4))
        suits = np.bincount(np.asarray(board) % 4, minlength=4)
        for i, value in enumerate(sorted(suits, reverse=True)): put(f'board_suit_count_{i}', value/5)
        put('board_straight_window', max(np.count_nonzero(bc[list(w)]) for w in _WINDOWS)/5)
        put('combined_straight_window', max(np.count_nonzero(counts[list(w)]) for w in _WINDOWS)/5)
        put('combined_max_suit', np.bincount(np.asarray(all_cards) % 4, minlength=4).max()/7)
        put('overpair', low == high and high > max(br))
        put('underpair', low == high and high < min(br))
        put('top_rank_match', hc[max(br)] > 0)
        put('overcard_fraction', np.mean(hr > max(br)))
        put('board_matching_fraction', np.mean(bc[hr] > 0))
        put('pocket_set_or_better', low == high and bc[high] > 0)
        # Unique threatening suit; no arbitrary suit tie-break on rainbow boards.
        if suits.max() >= 3:
            suit = int(suits.argmax())
            suited_hole = [c//4+2 for c in hole if c % 4 == suit]
            put('board_flush_suit_hole_high', max(suited_hole, default=0)/14)
            put('board_flush_suit_hole_count', len(suited_hole)/2)
            put('board_flush_suit_ace_blocker', 14 in suited_hole)
            highest_remaining = max(r for r in range(2,15) if 4*(r-2)+suit not in board)
            put('nut_flush_blocker', highest_remaining in suited_hole)
        if len(board) == 5:
            put('plays_board', current[0] == rank_cards([board])[0][0])
        other = ranks[~np.isin(hands, table_hole).any(axis=1)]
        win, tie_p, loss = np.mean(current > other), np.mean(current == other), np.mean(current < other)
        for name, value in zip(('current_win','current_tie','current_loss','current_strength',
                               'currently_unbeatable','currently_strict_nuts'),
                              (win,tie_p,loss,win+.5*tie_p,loss == 0, win == 1)):
            put(name, value)
        if len(board) < 5:
            next_cards = [c for c in range(52) if c not in all_cards]
            next_cat = rank_cards(np.column_stack((np.tile(all_cards, (len(next_cards),1)), next_cards)))[1]
            put('next_card_available', 1)
            for i, mass in enumerate(np.bincount(next_cat, minlength=9)/len(next_cards)):
                put('next_card_' + CATEGORIES[i], mass)
    if len(board) == 5:
        for name, value in zip(('showdown_win','showdown_tie','showdown_loss','showdown_equity'),
                              (win,tie_p,loss,win+.5*tie_p)): put(name,value)
        put('showdown_exact', 1)
        put('showdown_category_' + CATEGORIES[cat[0]], 1)
    else:
        n = POSTFLOP_SAMPLES if board else PREFLOP_SAMPLES
        samples = _sample_completions(hole, board, n)
        final_board = np.column_stack((np.tile(board, (n,1)), samples[:, 2:]))
        own_final, final_cat, _ = rank_cards(np.column_stack((np.tile(hole, (n,1)), final_board)))
        other_final = rank_cards(np.column_stack((samples[:, :2], final_board)))[0]
        result = np.sign(own_final - other_final)
        pay = (result + 1)/2
        for name, value in zip(('showdown_win','showdown_tie','showdown_loss','showdown_equity',
                               'equity_sampling_se'),
                              (np.mean(result > 0), np.mean(result == 0), np.mean(result < 0),
                               pay.mean(), pay.std(ddof=1)/np.sqrt(n))): put(name,value)
        for i, mass in enumerate(np.bincount(final_cat, minlength=9)/n):
            put('showdown_category_' + CATEGORIES[i], mass)
        if board:
            other_now = rank_cards(np.column_stack((samples[:, :2], np.tile(board, (n,1)))))[0]
            now = np.sign(current[0] - other_now)
            transition = np.bincount((now+1)*3 + result+1, minlength=9)/n
            put('transition_available', 1)
            start = CARD_INDEX['transition_loss_to_loss']
            out[start:start+9] = transition
    out.flags.writeable = False
    return out


def feature_spec():
    return {'names': list(CARD_NAMES), 'opponent_prior': 'uniform compatible holdings; never action-conditioned',
            'current_strength': 'exact enumeration; not final equity before river',
            'next_card_categories': 'exact marginal enumeration over visible-card-compatible next cards',
            'showdown': 'exact river; deterministic sampled future elsewhere',
            'preflop_samples': PREFLOP_SAMPLES, 'postflop_samples': POSTFLOP_SAMPLES,
            'sampling_id': SAMPLING_ID, 'card_cache_entries': 8192,
            'board_cache_entries': BOARD_CACHE_SIZE,
            'nut_flush_blocker': 'highest suited rank absent from board; ace blocker remains separate',
            'transition': 'joint current-to-showdown loss/tie/win mass; flop/turn only'}


@lru_cache(maxsize=8192)
def critic_comparison(hole0, hole1, board):
    """Privileged current comparison for the control variate only, never policy."""
    result = np.zeros(4, dtype=np.float32)
    if board:
        ranks = rank_cards([hole0+board, hole1+board])[0]
        result[:] = (1, ranks[0]>ranks[1], ranks[0]==ranks[1], ranks[0]<ranks[1])
    result.flags.writeable = False
    return result
