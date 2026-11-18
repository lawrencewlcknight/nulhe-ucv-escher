"""Exact river hidden-hand integration and information-set-respecting response.

Opponent ranges are explicitly stipulated at the root, NOT inferred from earlier
betting. Their subsequent action likelihoods update the range inside the tree.
Never maximise separately for each opponent hand: that would be clairvoyant.
"""
from itertools import combinations, permutations
from functools import lru_cache

import numpy as np

from ..card_features import rank_cards
from .library import build_state, replace_cards

PANELS = ('check_call', 'value', 'pressure')
RANGES = ('uniform', 'strong', 'weak')


def probabilities(policy, state):
    raw = policy.action_probabilities(state)
    legal = state.legal_actions()
    if not isinstance(raw, dict) or any(a not in (0,1,2) for a in raw):
        raise ValueError('Policy must return action-probability mapping')
    p = np.asarray([raw.get(a, 0.) for a in range(3)], dtype=np.float64)
    if (not np.isfinite(p).all() or (p < 0).any() or
            any(p[a] != 0 for a in range(3) if a not in legal) or abs(p.sum()-1) > 1e-6):
        raise ValueError('Invalid policy probabilities; no action fallback permitted')
    return p / p.sum()


class CachedPolicy:
    """Cache public decision paths only within one fixed hero hand/board."""
    def __init__(self, policy, deadline=lambda: None):
        self.policy, self.deadline, self.cache = policy, deadline, {}

    def action_probabilities(self, state):
        self.deadline()
        key = tuple(state.history())
        if key not in self.cache:
            p = probabilities(self.policy, state)
            self.cache[key] = {a:float(p[a]) for a in state.legal_actions()}
        return self.cache[key]


def consistency_checks(policy, position, deadline=lambda: None):
    state, _ = build_state(position)
    expected = probabilities(policy, state)
    unseen = [c for c in range(52) if c not in position['hole'] + position['board']]
    errors = []
    for other in (unseen[-2:], unseen[len(unseen)//2:len(unseen)//2+2]):
        deadline()
        errors.append(float(np.max(np.abs(probabilities(policy, replace_cards(state, opponent=other))-expected))))
    suit_error = 0.
    for perm in permutations(range(4)):
        deadline()
        suit_error = max(suit_error, float(np.max(np.abs(
            probabilities(policy, replace_cards(state, suit_permutation=perm))-expected))))
    hole_error = float(np.max(np.abs(probabilities(policy, replace_cards(state, swap_hole=True))-expected)))
    repeat_error = float(np.max(np.abs(probabilities(policy, state)-expected)))
    if max(errors + [hole_error, repeat_error]) > 1e-6:
        raise ValueError('Policy violates hidden-card/privacy, hole-order or deterministic-distribution contract')
    return dict(hidden_card_max_error=max(errors), hole_order_max_error=hole_error,
        repeated_query_max_error=repeat_error, suit_max_error=suit_error,
        suit_permutations=24, suit_invariant=suit_error <= 1e-6)


@lru_cache(maxsize=32)
def board_space(board):
    hands = np.asarray(list(combinations([c for c in range(52) if c not in board], 2)), dtype=np.int16)
    ranks = rank_cards(np.column_stack((hands, np.tile(board, (len(hands),1)))))[0]
    ordered = np.sort(ranks)
    # Board-only rank percentile. It does not condition the opponent on hero cards.
    strength = (np.searchsorted(ordered,ranks,'left') + np.searchsorted(ordered,ranks,'right'))/(2*len(ranks))
    return hands, ranks, strength


class RiverSpace:
    def __init__(self, position):
        if len(position['board']) != 5:
            raise ValueError('Exact continuation oracle is river-only')
        hands, ranks, strength = board_space(tuple(position['board']))
        legal = ~np.isin(hands, position['hole']).any(axis=1)
        self.hands, self.ranks, self.strength = hands[legal], ranks[legal], strength[legal]
        hero_rank = rank_cards([position['hole'] + position['board']])[0][0]
        self.outcome = np.sign(hero_rank - self.ranks)
        self.position = position
        self.state, self.spent = build_state(position)
        self.hero = position['hero']

    def weights(self, name):
        if name not in RANGES:
            raise ValueError('Unknown root range')
        return np.exp((0 if name == 'uniform' else 4 if name == 'strong' else -4)*(self.strength-.5))


def opponent_probabilities(name, state, strength, spent):
    if name not in PANELS:
        raise ValueError('Unknown frozen opponent')
    p = np.zeros((len(strength),3), dtype=np.float64)
    legal = state.legal_actions()
    if name == 'check_call':
        p[:,1] = 1
    elif name == 'value':
        if 0 in legal:
            p[:,0] = np.where(strength < .55, .85, .05)
            p[:,2] = np.where(strength >= .85, .65, .02)
            p[:,1] = 1-p[:,0]-p[:,2]
        else:
            p[:,2] = np.where(strength >= .7, .85, .05)
            p[:,1] = 1-p[:,2]
    else:
        p[:,0] = .1*(1-strength)
        p[:,2] = .25 + .5*strength
        p[:,1] = 1-p[:,0]-p[:,2]
    p[:,[a for a in range(3) if a not in legal]] = 0
    return p/p.sum(axis=1,keepdims=True)


def river_audit(space, candidate, opponent='value', weights=None, deadline=lambda: None):
    weights = space.weights('uniform') if weights is None else np.asarray(weights,dtype=np.float64)
    if weights.shape != (len(space.hands),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid root range weights')
    if not np.isfinite(weights.sum()):
        raise ValueError('Nonfinite total range weight')
    if weights.sum() == 0:
        return dict(status='UNSUPPORTED', reason='Root range has no compatible mass')
    weights = weights/weights.sum()
    decisions = []
    hero = space.hero

    def walk(state, spent, reach, path):
        deadline()
        mass = float(reach.sum())
        if mass == 0:
            return 0.,0.
        if state.is_terminal():
            values = np.where(space.outcome > 0, spent[1-hero],
                np.where(space.outcome < 0, -spent[hero], (spent[1-hero]-spent[hero])/2))/100
            value = float(reach @ values)
            return value,value
        actor = state.current_player()
        legal = state.legal_actions()
        q, brq = [], []
        opp = opponent_probabilities(opponent,state,space.strength,spent) if actor != hero else None
        for action in legal:
            branch = reach if actor == hero else reach*opp[:,action]
            child = state.child(action)
            next_spent = list(spent)
            if action == 0:
                # Folding payoff is independent of private hand, including hero folds.
                value = float(branch.sum())*child.returns()[hero]/100
                v,b = value,value
            else:
                next_spent[actor] = max(spent)+(200 if action == 2 else 0)
                v,b = walk(child,next_spent,branch,path+(action,))
            q.append(v); brq.append(b)
        if actor != hero:
            return sum(q),sum(brq)
        p = probabilities(candidate,state)
        current = float(p[legal] @ q)
        best = max(brq)  # Hidden hands integrated BEFORE this maximisation.
        local = (max(q)-current)/mass
        gap = (best-current)/mass
        if min(local,gap) < -1e-8:
            raise ArithmeticError('Negative decision gap')
        terminal_call = 1 in legal and state.child(1).is_terminal()
        cost = max(spent)-spent[hero]
        threshold = cost/(sum(spent)+cost) if terminal_call and cost > 0 else None
        equity = float(np.clip(reach @ ((space.outcome>0)+.5*(space.outcome==0))/mass,0,1))
        universal_fold_error = terminal_call and cost > 0 and bool(np.all(space.outcome >= 0))
        guaranteed_equity = .5 if np.any(space.outcome == 0) else 1.
        fold_cost_bound = (guaranteed_equity*(sum(spent)+cost)-cost)/100
        certain_loss = bool(np.all(space.outcome[reach > 0] < 0))
        best_actions = [a for a,v in zip(legal,q) if abs(v-max(q)) <= 1e-10*mass]
        decisions.append(dict(path=list(path), legal=legal, probabilities=p.tolist(),
            q_bb={str(a):v/mass for a,v in zip(legal,q)}, local_gap_bb=max(0.,local),
            remaining_response_gap_bb=max(0.,gap), candidate_value_bb=current/mass,
            best_response_value_bb=best/mass, best_local_actions=best_actions,
            equity=equity, pot_bb=sum(spent)/100, call_cost_bb=cost/100,
            pot_odds_threshold=threshold, terminal_call=terminal_call,
            globally_unbeatable=bool(np.all(space.outcome>=0)),
            range_certain_loss=certain_loss, compatible_hands=len(space.hands),
            conditional_support=int(np.count_nonzero(reach)),
            provable_fold_probability=float(p[0]) if universal_fold_error else None,
            provable_fold_cost_lower_bound_bb=float(p[0])*fold_cost_bound if universal_fold_error else None,
            range_losing_call_probability=float(p[1]) if certain_loss and terminal_call and cost>0 else None))
        return current,best

    value,best = walk(space.state,space.spent,weights,())
    root = next(d for d in decisions if not d['path'])
    return dict(status='COMPLETE', root=root, candidate_value_bb=value,
        best_response_value_bb=best, decision_count=len(decisions),
        worst_continuation_decisions=sorted(decisions,key=lambda d:d['local_gap_bb'],reverse=True)[:3])
