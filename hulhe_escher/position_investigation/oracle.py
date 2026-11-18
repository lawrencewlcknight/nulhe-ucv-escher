"""History-conditioned hidden ranges and frozen learned river opponents.

Opponent own-card information is available to its policy; hero private cards and
future public cards are never passed into that policy's information-state input.
The observed hero action likelihood cancels conditional on the hero's known hand.
Uniform chance likelihoods also cancel after blocking all visible cards.
"""
import numpy as np
import torch

from ..strategic_audit.engine import probabilities, RiverSpace, river_audit


def with_hand(state, hero, hand):
    history=list(state.history())
    history[2*(1-hero):2*(1-hero)+2]=list(map(int,hand))
    replay=state.get_game().new_initial_state()
    for action in history: replay.apply_action(action)
    return replay


def batch_probabilities(policy, states, deadline=lambda:None):
    # Use native network batches where possible. Unknown policies keep their
    # own validated action interface (including regret-matching tie rules).
    if hasattr(policy,'feature_encoder') and hasattr(policy,'model'):
        output=[]
        for start in range(0,len(states),128):
            deadline(); chunk=states[start:start+128]
            x=np.stack([policy.feature_encoder.information_state(s,s.current_player()) for s in chunk])
            masks=np.asarray([s.legal_actions_mask() for s in chunk],bool)
            with torch.no_grad():
                p=torch.softmax(policy.model(torch.from_numpy(x)).masked_fill(~torch.from_numpy(masks),-1e21),-1).numpy()
            if not np.isfinite(p).all(): raise ValueError('Nonfinite batched policy')
            p=p.astype(np.float64)
            output.extend(p/p.sum(axis=1,keepdims=True))
        return np.asarray(output,dtype=np.float64)
    result=[]
    for s in states:
        deadline(); result.append(probabilities(policy,s))
    return np.asarray(result)


class LearnedOpponent:
    def __init__(self, space, policy, deadline=lambda:None):
        self.space,self.policy,self.deadline=space,policy,deadline
        self.cache={}

    def __call__(self,state):
        key=tuple(state.history()[4:])
        if key not in self.cache:
            self.cache[key]=batch_probabilities(self.policy,
                [with_hand(state,self.space.hero,h) for h in self.space.hands],self.deadline)
        return self.cache[key]

    def posterior(self):
        space=self.space
        states=[space.state.get_game().new_initial_state() for _ in space.hands]
        histories=[with_hand(space.state,space.hero,h).history() for h in space.hands]
        logs=np.zeros(len(states))
        for step,action in enumerate(space.state.history()):
            self.deadline()
            if not states[0].is_chance_node() and states[0].current_player()!=space.hero:
                p=batch_probabilities(self.policy,states,self.deadline)[:,action]
                with np.errstate(divide='ignore'): logs+=np.log(p)
            for s,h in zip(states,histories): s.apply_action(h[step])
        finite=np.isfinite(logs)
        if not finite.any(): return np.zeros(len(states))
        weights=np.zeros(len(states)); weights[finite]=np.exp(logs[finite]-logs[finite].max())
        return weights/weights.sum()


def learned_audit(position,candidate,opponent,deadline=lambda:None):
    space=RiverSpace(position); response=LearnedOpponent(space,opponent,deadline)
    weights=response.posterior()
    result=river_audit(space,candidate,response,weights,deadline)
    result['range_kind']='Exact frozen-opponent action-likelihood posterior; uniform compatible prior'
    result['posterior_effective_hands']=float(1/(weights@weights)) if weights.sum() else 0.
    return result
