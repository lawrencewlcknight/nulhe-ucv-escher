"""Read-only P104 inference and independent rollout/reference machinery.

No replay restoration, optimizer updates, or writes to the source checkpoint.
All stochastic calls take an explicit private RNG.
"""
from copy import deepcopy
from itertools import combinations

import numpy as np
import torch

from unbiased_escher.estimator import control_variate_advantage, variance_optimal_beta
from .checkpointing import load_checkpoint_payload, sha256_file
from .config import make_solver
from .game import SMALL_BET_CHIPS, serialisable_game_definition
from .p104_protocol import ITERATION, POLICY_SHA256, TRAINER_FINGERPRINT
from .policy_replay_artifacts import read_metadata
from .strategic_audit.library import build_state


ARMS = ('current', 'average4', 'all_actions', 'uniform')


def forbidden(*args, **kwargs):
    raise RuntimeError('Learning/replay writes are forbidden in the supervision audit')


class FrozenRuntime:
    def __init__(self, metadata):
        self.config = deepcopy(metadata['config'])
        if self.config['use_instantaneous_predictor'] or self.config['q_ensemble_size'] != 2:
            raise ValueError('Audit supports the P104 two-critic, nonpredictive learner')
        scratch = deepcopy(self.config)
        for name in ('advantage_buffer_size','ave_policy_buffer_size','baseline_buffer_size','calibration_buffer_size'):
            scratch[name] = 8
        self.solver = make_solver(scratch, seed=0, workers=1)
        solver = self.solver
        if metadata['game'] != serialisable_game_definition() or metadata['encoder'] != solver.feature_encoder.metadata():
            raise ValueError('Frozen game/encoder mismatch')
        self.source_iteration = int(metadata['solver']['num_iteration'])
        # Freeze both players at P104, evaluating the prospective next update.
        solver.num_iteration = self.source_iteration + 1
        for trainer, saved in zip(solver.regret_trainers, metadata['regret_trainers'], strict=True):
            trainer.model.load_state_dict(saved['model'])
            trainer.target_model.load_state_dict(saved['target_model'])
        solver.ave_policy_trainer.model.load_state_dict(metadata['average_policy_trainer']['model'])
        for trainer, saved in zip(solver.q_value_trainer.members, metadata['q_ensemble']['members'], strict=True):
            trainer.model.load_state_dict(saved['model'])
            trainer.target_model.load_state_dict(saved['target_model'])
            trainer.target_version = int(saved['target_version'])
        if (metadata['calibration'] is None) != (solver.calibration_trainer is None):
            raise ValueError('Frozen calibration mismatch')
        if solver.calibration_trainer is not None:
            for name in ('model','target_model'):
                getattr(solver.calibration_trainer,name).load_state_dict(metadata['calibration'][name])
        trainers = [*solver.regret_trainers, solver.ave_policy_trainer,
                    *solver.q_value_trainer.members]
        if solver.calibration_trainer is not None:
            trainers.append(solver.calibration_trainer)
            solver.calibration_trainer.add = forbidden
        for trainer in trainers:
            for name in ('model','target_model'):
                model = getattr(trainer,name,None)
                if model is not None:
                    model.eval()
                    model.requires_grad_(False)
            trainer.add_data = forbidden
            trainer.train_model = forbidden
        solver.q_value_trainer.add_data = forbidden
        solver.iteration = forbidden
        self.game = solver.game
        self.encoder = solver.feature_encoder
        self.bb_scale = float(solver.max_utility / SMALL_BET_CHIPS)

    def policy(self, state):
        p = np.asarray(self.solver.regret_trainers[state.current_player()].get_policy(
            state, self.solver.num_iteration), dtype=np.float64)
        return p / p.sum()

    def policy_batch(self, states):
        """Batched inference, with the production NumPy regret-matching rule."""
        result = []
        for start in range(0,len(states),128):
            chunk = states[start:start+128]
            actor = chunk[0].current_player()
            if any(s.current_player() != actor for s in chunk):
                raise ValueError('Mixed actors in a public-history batch')
            trainer = self.solver.regret_trainers[actor]
            x = np.stack([self.encoder.information_state(s,actor) for s in chunk])
            masks = np.asarray([s.legal_actions_mask() for s in chunk],np.float32)
            with torch.no_grad():
                raw = trainer.model(torch.from_numpy(x)).numpy()*masks
            for s,r in zip(chunk,raw):
                p = trainer.regret_matching(r,s.legal_actions()).astype(np.float64)
                result.append(p/p.sum())
        return np.asarray(result)

    def controls(self, state, hero, fold):
        solver = self.solver
        solver.q_value_trainer.active_fold = fold
        q, disagreement = solver.q_value_trainer.get_baseline_and_disagreement(state,hero)
        if solver.calibration_trainer is None:
            means, variances = np.zeros(3),np.ones(3)
        else:
            means,variances,_ = solver.calibration_trainer.predict_all(
                solver.get_infostate_tensor(state,hero),solver.num_iteration,disagreement,hero)
        beta = (np.full(3,solver.fixed_control_variate_beta) if solver.fixed_control_variate_beta is not None
                else variance_optimal_beta(q,means,beta_min=solver.beta_min,beta_max=solver.beta_max,ridge=solver.beta_ridge))
        policy = self.policy(state)
        mask = np.asarray(state.legal_actions_mask(),float)
        proposal = solver._traverser_sampling_policy(q_values=q,beta=beta,residual_means=means,
            predicted_variances=variances,policy=policy,legal_mask=mask)
        return q,beta,proposal,policy,mask

    def proposal_batch(self, states, hero, fold):
        """Range audit: call the native sampling hook using batched frozen inputs."""
        solver = self.solver
        result = []
        for start in range(0,len(states),128):
            chunk = states[start:start+128]
            masks = np.asarray([s.legal_actions_mask() for s in chunk],float)
            histories = np.stack([self.encoder.full_state(s) for s in chunk])
            with torch.no_grad():
                # Two-fold leave-one-out inference has exactly one held-out critic.
                q = solver.q_value_trainer.members[1-fold].target_model(torch.from_numpy(histories)).numpy()
            q = q*(1 if hero == 0 else -1)*masks
            disagreement = np.zeros_like(q)
            if solver.calibration_trainer is None:
                means,variances = np.zeros_like(q),np.ones_like(q)
            else:
                calibration = solver.calibration_trainer
                features = np.stack([calibration.feature(self.encoder.information_state(s,hero),a,
                    solver.num_iteration,0.,hero) for s in chunk for a in range(3)])
                with torch.no_grad():
                    predicted = calibration.target_model(torch.from_numpy(features)).reshape(len(chunk),3,2)
                    means = predicted[:,:,0].numpy().astype(float)
                    variances = (predicted[:,:,1].clamp(-8,6).exp()+calibration.minimum_variance).numpy().astype(float)
            policies = self.policy_batch(chunk)
            for i in range(len(chunk)):
                beta = (np.full(3,solver.fixed_control_variate_beta) if solver.fixed_control_variate_beta is not None
                        else variance_optimal_beta(q[i],means[i],beta_min=solver.beta_min,beta_max=solver.beta_max,ridge=solver.beta_ridge))
                result.append(solver._traverser_sampling_policy(q_values=q[i],beta=beta,
                    residual_means=means[i],predicted_variances=variances[i],policy=policies[i],legal_mask=masks[i]))
        return np.asarray(result)

    def chance(self, state, rng):
        while state.is_chance_node():
            actions, probabilities = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(actions,p=probabilities)))
        return state

    def direct(self, state, hero, rng, check=lambda:None):
        """Independent terminal-payoff rollout; never query any critic/calibrator."""
        while not state.is_terminal():
            check()
            self.chance(state,rng)
            if not state.is_terminal():
                state.apply_action(int(rng.choice(3,p=self.policy(state))))
        return float(state.returns()[hero]/self.solver.max_utility)

    def estimate(self, state, hero, fold, rng, *, arm='current', check=lambda:None, root=True):
        """Production recursive UCV estimator, with a selected-root-only ablation."""
        check()
        self.chance(state,rng)
        if state.is_terminal():
            return float(state.returns()[hero]/self.solver.max_utility),None
        actor = state.current_player()
        if root and arm == 'all_actions':
            if actor != hero:
                raise ValueError('All-action root must belong to the traverser')
            policy = self.policy(state)
            q = np.zeros(3)
            for action in state.legal_actions():
                q[action],_ = self.estimate(state.child(action),hero,fold,rng,check=check,root=False)
            value = float(policy@q)
            advantage = np.where(state.legal_actions_mask(),q-value,0.)
            return value,dict(q=q,advantage=advantage,critic=None,correction=0.,sample_probability=1.)
        q,beta,proposal,policy,mask = self.controls(state,hero,fold)
        if actor == hero:
            sample = mask/mask.sum() if root and arm == 'uniform' else proposal
        else:
            sample = np.where(mask,policy,0.)
            sample /= sample.sum()
        action = int(rng.choice(3,p=sample))
        value,_ = self.estimate(state.child(action),hero,fold,rng,check=check,root=False)
        result = control_variate_advantage(q,beta=beta,sampled_action=action,
            sample_probability=float(sample[action]),sampled_return=value,policy=policy,legal_actions_mask=mask)
        return result.policy_value,dict(q=result.q_values,advantage=result.advantages,critic=q,
            correction=result.importance_correction,sample_probability=float(sample[action]))

    def update_strategy(self, state, advantage):
        """One hypothetical production target update; no learning or model mutation."""
        trainer = self.solver.regret_trainers[state.current_player()]
        x = torch.as_tensor(self.encoder.information_state(state),dtype=torch.float32)
        mask = np.asarray(state.legal_actions_mask(),float)
        with torch.no_grad():
            previous = np.maximum(trainer.target_model(x).numpy()*mask,0)
        age = (self.solver.num_iteration-1)**trainer.alpha
        target = previous*age/(age+1.5)+advantage
        return trainer.regret_matching(target,state.legal_actions()).astype(float)


def load_runtime(state, policy, *, smoke=False):
    _,metadata = read_metadata(state)
    saved = load_checkpoint_payload(policy)
    if (saved['game'] != metadata['game'] or saved['feature_encoder'] != metadata['encoder']
            or saved['training_config'] != metadata['config']
            or saved['outer_iteration'] != metadata['solver']['num_iteration']
            or any(not torch.equal(v,saved['policy_state_dict'][k])
                   for k,v in metadata['average_policy_trainer']['model'].items())):
        raise ValueError('Source metadata and policy do not match')
    if not smoke and (sha256_file(policy) != POLICY_SHA256
            or metadata['solver']['num_iteration'] != ITERATION
            or metadata['repository_commit'] != TRAINER_FINGERPRINT):
        raise ValueError('Production audit requires pinned P104 metadata')
    return FrozenRuntime(metadata)


def normalise_logs(logs):
    finite = np.isfinite(logs)
    if not finite.any():
        return None
    weights = np.zeros(len(logs))
    weights[finite] = np.exp(logs[finite]-logs[finite].max())
    return weights/weights.sum()


def conditional_ranges(runtime, position, check=lambda:None):
    """Opponent-reach posterior and actual own-proposal-weighted visitation range."""
    seen = position['hole']+position['board']
    hands = list(combinations([c for c in range(52) if c not in seen],2))
    roots = [build_state(position,h)[0] for h in hands]
    histories = [s.history() for s in roots]
    states = [runtime.game.new_initial_state() for _ in hands]
    opponent_logs = np.zeros(len(hands))
    own_logs = np.zeros((2,len(hands)))
    hero = position['hero']
    for step in range(len(histories[0])):
        check()
        if not states[0].is_chance_node():
            action = histories[0][step]
            if states[0].current_player() != hero:
                probabilities = runtime.policy_batch(states)[:,action]
                with np.errstate(divide='ignore'):
                    opponent_logs += np.log(probabilities)
            else:
                for fold in range(2):
                    probabilities = runtime.proposal_batch(states,hero,fold)[:,action]
                    if np.any(probabilities <= 0):
                        raise ValueError('Traverser proposal lost full support')
                    own_logs[fold] += np.log(probabilities)
        for s,h in zip(states,histories):
            s.apply_action(h[step])
    target = normalise_logs(opponent_logs)
    if target is None:
        raise ValueError('Zero opponent reach at a collected position')
    production = [normalise_logs(opponent_logs+own_logs[f]) for f in range(2)]
    return hands,roots,np.asarray([target,*production]),dict(
        range_names=['opponent_reach','production_fold0','production_fold1'],
        compatible_hands=len(hands),posterior_effective_hands=float(1/(target@target)),
        production_total_variation=[float(np.abs(p-target).sum()/2) for p in production],
        own_reach_log_range=[float(np.ptp(x)) for x in own_logs],
        scope='One frozen iteration. No claim about varying information-state weights across training iterations.')


def exact_river_q(runtime, roots, hero, check=lambda:None):
    """Enumerate every remaining action for every compatible private hand."""
    if any(s.is_chance_node() for s in roots):
        raise ValueError('Exact river reference cannot contain future chance')
    def value(states):
        check()
        if states[0].is_terminal():
            return np.asarray([s.returns()[hero]/runtime.solver.max_utility for s in states])
        if states[0].is_chance_node():
            raise ValueError('Exact reference supports rivers only')
        if states[0].current_player() == hero:
            # Only hero-visible information is used by the policy.
            probabilities = np.broadcast_to(runtime.policy(states[0]),(len(states),3))
        else:
            probabilities = runtime.policy_batch(states)
        answer = np.zeros(len(states))
        for action in states[0].legal_actions():
            if np.any(probabilities[:,action]):
                answer += probabilities[:,action]*value([s.child(action) for s in states])
        return answer
    result = np.zeros((len(roots),3))
    for action in roots[0].legal_actions():
        result[:,action] = value([s.child(action) for s in roots])
    return result
