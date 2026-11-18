"""Conditional unit-root traversal checks; never confuse increments with cumulative regrets."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import t as student_t

from ..checkpointing import sha256_file
from ..strategic_audit.engine import RiverSpace, river_audit, probabilities
from ..strategic_audit.library import build_state
from ..strategic_audit.workflow import write_json
from .common import deadline_after
from .oracle import LearnedOpponent, with_hand


class SolverPolicy:
    def __init__(self,solver): self.solver=solver
    def action_probabilities(self,state):
        p=self.solver.regret_trainers[state.current_player()].get_policy(state,self.solver.num_iteration)
        return {a:float(p[a]) for a in state.legal_actions()}


@contextmanager
def read_only_traversal(solver, observe):
    """No replay/calibration/controller writes; restore all diagnostic state and RNGs."""
    import unbiased_escher.solver as module
    rng=solver._capture_rng_state(); original=module.control_variate_advantage
    threads=torch.get_num_threads()
    saved={name:deepcopy(getattr(solver,name)) for name in
           ('nodes_touched','episode','_minimum_sample_probability','_architecture_stats') if hasattr(solver,name)}
    fold=solver.q_value_trainer.active_fold
    overrides=[]
    def disable(obj,name):
        if hasattr(obj,name):
            overrides.append((obj,name,getattr(obj,name)));setattr(obj,name,lambda *a,**k:None)
    def wrapped(*args,**kwargs):
        value=original(*args,**kwargs)
        observe(value,kwargs)
        return value
    try:
        for trainer in solver.regret_trainers: disable(trainer,'add_data')
        disable(solver.ave_policy_trainer,'add_data');disable(solver.q_value_trainer,'add_data')
        disable(solver.q_value_trainer,'observe_control_return')
        if solver.calibration_trainer is not None: disable(solver.calibration_trainer,'add')
        disable(solver,'_record_estimate_diagnostics')
        module.control_variate_advantage=wrapped
        torch.set_num_threads(1)
        yield
    finally:
        module.control_variate_advantage=original
        for obj,name,method in reversed(overrides): setattr(obj,name,method)
        for name,value in saved.items(): setattr(solver,name,value)
        solver.q_value_trainer.active_fold=fold;solver._restore_rng_state(rng)
        torch.set_num_threads(threads)


def mean_diagnostic(samples,expected,alpha):
    samples=np.asarray(samples,dtype=float);expected=np.asarray(expected,dtype=float)
    n=len(samples)
    if n<2 or not np.isfinite(samples).all(): raise ValueError('Invalid estimator samples')
    se=samples.std(0,ddof=1)/np.sqrt(n)
    width=student_t.ppf(1-alpha/2,n-1)*se
    mean=samples.mean(0)
    return dict(repeats=n,mean=mean.tolist(),reference=expected.tolist(),standard_error=se.tolist(),
        difference=(mean-expected).tolist(),difference_interval=np.stack((mean-expected-width,mean-expected+width),axis=1).tolist(),
        confidence=1-alpha,reference_outside_interval=(np.abs(mean-expected)>width+1e-7).tolist(),
        caution='Approximate t intervals for independent traversals; heavy tails can impair coverage. A flag requires investigation, not proof of bias.')


class Hooks:
    def __init__(self,selection,output,*,repeats=64,max_seconds=1200):
        if repeats<2: raise ValueError('At least two repetitions required')
        self.selection=json.loads(Path(selection).read_text())
        self.output=Path(output); self.output.mkdir(parents=True,exist_ok=False)
        self.repeats=repeats;self.max_seconds=max_seconds;self.report=dict(status='STARTING',positions=[],
            scope='Unit-root conditional river advantage under frozen instantaneous regret strategies. '
            'Opponent cards sampled from its history-conditioned posterior. No prefix reach multiplier: '
            'this solver dfs resets to a unit root and returns normalized conditional advantages. '
            'This does not validate whole-game visitation weighting or historical cumulative regrets.',
            target_note='Probe targets are synthetic frozen one-step targets, not assumed to occur in the production replay. '
            'Separate iteration captures contain actual collected training rows.')
        self.probes=[]

    def before(self,solver):
        started=time.monotonic();check=deadline_after(self.max_seconds)
        policy=SolverPolicy(solver)
        choices=self.selection['positions']
        if not choices: raise ValueError('No supported development positions selected')
        # Simultaneous exploratory intervals: 24 positions * folds * at most 3 actions.
        alpha=.05/(len(choices)*len(solver.q_value_trainer.members)*3)
        self.report.update(status='CHECKING',alpha_per_action=alpha,requested_positions=len(choices),
            source_iteration=solver.num_iteration,normalizer_chips=solver.max_utility)
        last=[]
        def observe(value,kwargs):
            # DFS unwinds descendants first. Last callback is the selected root.
            last[:]=[np.asarray(value.advantages).copy()]
        try:
            with read_only_traversal(solver,observe):
                np.random.seed(481901)
                hand_rng=np.random.default_rng(481902)
                for choice in choices:
                    check();p=choice['position']
                    if p['partition']!='development' or len(p['board'])!=5:
                        raise ValueError('Upstream checks require development river roots')
                    space=RiverSpace(p);response=LearnedOpponent(space,policy,check);weights=response.posterior()
                    oracle=river_audit(space,policy,response,weights,check)
                    if oracle['status']!='COMPLETE':
                        self.report['positions'].append(dict(id=p['id'],status='UNSUPPORTED',reason='Zero regret-opponent history support'))
                        continue
                    s=space.state;hero=p['hero'];legal=s.legal_actions()
                    q=np.asarray([oracle['root']['q_bb'][str(a)] for a in legal])*100/solver.max_utility
                    current=probabilities(policy,s)[legal]
                    expected=q-float(current@q)
                    trainer=solver.regret_trainers[hero]
                    x=torch.as_tensor(solver.get_infostate_tensor(s)); mask=torch.as_tensor(s.legal_actions_mask(),dtype=torch.float32)
                    with torch.no_grad():
                        before=trainer.predict(trainer.model,x,mask).numpy().copy()
                        previous=trainer.predict(trainer.target_model,x,mask).clamp_min(0).numpy().copy()
                    age=solver.num_iteration**trainer.alpha
                    synthetic=previous*age/(age+1.5);synthetic[legal]+=expected
                    row=dict(id=p['id'],status='COMPLETE',selection_group=choice['selection_group'],legal=legal,
                        conditional_gap_bb=oracle['root']['local_gap_bb'],expected_increment=expected.tolist(),
                        cumulative_prediction_before=before.tolist(),synthetic_next_target=synthetic.tolist(),
                        synthetic_before_mse=float(np.mean((before[legal]-synthetic[legal])**2)),folds=[])
                    for fold in range(len(solver.q_value_trainer.members)):
                        solver.q_value_trainer.active_fold=fold;samples=[]
                        tick=time.monotonic()
                        for _ in range(self.repeats):
                            check();hand=space.hands[hand_rng.choice(len(weights),p=weights)]
                            last.clear();solver.dfs(with_hand(s,hero,hand),hero)
                            if not last: raise RuntimeError('Traversal did not produce a root estimate')
                            samples.append(last[-1][legal])
                        row['folds'].append(dict(fold=fold,seconds=time.monotonic()-tick,
                            **mean_diagnostic(samples,expected,alpha)))
                    self.probes.append((row,p,synthetic.copy()))
                    self.report['positions'].append(row)
                    write_json(self.output/'report.json',self.report)
            self.report['status']='AWAITING_ISOLATED_ITERATION' if self.probes else 'INCOMPLETE'
            if not self.probes:self.report['reason']='No selected root has positive regret-opponent history support'
        except TimeoutError as exc:
            self.report.update(status='INCOMPLETE',reason=str(exc))
        except BaseException as exc:
            self.report.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
        finally:
            self.report['probe_seconds']=time.monotonic()-started
            write_json(self.output/'report.json',self.report)

    def after(self,solver):
        for row,p,target in self.probes:
            s,_=build_state(p);trainer=solver.regret_trainers[p['hero']]
            x=torch.as_tensor(solver.get_infostate_tensor(s));mask=torch.as_tensor(s.legal_actions_mask(),dtype=torch.float32)
            with torch.no_grad(): after=trainer.predict(trainer.model,x,mask).numpy()
            row['cumulative_prediction_after']=after.tolist()
            row['synthetic_after_mse']=float(np.mean((after[row['legal']]-target[row['legal']])**2))
            row['actual_iteration_exact_replay_matches']=int(np.sum(np.all(
                trainer.buffer.infostate_buf[:len(trainer.buffer)]==x.numpy(),axis=1)))
            row['interpretation']=('Prediction change on a fixed probe; no claim this position was trained unless actual replay matched. '
                'Synthetic target freezes both players before the iteration; player updates during production collection can differ.')
        self.report['iteration_after']=solver.num_iteration
        if self.report['status']=='AWAITING_ISOLATED_ITERATION': self.report['status']='COMPLETE'
        write_json(self.output/'report.json',self.report)


def summarize_captures(root,state):
    """Actual training-row errors by common family, separate from synthetic probes."""
    from collections import defaultdict
    from ..policy_replay_artifacts import read_metadata
    from ..features import encoder_from_metadata
    from .common import families
    _,metadata=read_metadata(state);encoder=encoder_from_metadata(metadata['encoder'])
    result={}
    for player in (0,1):
        before=np.load(Path(root)/f'regret_{player}_before.npz',allow_pickle=False)
        after=np.load(Path(root)/f'regret_{player}_after.npz',allow_pickle=False)['prediction']
        bins=defaultdict(lambda:np.zeros(5))
        for i,key in enumerate(families(before['features'],encoder)):
            mask=before['mask'][i].astype(bool)
            target=before['target'][i,mask]
            bins[key]+=np.array([1,mask.sum(),np.square(before['before'][i,mask]-target).sum(),
                np.square(after[i,mask]-target).sum(),np.square(before['increment'][i,mask]).sum()])
        result[str(player)]={key:dict(rows=int(v[0]),legal_targets=int(v[1]),before_mse=float(v[2]/v[1]),
            after_mse=float(v[3]/v[1]),increment_rms=float(np.sqrt(v[4]/v[1]))) for key,v in bins.items()}
    report=dict(status='COMPLETE',by_player_family=result,
        note='Sampled actual production training rows, not held-out errors; increments/targets use solver utility units.')
    write_json(Path(root)/'family_errors.json',report)
    return report
