#!/usr/bin/env python3
"""Instrument ONE restored iteration using only the archived training package.

No resume checkpoint is written. Target capture uses private diagnostic RNGs and
runs immediately before the corresponding learner, preserving update order.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time


def run(source, state, contract, policy, output, *, before_iteration=None, after_iteration=None):
    sys.path.insert(0,str(Path(source).resolve()))
    from importlib.metadata import version
    import numpy as np
    import torch
    from hulhe_escher.train import source_fingerprint, write_json
    from hulhe_escher.config import make_solver
    from hulhe_escher.training_state import read_training_state, restore_training_state
    from hulhe_escher.checkpointing import load_checkpoint_payload, sha256_file
    from adaptive_escher.frozen_target_cache import frozen_targets, transition_batch
    contract=json.loads(Path(contract).read_text())
    runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        open_spiel=version('open_spiel'),ray=version('ray') if contract['workers']>1 else None)
    if runtime!=contract['runtime'] or source_fingerprint()!=contract['source_fingerprint']:
        raise ValueError('Archived trainer source/runtime differs from original recovery contract')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    payload=read_training_state(Path(state))
    torch.set_num_threads(contract['threads'])
    solver=make_solver(contract['config'],seed=contract['seed'],workers=contract['workers'])
    report=dict(status='RESTORING',source_fingerprint=source_fingerprint(),runtime=runtime,
        source_policy_sha256=sha256_file(policy),source_manifest_sha256=sha256_file(Path(state)/'manifest.json'),
        scope='One isolated production-sized iteration; no downstream policy refit or reusable continuation saved',
        captures=[],phases=[],variance_note='Residual/disagreement/importance-correction statistics are diagnostic signals, not a causal estimate of variance reduction or proof that Bellman targets are correct.')
    write_json(output/'report.json',report)
    diagnostic_seconds=0.
    def summary(values):
        x=np.asarray(values,dtype=np.float64).reshape(-1)
        if not np.isfinite(x).all(): raise FloatingPointError('Nonfinite captured target/prediction')
        return dict(count=len(x),mean=float(x.mean()),sd=float(x.std()),
                    abs_p50=float(np.quantile(abs(x),.5)),abs_p95=float(np.quantile(abs(x),.95)),
                    abs_p99=float(np.quantile(abs(x),.99)),max_abs=float(abs(x).max()))
    def capture(trainer,kind,index,iteration):
        tick=time.monotonic(); size=len(trainer.buffer)
        if not size: return None
        ids=np.sort(np.random.default_rng(9300+index+(100 if kind=='critic' else 0)).choice(size,min(size,4096),replace=False))
        if kind=='regret':
            x=torch.as_tensor(trainer.buffer.infostate_buf[ids]); mask=torch.as_tensor(trainer.buffer.q_value_mask_buf[ids])
            values=torch.as_tensor(trainer.buffer.q_value_buf[ids]); age=(iteration-1)**trainer.alpha
            with torch.no_grad():
                target=trainer.predict(trainer.target_model,x,mask).clamp_min(0)*age/(age+1.5)+values
                before=trainer.predict(trainer.model,x,mask)
            features=x.numpy(); legal=mask.numpy().astype(bool)
            arrays=dict(features=features,mask=mask.numpy(),increment=values.numpy())
            predict=lambda:trainer.predict(trainer.model,x,mask)
        else:
            batch=transition_batch(trainer.buffer,ids,'cpu')
            x=torch.as_tensor(trainer.buffer.history_buf[ids]); action=torch.as_tensor(trainer.buffer.action_buf[ids],dtype=torch.long)
            with torch.no_grad():
                target=frozen_targets(trainer,batch,iteration)
                before=trainer.model(x).gather(1,action[:,None]).squeeze(1)
            arrays=dict(history=x.numpy(),actions=action.numpy(),reward=batch[3].numpy(),done=batch[-1].numpy(),
                        next_history=batch[1].numpy(),next_state=batch[2].numpy(),next_mask=batch[4].numpy(),next_player=batch[5].numpy())
            legal=np.ones(len(ids),bool)
            predict=lambda:trainer.model(x).gather(1,action[:,None]).squeeze(1)
        name=f'{kind}_{index}'
        arrays.update(row_indices=ids,target=target.numpy(),before=before.numpy())
        # Stored before calling the original learner. Captures are bounded, not full replay copies.
        np.savez_compressed(output/f'{name}_before.npz',**arrays)
        row=dict(name=name,rows=size,sampled_rows=len(ids),target=summary(target.numpy()[legal]),
            before_error=summary((before-target).numpy()[legal]),
            before_mse=float(np.mean(np.square((before-target).numpy()[legal]))),
            artifact=f'{name}_before.npz',sha256=sha256_file(output/f'{name}_before.npz'))
        report['captures'].append(row)
        return row,predict,target,legal,time.monotonic()-tick
    try:
        restore_training_state(solver,payload,seed=contract['seed'],repository_commit=contract['source_fingerprint'],config=contract['config'])
        saved=load_checkpoint_payload(policy)
        if solver.num_iteration!=saved['outer_iteration'] or any(not torch.equal(v.cpu(),saved['policy_state_dict'][k]) for k,v in solver.ave_policy_trainer.model.state_dict().items()):
            raise ValueError('Restored solver and P32 export differ')
        report.update(source_iteration=solver.num_iteration,restore_seconds=time.monotonic()-started,status='ITERATING')
        if before_iteration is not None:
            before_iteration(solver)
        # Releasing restored mmap metadata reduces resident working set pressure.
        del payload
        old_collect=solver.collect_training_data
        def collect(player):
            tick=time.monotonic(); nodes=solver.nodes_touched; episodes=solver.episode
            result=old_collect(player); seconds=time.monotonic()-tick
            report['phases'].append(dict(name=f'collect_{player}',seconds=seconds,
                trajectories=solver.episode-episodes,nodes=solver.nodes_touched-nodes,
                trajectories_per_second=(solver.episode-episodes)/seconds,nodes_per_second=(solver.nodes_touched-nodes)/seconds))
            write_json(output/'report.json',report)
            return result
        solver.collect_training_data=collect
        def wrap(trainer,kind,index):
            original=trainer.train_model
            def train(iteration):
                nonlocal diagnostic_seconds
                snap=capture(trainer,kind,index,iteration)
                if snap: diagnostic_seconds+=snap[-1]
                tick=time.monotonic(); result=original(iteration); seconds=time.monotonic()-tick
                report['phases'].append(dict(name=f'fit_{kind}_{index}',seconds=seconds,
                    stats=getattr(trainer,'last_fit_stats',{})))
                if snap:
                    row,predict,target,legal,_=snap; tick=time.monotonic()
                    with torch.no_grad(): after=predict().numpy()
                    row.update(after_mse=float(np.mean(np.square(after-target.numpy())[legal])),
                        after_error=summary((after-target.numpy())[legal]),
                        note='Same sampled replay rows before/after fitting; in-sample approximation, not held-out accuracy')
                    np.savez_compressed(output/f'{kind}_{index}_after.npz',prediction=after)
                    diagnostic_seconds+=time.monotonic()-tick
                write_json(output/'report.json',report)
                return result
            trainer.train_model=train
        for i,t in enumerate(solver.regret_trainers): wrap(t,'regret',i)
        for i,t in enumerate(solver.q_value_trainer.members): wrap(t,'critic',i)
        tick=time.monotonic(); before=solver.num_iteration
        solver.iteration()
        if solver.num_iteration!=before+1 or solver._stop_requested: raise RuntimeError('Isolated iteration did not complete')
        report.update(status='COMPLETE',iteration=solver.num_iteration,iteration_seconds=time.monotonic()-tick,
            diagnostic_capture_seconds=diagnostic_seconds,architecture_statistics=dict(solver._architecture_stats),
            minimum_sampling_probability=solver._minimum_sample_probability)
        if after_iteration is not None:
            after_iteration(solver)
        report['root_control_variance']=root_control_variance(solver)
    except BaseException as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started; write_json(output/'report.json',report)
        if hasattr(solver,'close'): solver.close()
    return report


def root_control_variance(solver, repeats=64):
    """Paired root-only counterfactual with identical downstream return estimates.

    Fixed full states deliberately condition on opponent cards. This is a local
    variance diagnostic, not an information-set value oracle or exploitability.
    Disable all replay writes and restore RNG/counters, including on failure.
    """
    import numpy as np
    import torch
    import unbiased_escher.solver as module
    saved_rng=solver._capture_rng_state()
    saved_nodes=solver.nodes_touched
    saved_min=solver._minimum_sample_probability
    saved_fold=solver.q_value_trainer.active_fold
    saved_threads=torch.get_num_threads()
    original=module.control_variate_advantage
    originals=[]
    def disable(obj,name):
        originals.append((obj,name,getattr(obj,name)))
        setattr(obj,name,lambda *a,**kw: None)
    samples=[]
    def observe(q_values,**kwargs):
        value=original(q_values,**kwargs)
        zero=original(np.zeros_like(q_values),**dict(kwargs,beta=np.zeros_like(kwargs['beta'])))
        # DFS unwinds children before their parent, so the last observation is root.
        samples[:] = [(value.advantages.copy(),zero.advantages.copy(),float(value.q_residual),
                       float(value.importance_correction),float(kwargs['sample_probability']))]
        return value
    records=[]
    try:
        for t in solver.regret_trainers: disable(t,'add_data')
        disable(solver.ave_policy_trainer,'add_data');disable(solver.q_value_trainer,'add_data')
        if solver.calibration_trainer is not None: disable(solver.calibration_trainer,'add')
        disable(solver,'_record_estimate_diagnostics')
        module.control_variate_advantage=observe
        torch.set_num_threads(1)
        deck=np.random.default_rng(193001).choice(52,size=9,replace=False).tolist()
        state=solver.game.new_initial_state(); dealt=0
        positions=[]
        for goal in (4,7,9):
            while dealt<goal:
                if state.is_chance_node(): state.apply_action(deck[dealt]);dealt+=1
                else: state.apply_action(1)
            positions.append(state.clone())
        np.random.seed(193002)
        for index,state in enumerate(positions):
            for fold in range(len(solver.q_value_trainer.members)):
                solver.q_value_trainer.active_fold=fold
                controlled=[];zero=[];residuals=[];corrections=[];probabilities=[]
                for _ in range(repeats):
                    samples.clear();solver.dfs(state.clone(),state.current_player())
                    a,b,r,c,p=samples[-1]
                    controlled.append(a);zero.append(b);residuals.append(r);corrections.append(c);probabilities.append(p)
                mask=np.asarray(state.legal_actions_mask(),bool)
                a=np.asarray(controlled)[:,mask];b=np.asarray(zero)[:,mask]
                va=float(a.var(axis=0,ddof=1).mean());vb=float(b.var(axis=0,ddof=1).mean())
                records.append(dict(street=('preflop','flop','river')[index],fold=fold,repeats=repeats,
                    history=state.history(),controlled_advantage_variance=va,zero_root_control_variance=vb,
                    variance_ratio=va/vb if vb>0 else None,
                    paired_mean_difference=(a-b).mean(axis=0).tolist(),
                    paired_mean_difference_se=((a-b).std(axis=0,ddof=1)/np.sqrt(repeats)).tolist(),
                    residual_rms=float(np.sqrt(np.mean(np.square(residuals)))),
                    importance_correction_abs_p95=float(np.quantile(np.abs(corrections),.95)),
                    minimum_sampling_probability=min(probabilities)))
    finally:
        module.control_variate_advantage=original
        for obj,name,method in originals: setattr(obj,name,method)
        solver.nodes_touched=saved_nodes;solver._minimum_sample_probability=saved_min
        solver.q_value_trainer.active_fold=saved_fold
        solver._restore_rng_state(saved_rng);torch.set_num_threads(saved_threads)
    return dict(iteration=solver.num_iteration,records=records,
        scope='Exploratory root-only control ablation at three fixed full states after the isolated iteration, both critic folds. Same proposal and same downstream controlled return for each paired observation; not a full no-control training comparison. Replay writes disabled and RNG/counters restored.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','state','contract','policy','output'): p.add_argument('--'+n,type=Path,required=True)
    run(**vars(p.parse_args()))

if __name__=='__main__': main()
