"""Predeclared policy optimisation/capacity study on frozen P32 replay."""
import argparse
from copy import deepcopy
from hashlib import blake2b
import json
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from unbiased_escher.holdem_fitting import fit_validated_policy, policy_validation_mask
from unbiased_escher.replay_grouping import grouped_policy_data, exact_group_indices, IndexedFeatureRows
from .checkpointing import LoadedHULHEPolicy, sha256_file
from .diagnostic_model import ResidualPolicy
from .diagnostic_probes import compare_probes
from .game import load_hulhe_game, serialisable_game_definition
from .policy_replay_artifacts import load_policy_replay
from .policy_sampling_experiment import cells_for_groups, model_hash
from .train import write_json

ARMS = ('corrected_current_lr', 'corrected_low_lr', 'residual_low_lr')


def splits(features, fraction):
    holdout = policy_validation_mask(features, fraction)
    audit = np.zeros(len(features), dtype=bool)
    for start in range(0, len(features), 4096):
        x = features.numpy_batch(start, min(start+4096, len(features))).copy()
        x[x == 0] = 0
        audit[start:start+len(x)] = [blake2b(row.tobytes(), digest_size=1,
            person=b'policy-audit-v1').digest()[0] < 128 for row in x]
    return np.flatnonzero(~holdout), np.flatnonzero(holdout & ~audit), np.flatnonzero(holdout & audit)


def metrics(model, data, ids, cells, counts, batch=2048):
    # frequency bins are retained raw-row multiplicity, NOT objective weight.
    freq = np.digitize(counts, [2, 10, 100])
    sums = np.zeros((4,4,5), dtype=np.float64)
    numbers = np.zeros((4,4), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            idx = ids[start:start+batch]
            y, mask, w = [v[idx].double() for v in data[1:]]
            logp = torch.log_softmax(model(data[0][idx]).masked_fill(mask != 1, -1e20), -1).double()
            entropy = -(y*y.clamp_min(1e-30).log()).sum(-1)
            kl = -(y*logp).sum(-1)-entropy
            l1 = (logp.exp()-y).abs().sum(-1)
            disagreement = (logp.argmax(-1) != y.argmax(-1)).double()
            rows = torch.stack((torch.ones_like(w), kl, l1, entropy, disagreement),-1)*w[:,None]
            np.add.at(sums,(cells[idx]%4,freq[idx]),rows.numpy())
            np.add.at(numbers,(cells[idx]%4,freq[idx]),1)
    def report(row,n):
        if row[0] <= 0: return dict(groups=int(n), weight_mass=0, weighted_kl=None)
        _,kl,l1,entropy,disagreement = row/row[0]
        return dict(groups=int(n),weight_mass=float(row[0]),weighted_kl=float(kl),
                    weighted_action_l1=float(l1),target_entropy=float(entropy),
                    weighted_argmax_disagreement=float(disagreement))
    streets=('preflop','flop','turn','river'); bins=('1','2-9','10-99','100+')
    return dict(overall=report(sums.sum((0,1)),numbers.sum()),
        by_street={s:report(sums[i].sum(0),numbers[i].sum()) for i,s in enumerate(streets)},
        by_frequency={s:report(sums[:,i].sum(0),numbers[:,i].sum()) for i,s in enumerate(bins)},
        by_street_frequency={s:{b:report(sums[i,j],numbers[i,j]) for j,b in enumerate(bins)}
                             for i,s in enumerate(streets)})


def inference_timing(model, data):
    result = {}
    with torch.no_grad():
        for batch,repeats in ((1,200),(min(2048,len(data[0])),20)):
            x=data[0][np.arange(batch)]
            for _ in range(5): model(x)
            tick=time.perf_counter()
            for _ in range(repeats): model(x)
            elapsed=time.perf_counter()-tick
            result[str(batch)]=dict(seconds_per_call=elapsed/repeats,rows_per_second=batch*repeats/elapsed)
    return result


def select_arm(rows):
    scores={arm:float(np.mean([r['validation']['overall']['weighted_kl'] for r in rows if r['arm']==arm])) for arm in ARMS}
    if not all(np.isfinite(list(scores.values()))): raise ValueError('Nonfinite selection score')
    return min(ARMS,key=lambda a:(scores[a],ARMS.index(a))),scores


def run_refits(state, policy, output, *, seeds=(2701,2702), threads=8, smoke=False):
    if len(seeds)!=2 or len(set(seeds))!=2 or min(seeds)<0 or threads<1:
        raise ValueError('Exactly two distinct nonnegative seeds and positive threads required')
    torch.set_num_threads(threads)
    buffer,metadata,manifest=load_policy_replay(state)
    loaded=LoadedHULHEPolicy(load_hulhe_game(),policy)
    config=metadata['config']; payload=loaded.checkpoint; iteration=metadata['solver']['num_iteration']
    if (metadata['game']!=serialisable_game_definition() or metadata['encoder']!=payload['feature_encoder']
        or config!=payload['training_config'] or iteration!=payload['outer_iteration']):
        raise ValueError('Policy and full checkpoint identities differ')
    if model_hash(loaded.model) != model_hash_from_state(metadata['average_policy_trainer']['model'],loaded.model):
        raise ValueError('Source policy weights differ from recovery')
    if not config['average_policy_reset_each_fit']: raise ValueError('Requires reset-each-fit source')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    report=dict(status='FITTING',smoke=smoke,source_policy_sha256=sha256_file(policy),
        source_manifest_sha256=sha256_file(Path(state)/'manifest.json'),source_iteration=iteration,
        fit_seeds=list(seeds),arms=list(ARMS),results=[],sampling='objective_mass',
        selection_rule='Lowest mean validation weighted KL across both seeds; ties prefer simpler arm. Export first predeclared seed; never choose best seed.',
        split='Original canonical 5% holdout divided by independent fixed hash into validation and audit; all other groups train',
        inference_scope='One frozen replay; audit unused by optimiser/selection; independent fresh-deal playing evaluation',
        versions=dict(numpy=np.__version__,torch=torch.__version__))
    write_json(output/'experiment.json',report)
    started=time.monotonic()
    try:
        data,coverage=grouped_policy_data(buffer,iteration,config['gamma'],'cpu',context_columns=loaded.feature_encoder.replay_context_columns())
        train,valid,audit=splits(data[0],config['policy_validation_fraction'])
        if not all(len(x) for x in (train,valid,audit)): raise ValueError('All three partitions must be nonempty')
        first,inverse=exact_group_indices(buffer.infostate_buf)
        if not np.array_equal(first,data[0].rows): raise ValueError('Grouping order differs')
        counts=np.bincount(inverse,minlength=len(first)); del inverse
        cells=cells_for_groups(data[0],loaded.feature_encoder)
        keep=np.sort(np.r_[train,valid])
        fitting=(IndexedFeatureRows(data[0].source,data[0].rows[keep],'cpu'),*[v[keep] for v in data[1:]])
        report.update(coverage=coverage,partition_sizes=dict(train=len(train),validation=len(valid),audit=len(audit)),
            partition_sha256={n:blake2b(ids.tobytes()).hexdigest() for n,ids in zip(('train','validation','audit'),(train,valid,audit))})
        models={'P32':loaded.model}
        for seed in seeds:
            torch.manual_seed(seed); initial=deepcopy(loaded.model); initial.reset_parameters()
            arms=ARMS if seed==seeds[0] else ARMS[::-1]
            for arm in arms:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                model=(ResidualPolicy(loaded.feature_encoder.policy_layout,**(dict(width=16,blocks=1,head=8,branch_width=4) if smoke else {}))
                       if arm=='residual_low_lr' else deepcopy(initial))
                lr=config['average_policy_learning_rate'] if arm=='corrected_current_lr' else .001
                updates=8 if smoke else int(config['average_policy_train_steps'])
                trainer=SimpleNamespace(model=model,optimizer=torch.optim.Adam(model.parameters(),lr=lr),
                    _grouped_training_data=lambda _:fitting,validation_fraction=config['policy_validation_fraction'],
                    validation_steps=[4,8] if smoke else config['policy_validation_steps'],train_steps=updates,
                    batch_size=config['ave_policy_batch_size'],policy_group_sampling='objective_mass',policy_sampling_seed=seed)
                initial_hash=model_hash(model); tick=time.monotonic()
                fit_validated_policy(trainer,iteration)
                seconds=time.monotonic()-tick
                name=f'{arm}_seed{seed}'
                row=dict(arm=arm,fit_seed=seed,initial_model_sha256=initial_hash,learning_rate=lr,
                    parameter_count=sum(p.numel() for p in model.parameters()),model=model.checkpoint_metadata(),
                    fit_seconds=seconds,updates_per_second=updates/seconds,fit=trainer.last_fit_diagnostics,
                    training=metrics(model,data,train,cells,counts),validation=metrics(model,data,valid,cells,counts),
                    inference=inference_timing(model,data))
                cp=deepcopy(payload); cp.update(policy_model=model.checkpoint_metadata(),
                    policy_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    checkpoint_kind='frozen_replay_bottleneck_refit',experiment_id='hulhe_bottleneck_v1',
                    policy_refit=dict(arm=arm,fit_seed=seed,learning_rate=lr,source_sha256=report['source_policy_sha256']))
                with (output/f'{name}.pkl').open('xb') as handle: pickle.dump(cp,handle,pickle.HIGHEST_PROTOCOL)
                row.update(checkpoint=f'{name}.pkl',sha256=sha256_file(output/f'{name}.pkl'))
                models[name]=model; report['results'].append(row); write_json(output/'experiment.json',report)
                print(json.dumps(dict(arm=arm,seed=seed,fit_seconds=seconds,validation_kl=row['validation']['overall']['weighted_kl'])),flush=True)
        arm,scores=select_arm(report['results'])
        winner=next(r for r in report['results'] if r['arm']==arm and r['fit_seed']==seeds[0])
        report['selection']=dict(arm=arm,fit_seed=seeds[0],checkpoint=winner['checkpoint'],sha256=winner['sha256'],scores=scores)
        # Freeze selection before looking at the untouched audit split or any matches.
        write_json(output/'selection.json',report['selection'])
        for row in report['results']:
            row['audit']=metrics(models[Path(row['checkpoint']).stem],data,audit,cells,counts)
        report['source_policy_metrics']=dict(validation=metrics(loaded.model,data,valid,cells,counts),
            audit=metrics(loaded.model,data,audit,cells,counts),note='P32 may have trained on audit groups; source score is descriptive only')
        write_json(output/'probes.json',compare_probes(data,loaded.feature_encoder,metadata,models))
        report['status']='COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started; write_json(output/'experiment.json',report)
    return report


def model_hash_from_state(state,model):
    clone=deepcopy(model); clone.load_state_dict(state); return model_hash(clone)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','policy','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--threads',type=int,default=8)
    run_refits(**vars(p.parse_args()))

if __name__=='__main__': main()
