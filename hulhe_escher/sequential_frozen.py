"""Bounded supervised checks of sequential specialists on immutable saved data.

No self-play, policy replacement, production recovery, or poker-strength claim.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from unbiased_escher.replay_grouping import grouped_policy_data
from .checkpointing import LoadedHULHEPolicy, sha256_file
from .game import load_hulhe_game, serialisable_game_definition
from .policy_replay_artifacts import load_policy_replay
from .position_investigation.common import deadline_after, STREETS
from .regret_diagnostic import partition, load_captures, by_street
from .sequential_model import SequentialStreetNetwork
from . import sequential_protocol as protocol
from .train import write_json


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def dataset(x, y, mask, weight, encoder, kind):
    x, y, mask = [np.asarray(a, dtype=np.float32) for a in (x, y, mask)]
    weight = np.asarray(weight, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != encoder.policy_layout.total_size
            or y.shape != (len(x), 3) or mask.shape != y.shape or weight.shape != (len(x),)
            or not all(np.isfinite(a).all() for a in (x, y, mask, weight))
            or not np.isin(mask, [0, 1]).all() or np.any(mask.sum(1) == 0) or np.any(weight <= 0)):
        raise ValueError('Invalid frozen data')
    sc = encoder.replay_context_columns()[1]
    if not np.isin(x[:, sc], [0, 1]).all() or not np.all(x[:, sc].sum(1) == 1):
        raise ValueError('Invalid street identity')
    if kind == 'policy' and (np.any(y < 0) or not np.allclose(y.sum(1), 1, atol=1e-5)
                             or np.any(y[mask == 0] != 0)):
        raise ValueError('Invalid policy targets')
    parts = partition(x, mask)
    if any(len(ids) == 0 for ids in parts):
        raise ValueError('Empty information-state partition')
    return dict(features=x, target=y, mask=mask, weight=weight, street=x[:, sc].argmax(1),
                partitions=parts, kind=kind)


def prepare_data(state, policy, captures, *, smoke=False):
    loaded = LoadedHULHEPolicy(load_hulhe_game(), policy)
    buffer, metadata, _ = load_policy_replay(state)
    cp = loaded.checkpoint
    if (metadata['encoder'] != cp['feature_encoder'] or metadata['config'] != cp['training_config']
            or metadata['solver']['num_iteration'] != cp['outer_iteration']
            or metadata['game'] != serialisable_game_definition()):
        raise ValueError('Policy/replay identity mismatch')
    if not smoke:
        if (sha256_file(policy) != protocol.POLICY_SHA256
                or sha256_file(Path(state)/'manifest.json') != protocol.MANIFEST_SHA256
                or metadata['solver']['num_iteration'] != protocol.ITERATION
                or metadata['repository_commit'] != protocol.TRAINER_FINGERPRINT):
            raise ValueError('Requires pinned P104 inputs')
        for name, digest in protocol.CAPTURE_HASHES.items():
            if sha256_file(Path(captures)/name) != digest:
                raise ValueError('Pinned capture checksum mismatch')
    capture, regrets = load_captures(captures, loaded.feature_encoder)
    if (capture['source_policy_sha256'] != sha256_file(policy)
            or capture['source_manifest_sha256'] != sha256_file(Path(state)/'manifest.json')
            or capture['source_iteration'] != cp['outer_iteration']):
        raise ValueError('Captured targets have a different source')
    data = {f'regret_{p}': dataset(d['features'], d['target'], d['mask'], np.ones(len(d['target'])),
                                loaded.feature_encoder, 'regret') for p, d in enumerate(regrets)}
    grouped, coverage = grouped_policy_data(buffer, cp['outer_iteration'], metadata['config']['gamma'], 'cpu',
        context_columns=loaded.feature_encoder.replay_context_columns())
    features, targets, masks, weights = grouped
    # Bound fitting/validation cost AFTER exact grouping of the full reservoir.
    # This is an explicitly restricted diagnostic objective, not a full replay refit.
    sc = loaded.feature_encoder.replay_context_columns()[1]
    streets = np.empty(len(features), dtype=np.int8)
    for start in range(0, len(features), 4096):
        streets[start:start+4096] = features.numpy_batch(start, start+4096)[:, sc].argmax(1)
    rng = np.random.default_rng(240700)
    retained = []
    calibrated_weights = weights.numpy().astype(np.float64, copy=True)
    coverage['diagnostic_subset'] = []
    for s, name in enumerate(STREETS):
        ids = np.flatnonzero(streets == s)
        take = np.sort(rng.choice(ids, min(len(ids), protocol.POLICY_GROUPS_PER_STREET), replace=False))
        retained.extend(take)
        full_mass = float(calibrated_weights[ids].sum())
        retained_mass = float(calibrated_weights[take].sum())
        # Keeping every preflop group but only a fraction of later streets must
        # not accidentally turn this into an almost-all-preflop fitting task.
        scale = full_mass/retained_mass if retained_mass else 1.
        calibrated_weights[take] *= scale
        coverage['diagnostic_subset'].append(dict(street=name, total_groups=len(ids), retained_groups=len(take),
            full_mass=full_mass, retained_mass=retained_mass, street_mass_scale=scale))
    ids = np.sort(np.asarray(retained, dtype=np.int64))
    data['policy'] = dataset(features[ids].numpy(), targets[ids].numpy(), masks[ids].numpy(),
                             calibrated_weights[ids], loaded.feature_encoder, 'policy')
    return data, loaded.feature_encoder, dict(policy_sha256=sha256_file(policy),
        state_manifest_sha256=sha256_file(Path(state)/'manifest.json'),
        capture_report_sha256=sha256_file(Path(captures)/'report.json'),
        source_iteration=cp['outer_iteration'], captured_iteration=capture['iteration'], policy_coverage=coverage)


def sample(d, ids, n, rng):
    if not len(ids):
        raise ValueError('No training examples for requested street')
    mass = d['weight'][ids]
    return rng.choice(ids, n, replace=True, p=mass/mass.sum())


def tensors(d, ids):
    return [torch.from_numpy(d[k][ids]) for k in ('features', 'target', 'mask')]


def losses(prediction, target, mask, kind):
    if kind == 'regret':
        return ((prediction-target)*mask).square().mean(-1)  # Production mean over three actions.
    return -(target * torch.log_softmax(prediction.masked_fill(mask == 0, -1e20), -1)).sum(-1)


def policy_kl(prediction, reference, mask):
    logp = torch.log_softmax(prediction.masked_fill(mask == 0, -1e20), -1)
    logq = torch.log_softmax(reference.masked_fill(mask == 0, -1e20), -1)
    return (logq.exp() * (logq-logp)).sum(-1)


def update(model, optimizer, d, ids, *, street=None, aux_ids=None, coefficient=0., teacher=None):
    model.trainable_street(street)
    optimizer.zero_grad(set_to_none=True)  # Inactive Adam parameters must have grad=None, not zero.
    x, y, mask = tensors(d, ids)
    primary = losses(model(x), y, mask, d['kind']).mean()
    auxiliary = primary.new_zeros(())
    if coefficient:
        ax, ay, am = tensors(d, aux_ids)
        pred = model(ax)
        if d['kind'] == 'policy':
            with torch.no_grad():
                ref = teacher(ax)
            auxiliary = policy_kl(pred, ref, am).mean()
        else:
            auxiliary = losses(pred, ay, am, 'regret').mean()
    loss = primary + coefficient*auxiliary
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite sequential loss')
    loss.backward()
    gradients = [float(sum(p.grad.square().sum().item() for p in block.parameters() if p.grad is not None)**.5)
                 for block in model.specialists]
    if street is not None and any(p.grad is not None for s, block in enumerate(model.specialists)
                                  if s != street for p in block.parameters()):
        raise AssertionError('Inactive specialist received a gradient')
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True))
    optimizer.step()
    return dict(primary=float(primary.detach()), auxiliary=float(auxiliary.detach()),
                gradient_norm=norm, specialist_gradient_norms=gradients)


def predict(model, d, ids):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(d['features'][ids[i:i+1024]])).numpy()
                               for i in range(0, len(ids), 1024)])


def scores(model, d, ids, *, decomposition=False):
    prediction = predict(model, d, ids)
    y, mask = d['target'][ids], d['mask'][ids]
    values = losses(torch.from_numpy(prediction), torch.from_numpy(y), torch.from_numpy(mask), d['kind']).numpy()
    if d['kind'] == 'policy':
        entropy = -(y*np.log(np.maximum(y, 1e-30))).sum(1)
        values = np.maximum(values-entropy, 0.)
        probabilities = torch.from_numpy(prediction).masked_fill(torch.from_numpy(mask) == 0, -1e20).softmax(-1).numpy()
    else:
        positive = np.maximum(prediction, 0)*mask
        probabilities = np.divide(positive, positive.sum(1, keepdims=True),
            out=mask/mask.sum(1, keepdims=True), where=positive.sum(1, keepdims=True) > 0)
        positive_y = np.maximum(y, 0)*mask
        y = np.divide(positive_y, positive_y.sum(1, keepdims=True),
            out=mask/mask.sum(1, keepdims=True), where=positive_y.sum(1, keepdims=True) > 0)
    tv = .5*np.abs(probabilities-y).sum(1)
    def cell(keep):
        if not keep.any():
            return dict(rows=0, error=None, action_tv=None)
        w = d['weight'][ids][keep]
        return dict(rows=int(keep.sum()), error=float(np.average(values[keep], weights=w)),
                    action_tv=float(np.average(tv[keep], weights=w)))
    result = dict(metric='weighted_KL' if d['kind'] == 'policy' else 'row_MSE_over_three_actions',
        overall=cell(np.ones(len(ids), dtype=bool)),
        streets={name:cell(d['street'][ids] == s) for s, name in enumerate(STREETS)})
    if decomposition and d['kind'] == 'regret':
        sub = {k:d[k][ids] for k in ('features', 'target', 'mask', 'street')}
        result['target_decomposition'] = by_street(sub, prediction)
    return result


def interference(model, d, arm, seed, batch):
    """Independent five-step interventions from one warm start; validation only.

    A KL penalty has zero gradient at the reference itself. Multiple steps are
    essential to test protection after the first displacement.
    """
    train, valid, _ = d['partitions']
    rng = np.random.default_rng(seed+90000)
    probe = np.concatenate([ids[:128] for s in range(4)
                            if len(ids := valid[d['street'][valid] == s])])
    before = scores(model, d, probe)
    matrix = []
    for street in range(4):
        ids = train[d['street'][train] == street]
        others = train[d['street'][train] != street]
        if not len(ids) or not len(others):
            continue
        clone = deepcopy(model)
        optimizer = torch.optim.Adam(clone.parameters(), lr=protocol.LEARNING_RATE)
        for _ in range(5):
            routing = update(clone, optimizer, d, sample(d, ids, batch, rng),
                street=None if arm == 'unrestricted' else street,
                aux_ids=sample(d, others, min(batch, protocol.AUX_BATCH_SIZE), rng),
                coefficient=protocol.ARMS[arm], teacher=model)
        after = scores(clone, d, probe)
        matrix.append(dict(updated_street=STREETS[street], routing=routing,
            error_change={name:(None if before['streets'][name]['error'] is None else
                after['streets'][name]['error']-before['streets'][name]['error']) for name in STREETS}))
    return dict(before=before, intervention_updates=5, error_changes=matrix,
        interpretation='Positive error change is worse; validation probe only. Shared weights can change every street.')


def fit_dataset(d, encoder, seed, check, *, smoke=False):
    torch.manual_seed(seed)
    warm = SequentialStreetNetwork(encoder.policy_layout)
    optimizer = torch.optim.Adam(warm.parameters(), lr=protocol.LEARNING_RATE)
    rng = np.random.default_rng(seed)
    train, valid, _ = d['partitions']
    batch = 16 if smoke else protocol.BATCH_SIZE
    warm_steps = 3 if smoke else protocol.WARMUP_UPDATES
    budgets = (0, 1, 3) if smoke else protocol.BUDGETS
    start = time.monotonic()
    for _ in range(warm_steps):
        check()
        update(warm, optimizer, d, sample(d, train, batch, rng))
    warm_hash = state_hash(warm)
    warm_report = dict(updates=warm_steps, seconds=time.monotonic()-start, model_sha256=warm_hash,
                       validation=scores(warm, d, valid), architecture=warm.checkpoint_metadata())
    rows, selected = [], {}
    for arm, coefficient in protocol.ARMS.items():
        check()
        model, teacher = deepcopy(warm), deepcopy(warm)
        for p in teacher.parameters():
            p.requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol.LEARNING_RATE)
        rng = np.random.default_rng(seed+10000)  # Identical main and auxiliary draws across arms.
        best = deepcopy(model)
        score = scores(model, d, valid)
        best_score, best_step = score['overall']['error'], 0
        history = [dict(update=0, validation=score)]
        mass = np.array([d['weight'][train[d['street'][train] == s]].sum() for s in range(4)])
        start = time.monotonic()
        aggregate = dict(primary=0., auxiliary=0., max_gradient_norm=0.)
        for step in range(1, max(budgets)+1):
            check()
            if (step-1) % protocol.TEACHER_REFRESH == 0:
                teacher.load_state_dict(model.state_dict())
            street = int(rng.choice(4, p=mass/mass.sum()))
            ids = sample(d, train[d['street'][train] == street], batch, rng)
            other = train[d['street'][train] != street]
            aux = sample(d, other, min(batch, protocol.AUX_BATCH_SIZE), rng)
            stats = update(model, optimizer, d, ids, street=None if arm == 'unrestricted' else street,
                           aux_ids=aux, coefficient=coefficient, teacher=teacher)
            aggregate['primary'] += stats['primary']; aggregate['auxiliary'] += stats['auxiliary']
            aggregate['max_gradient_norm'] = max(aggregate['max_gradient_norm'], stats['gradient_norm'])
            if step in budgets:
                score = scores(model, d, valid)
                history.append(dict(update=step, validation=score))
                if score['overall']['error'] < best_score:
                    best_score, best_step, best = score['overall']['error'], step, deepcopy(model)
        elapsed = time.monotonic()-start
        selected[arm] = best
        rows.append(dict(arm=arm, seed=seed, coefficient=coefficient, initial_sha256=warm_hash,
            selected_update=best_step, updates_completed=max(budgets), fit_seconds=elapsed,
            updates_per_second=max(budgets)/elapsed, losses=aggregate, validation_history=history,
            training=scores(best, d, train, decomposition=True),
            validation=scores(best, d, valid, decomposition=True),
            interference=interference(warm, d, arm, seed, batch)))
    return warm_report, rows, selected


def write_summary(root, report):
    lines = ['# Sequential network frozen-target checks', '', f"Status: {report['status']}", '',
        'Supervised diagnostics only; no new self-play or evidence of improved poker strength.',
        'Policy metrics use a bounded subset after full replay grouping. Regret targets come from one saved iteration.', '',
        '| Dataset | Seed | Arm | Selected updates | Validation error | Audit error | Fit seconds |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in report['results']:
        audit = row.get('audit', {}).get('overall', {}).get('error')
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['arm']} | {row['selected_update']} | "
                     f"{row['validation']['overall']['error']:.6g} | {audit if audit is not None else 'pending'} | {row['fit_seconds']:.1f} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run(state, policy, captures, output, *, threads=8, max_seconds=5400, smoke=False):
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    check = deadline_after(max_seconds)
    root = Path(output).resolve()
    # Never place results inside the read-only inputs (or vice versa).
    for path in (state, policy, captures):
        source = Path(path).resolve()
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError('Output overlaps source')
    root.mkdir(parents=True, exist_ok=False)
    report = dict(status='PREPARING', protocol=protocol.PROTOCOL, smoke=smoke, results=[], warmups=[],
        collection_iterations=0, threads=threads, arms=protocol.ARMS, seeds=list(protocol.SEEDS),
        learning_rate=protocol.LEARNING_RATE, teacher_refresh_updates=protocol.TEACHER_REFRESH,
        limitations=['Fresh-network short fits, not continued production training.',
            'Policy objective restricted to at most 4096 randomly retained exact groups per street; relative weights retained within street and calibrated to full street mass.',
            'Regret captures cover one iteration; sampled-target error does not establish target correctness.',
            'Regret action TV compares regret-matched sampled targets, not an oracle strategy.',
            'Information-state splits prevent duplicate leakage; not independent boards or trajectories.',
            'No poker matches, policy exports or resumable checkpoints. Two fit seeds do not establish playing strength.'])
    started = time.monotonic()
    try:
        data, encoder, provenance = prepare_data(state, policy, captures, smoke=smoke)
        report['provenance'] = provenance
        report['partitions'] = {name:{label:dict(rows=len(ids), sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            by_street=[int((d['street'][ids] == s).sum()) for s in range(4)])
            for label, ids in zip(('train', 'validation', 'audit'), d['partitions'])} for name, d in data.items()}
        if any(not np.all(np.bincount(d['street'][ids], minlength=4) > 0)
               for d in data.values() for ids in d['partitions']):
            raise ValueError('Every partition must cover all four streets')
        report['status'] = 'FITTING'
        candidates = {}
        for name, d in data.items():
            for seed in protocol.SEEDS:
                check()
                warm, rows, models = fit_dataset(d, encoder, seed, check, smoke=smoke)
                report['warmups'].append(dict(dataset=name, seed=seed, **warm))
                for row in rows:
                    row['dataset'] = name
                    report['results'].append(row)
                    candidates[(name, seed, row['arm'])] = models[row['arm']]
                write_json(root/'report.json', report); write_summary(root, report)
                print(json.dumps(dict(dataset=name, seed=seed, status='FITTED', elapsed_seconds=time.monotonic()-started)), flush=True)
        # Commit every per-fit budget before opening any audit metrics. No winner chosen on audit.
        write_json(root/'selection.json', dict(rule='Minimum validation target error per arm/seed; update zero eligible; earliest tie wins.',
            selections=[{k:r[k] for k in ('dataset', 'seed', 'arm', 'selected_update')} for r in report['results']]))
        for row in report['results']:
            check()
            d = data[row['dataset']]
            row['audit'] = scores(candidates[(row['dataset'], row['seed'], row['arm'])], d, d['partitions'][2], decomposition=True)
        report['status'] = 'COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(root/'report.json', report); write_summary(root, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'policy', 'captures', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--max-seconds', type=float, default=5400)
    p.add_argument('--smoke', action='store_true')
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
