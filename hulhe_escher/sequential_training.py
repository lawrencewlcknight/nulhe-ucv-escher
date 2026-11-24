"""Protected selective fitting v1. No changes to baseline learners by default."""
from copy import deepcopy

import numpy as np
import torch

from .config import baseline_config, smoke_config, validate_config

PROTECTION = .3
WARMUP_UPDATES = 200
REFERENCE_REFRESH = 50
COMPARISON_CAVEAT = (
    'Same 4+4 active-hour/checkpoint/evaluation protocol as the original learning check. '
    'New sequential architecture, protected fitting, policy LR 0.0003 and corrected '
    'objective-mass sampling differ from the historical recipe. This is not a capacity-only '
    'ablation. Equal hours are not equal samples or FLOPs; one training seed is not '
    'evidence across independent runs. Limit and no-limit scores are different games.'
)


def sequential_config(*, small=False):
    config = smoke_config() if small else baseline_config()
    config.update(num_layers=1, num_hiddens=384, average_policy_network_layers=[384],
                  structured_branch_width=128, critic_branch_width=96,
                  regret_policy_architecture='protected_sequential_v1',
                  policy_group_sampling='objective_mass', average_policy_learning_rate=.0003)
    validate_config(config)
    return config


class StreetSampler:
    """Sample street by mass, then iid examples conditional on that street.

    CDFs are built once per fit; no full replay scan per update. Primary draws
    estimate the same weighted objective, including its original loss scale.
    Auxiliary draws use the objective conditional on the other streets.
    """
    def __init__(self, features, ids, columns, weights, rng):
        self.rng = rng
        ids = np.asarray(ids, dtype=np.int64)
        streets = np.empty(len(ids), dtype=np.int8)
        for start in range(0, len(ids), 4096):
            block = features[torch.as_tensor(ids[start:start+4096])] if not isinstance(features, np.ndarray) else features[ids[start:start+4096]]
            block = np.asarray(block) if isinstance(block, np.ndarray) else block.detach().cpu().numpy()
            values = block[:, columns]
            if not np.all((values == 0) | (values == 1)) or not np.all(values.sum(1) == 1):
                raise ValueError('Protected training requires valid one-hot streets')
            streets[start:start+len(block)] = values.argmax(1)
        mass = np.ones(len(ids), dtype=np.float64) if weights is None else weights[torch.as_tensor(ids, device=weights.device)].detach().cpu().numpy().astype(np.float64)
        if not len(ids) or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError('Training mass must be positive and finite')
        self.ids, self.cdf, self.mass = [], [], np.zeros(4)
        for street in range(4):
            take = streets == street
            self.ids.append(ids[take])
            cdf = np.cumsum(mass[take])
            self.mass[street] = cdf[-1] if len(cdf) else 0
            self.cdf.append(cdf / cdf[-1] if len(cdf) else cdf)
        self.factor = float(mass.sum()/len(ids))

    def street(self, exclude=None):
        mass = self.mass.copy()
        if exclude is not None:
            mass[exclude] = 0
        if not mass.sum():
            return None
        return int(self.rng.choice(4, p=mass/mass.sum()))

    def draw(self, n, street=None, exclude=None):
        if street is not None:
            return self.ids[street][np.searchsorted(self.cdf[street], self.rng.random(n), side='right')]
        mass = self.mass.copy()
        if exclude is not None:
            mass[exclude] = 0
        if not mass.sum():
            return np.empty(0, dtype=np.int64)
        choices = self.rng.choice(4, size=n, p=mass/mass.sum())
        result = np.empty(n, dtype=np.int64)
        for s in range(4):
            take = choices == s
            if take.any():
                result[take] = self.draw(int(take.sum()), street=s)
        return result


class ProtectedFit:
    def __init__(self, trainer, features, targets, masks, ids, *, rng, kind, weights=None):
        self.trainer, self.features, self.targets, self.masks = trainer, features, targets, masks
        self.kind = kind
        self.sampler = StreetSampler(features, ids, trainer.feature_encoder.replay_context_columns()[1], weights, rng)
        self.batch = min(trainer.batch_size, len(ids))
        self.aux_batch = max(1, self.batch // 4)
        self.warmup = min(WARMUP_UPDATES, max(0, trainer.train_steps - 1))
        self.teacher = None
        self.street_updates = [0] * 4
        self.primary_sum = self.aux_sum = 0.
        self.updates = self.aux_updates = 0

    def tensors(self, ids):
        idx = torch.as_tensor(ids, device=self.trainer.device)
        x = (torch.as_tensor(self.features[ids], device=self.trainer.device)
             if isinstance(self.features, np.ndarray) else self.features[idx])
        return x, self.targets[idx], self.masks[idx]

    def loss(self, prediction, target, mask):
        if self.kind == 'regret':
            return ((prediction * mask - target).square()).mean()
        return -(target * torch.log_softmax(prediction.masked_fill(mask != 1, -1e20), -1)).sum(-1).mean()

    def update(self, step):
        trainer = self.trainer
        street = None if step <= self.warmup else self.sampler.street()
        trainer.model.trainable_street(street)
        # All optimizer parameters remain registered. None prevents Adam momentum
        # or weight decay from moving inactive specialists.
        trainer.optimizer.zero_grad(set_to_none=True)
        if street is not None:
            self.street_updates[street] += 1
        x, target, mask = self.tensors(self.sampler.draw(self.batch, street=street))
        primary = self.loss(trainer.model(x), target, mask) * self.sampler.factor
        auxiliary = primary.new_zeros(())
        if street is not None:
            if self.kind == 'policy' and (self.teacher is None or (step-self.warmup-1) % REFERENCE_REFRESH == 0):
                if self.teacher is None:
                    self.teacher = deepcopy(trainer.model).eval()
                    self.teacher.requires_grad_(False)
                else:
                    self.teacher.load_state_dict(trainer.model.state_dict())
            ids = self.sampler.draw(self.aux_batch, exclude=street)
            if len(ids):
                ax, ay, am = self.tensors(ids)
                pred = trainer.model(ax)
                if self.kind == 'policy':
                    with torch.no_grad():
                        ref = self.teacher(ax)
                    logp = torch.log_softmax(pred.masked_fill(am != 1, -1e20), -1)
                    logq = torch.log_softmax(ref.masked_fill(am != 1, -1e20), -1)
                    auxiliary = (logq.exp() * (logq-logp)).sum(-1).mean()
                else:
                    auxiliary = self.loss(pred, ay, am)
                auxiliary = auxiliary * self.sampler.factor
                self.aux_updates += 1
        loss = primary + PROTECTION * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite protected training loss')
        loss.backward()
        if self.kind == 'regret':
            trainer._clip_gradients()
        else:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 10., error_if_nonfinite=True)
        trainer.optimizer.step()
        self.updates += 1
        self.primary_sum += float(primary.detach())
        self.aux_sum += float(auxiliary.detach())
        return loss.detach()

    def finish(self):
        self.trainer.model.trainable_street(None)
        self.teacher = None  # Reference is local to this fit, never a production checkpoint.

    def diagnostics(self):
        return dict(protocol='protected_selective_v1', coefficient=PROTECTION,
            warmup_updates=self.warmup, specialist_updates_by_street=self.street_updates,
            auxiliary_updates=self.aux_updates, auxiliary_batch_size=self.aux_batch,
            reference_refresh_updates=REFERENCE_REFRESH if self.kind == 'policy' else None,
            primary_mean=self.primary_sum/max(1, self.updates),
            auxiliary_mean=self.aux_sum/max(1, self.updates),
            policy_preservation='refreshed_forward_KL' if self.kind == 'policy' else 'current_target_rehearsal')
