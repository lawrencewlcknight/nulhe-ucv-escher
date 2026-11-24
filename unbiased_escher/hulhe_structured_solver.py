"""Lossless suit-canonical HULHE variant of grouped-wide UCV-ESCHER."""

from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import torch

from hulhe_escher.optimization import ReplayScaledFitMixin
from hulhe_escher.features import (
    HULHEFeatureEncoder,
    ENCODER_ID,
    make_feature_encoder,
    FeatureLayout,
    StructuredHULHEMLP,
)
from vr_deep_cfr.solver import AvePolicyTrainer
from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer

from .grouped_wide_solver import (
    GROUPED_SOFT_TARGET_CROSS_ENTROPY,
    GroupedSoftTargetCrossEntropyAvePolicyTrainer,
    GroupedWideUnbiasedControlVariateEscher,
    TemporallyAveragedCrossFittedQEnsemble,
    TemporallyAveragedCrossFittedQMember,
)
from .holdem_fitting import RegretFitMixin, fit_validated_policy
from .efficient_replay import CompactCircularBuffer, CompactReservoirBuffer
from .solver import ResidualCalibrationTrainer, CalibrationBuffer
class _StructuredModelMixin:
    feature_layout: FeatureLayout
    structured_branch_width: int

    def init_model(self):
        if getattr(self, "regret_policy_architecture", "plain") == "protected_sequential_v1":
            from hulhe_escher.sequential_model import ProtectedSequentialNetwork
            return ProtectedSequentialNetwork(self.feature_layout).to(self.device)
        model_class = StructuredHULHEMLP
        layers = self.network_layers
        if getattr(self, "regret_policy_architecture", "plain") == "deep_residual_v1":
            from hulhe_escher.deep_model import DeepResidualHULHEMLP
            model_class = DeepResidualHULHEMLP
        elif getattr(self, "regret_policy_architecture", "plain") == "hybrid_residual_v1":
            from hulhe_escher.hybrid_model import HybridResidualHULHEMLP, HYBRID_LAYERS
            # The inherited regret trainer represents trunks with a uniform
            # width/depth descriptor; this versioned selector supplies actual widths.
            if list(layers) not in ([256] * 13, HYBRID_LAYERS):
                raise ValueError('Invalid hybrid network descriptor')
            model_class, layers = HybridResidualHULHEMLP, HYBRID_LAYERS
        return model_class(
            self.feature_layout,
            layers,
            self.output_size,
            branch_width=self.structured_branch_width,
        ).to(self.device)


class StructuredHULHERegretTrainer(RegretFitMixin, _StructuredModelMixin, VRDCFRPlusRegretTrainer):
    def __init__(
        self,
        *args,
        feature_encoder: HULHEFeatureEncoder,
        branch_width: int,
        replay_seed: int,
        regret_policy_architecture="plain",
        gradient_clip_norm=None,
        cache_frozen_targets: bool = True,
        uniform_regret_ties: bool = True,
        **kwargs,
    ):
        self.regret_policy_architecture = regret_policy_architecture
        self.feature_encoder = feature_encoder
        self.feature_layout = feature_encoder.policy_layout
        self.structured_branch_width = int(branch_width)
        self.replay_seed = int(replay_seed)
        self.gradient_clip_norm = gradient_clip_norm
        self.cache_frozen_targets = cache_frozen_targets
        self.uniform_regret_ties = uniform_regret_ties
        super().__init__(*args, **kwargs)
        self.predictor_enabled = False

    def get_infostate_tensor(self, state, player=None):
        return self.feature_encoder.information_state(state, player)

    def init_buffer(self):
        return CompactReservoirBuffer(
            self.buffer_size,
            self.input_size,
            self.output_size,
            device=self.device,
            seed=self.replay_seed,
        )


class StructuredHULHEAveragePolicyTrainer(
    _StructuredModelMixin, GroupedSoftTargetCrossEntropyAvePolicyTrainer
):
    def __init__(
        self,
        *args,
        feature_encoder: HULHEFeatureEncoder,
        branch_width: int,
        replay_seed: int,
        regret_policy_architecture="plain",
        validation_fraction: float = 0.05,
        validation_steps=(2500, 5000, 10000, 20000),
        policy_group_sampling="uniform",
        policy_sampling_seed=0,
        **kwargs,
    ):
        self.regret_policy_architecture = regret_policy_architecture
        self.feature_encoder = feature_encoder
        self.feature_layout = feature_encoder.policy_layout
        self.structured_branch_width = int(branch_width)
        self.replay_seed = int(replay_seed)
        self.validation_fraction = validation_fraction
        self.validation_steps = tuple(validation_steps)
        self.policy_group_sampling = policy_group_sampling
        self.training_seed = int(policy_sampling_seed)
        super().__init__(*args, **kwargs)

    def train_model(self, iteration):
        # Match the P104 archived correction: independent sampling for each fit,
        # reproducible across process exits and restored outer iterations.
        self.policy_sampling_seed = [self.training_seed, int(iteration)]
        loss = fit_validated_policy(self, iteration)
        self.last_fit_diagnostics["policy_sampling_seed"] = self.policy_sampling_seed
        return loss

    def get_infostate_tensor(self, state, player=None):
        return self.feature_encoder.information_state(state, player)

    def init_buffer(self):
        return CompactReservoirBuffer(
            self.buffer_size,
            self.input_size,
            self.output_size,
            device=self.device,
            seed=self.replay_seed,
        )


class StructuredHULHEQMember(
    ReplayScaledFitMixin, _StructuredModelMixin, TemporallyAveragedCrossFittedQMember
):
    def __init__(
        self,
        *args,
        feature_encoder: HULHEFeatureEncoder,
        branch_width: int,
        replay_seed: int,
        minimum_train_steps: int,
        **kwargs,
    ):
        self.feature_encoder = feature_encoder
        self.feature_layout = feature_encoder.full_state_layout
        self.structured_branch_width = int(branch_width)
        self.replay_seed = int(replay_seed)
        self.minimum_train_steps = int(minimum_train_steps)
        super().__init__(*args, **kwargs)

    def get_history_tensor(self, state):
        return self.feature_encoder.full_state(state)

    def init_buffer(self):
        return CompactCircularBuffer(
            self.buffer_size,
            self.input_size,
            self.state_size,
            self.output_size,
            device=self.device,
            seed=self.replay_seed,
        )


class StructuredHULHEQEnsemble(TemporallyAveragedCrossFittedQEnsemble):
    def __init__(
        self,
        *,
        feature_encoder: HULHEFeatureEncoder,
        branch_width: int,
        replay_seed: int,
        target_average_window: int,
        ensemble_size: int,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers,
        learning_rate: float,
        total_buffer_size: int,
        batch_size: int,
        train_steps: int,
        minimum_train_steps: int,
        logger,
        regret_trainers,
        device: str,
        gradient_clip_norm: float,
    ):
        self.ensemble_size = int(ensemble_size)
        self.target_average_window = int(target_average_window)
        if self.ensemble_size < 1 or self.target_average_window < 1:
            raise ValueError("Ensemble size and target-average window must be positive")
        member_buffer_size = max(1, int(total_buffer_size) // self.ensemble_size)
        self.members = [
            StructuredHULHEQMember(
                history_size,
                state_size,
                action_size,
                network_layers,
                learning_rate,
                member_buffer_size,
                batch_size,
                train_steps,
                logger,
                regret_trainers,
                device,
                gradient_clip_norm=gradient_clip_norm,
                target_average_window=self.target_average_window,
                minimum_train_steps=minimum_train_steps,
                feature_encoder=feature_encoder,
                branch_width=branch_width,
                replay_seed=int(replay_seed) + index,
            )
            for index in range(self.ensemble_size)
        ]
        self.active_fold = 0
        self._active_add_count = 0


class StructuredHULHEResidualCalibrationTrainer(ReplayScaledFitMixin, ResidualCalibrationTrainer):
    def __init__(
        self,
        *,
        infostate_size: int,
        action_size: int,
        hidden_layers,
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        minimum_train_steps: int,
        device: str,
        minimum_variance: float,
        feature_encoder: HULHEFeatureEncoder,
        branch_width: int,
    ):
        self.feature_encoder = feature_encoder
        self.minimum_train_steps = int(minimum_train_steps)
        super().__init__(
            infostate_size=infostate_size,
            action_size=action_size,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            train_steps=train_steps,
            device=device,
            minimum_variance=minimum_variance,
        )
        self.feature_encoder = feature_encoder
        self.feature_layout = FeatureLayout(
            "residual_calibration",
            self.feature_size,
            feature_encoder.policy_layout.card_size,
            feature_encoder.policy_layout.strategic_size,
        )
        hidden_layers = list(hidden_layers)
        self.model = StructuredHULHEMLP(
            self.feature_layout,
            hidden_layers,
            2,
            branch_width=int(branch_width),
        ).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(learning_rate)
        )


class StructuredHULHEGroupedWideUCVEscher(GroupedWideUnbiasedControlVariateEscher):
    """Selected UCV learner with four-street HULHE neural features."""

    def __init__(
        self,
        *args,
        structured_branch_width: int = 64,
        critic_branch_width: int | None = None,
        regret_policy_architecture: str = "plain",
        policy_group_sampling: str = "uniform",
        critic_network_layers=None,
        calibration_network_layers=None,
        calibration_branch_width: int | None = None,
        critic_min_train_steps: int = 250,
        calibration_min_train_steps: int = 100,
        feature_encoder_id: str = ENCODER_ID,
        cache_frozen_regret_targets: bool = True,
        uniform_regret_ties: bool = True,
        policy_validation_fraction: float = 0.05,
        policy_validation_steps=(2500, 5000, 10000, 20000),
        **kwargs,
    ):
        if regret_policy_architecture not in ("plain", "deep_residual_v1", "hybrid_residual_v1", "protected_sequential_v1"):
            raise ValueError("Unknown regret/policy architecture")
        self.regret_policy_architecture = regret_policy_architecture
        self.feature_encoder = make_feature_encoder(feature_encoder_id)
        self.critic_min_train_steps = int(critic_min_train_steps)
        self.calibration_min_train_steps = int(calibration_min_train_steps)
        self.cache_frozen_regret_targets = cache_frozen_regret_targets
        self.uniform_regret_ties = uniform_regret_ties
        self.policy_validation_fraction = policy_validation_fraction
        self.policy_validation_steps = tuple(policy_validation_steps)
        self.structured_branch_width = int(structured_branch_width)
        self.critic_branch_width = (self.structured_branch_width if critic_branch_width is None
                                    else int(critic_branch_width))
        if policy_group_sampling not in ("uniform", "objective_mass"):
            raise ValueError("Unknown policy group sampler")
        self.policy_group_sampling = policy_group_sampling
        # Optional overrides preserve the inherited architecture for callers
        # using the old constructor. The versioned baseline sets each explicitly.
        self.critic_network_layers = (
            None if critic_network_layers is None else list(critic_network_layers)
        )
        self.calibration_network_layers = (
            None if calibration_network_layers is None else list(calibration_network_layers)
        )
        self.calibration_branch_width = (
            self.structured_branch_width if calibration_branch_width is None
            else int(calibration_branch_width)
        )
        self.feature_replay_seed = int(kwargs.get("seed", 0))
        if min(self.structured_branch_width, self.calibration_branch_width, self.critic_branch_width) <= 0:
            raise ValueError("Structured branch widths must be positive")
        super().__init__(*args, **kwargs)
        if self.average_policy_loss != GROUPED_SOFT_TARGET_CROSS_ENTROPY:
            raise ValueError("The structured experiment requires grouped policy CE")

    def init_ave_policy_trainer(self):
        # DeepCumuAdv initially discovers OpenSpiel's raw size. From this hook
        # onwards every learner consistently uses the versioned HULHE encoder.
        self.raw_open_spiel_infostate_size = int(self.infostate_size)
        self.infostate_size = self.feature_encoder.policy_size

        # Preserve Experiment 35's legacy RNG anchor so the algorithmic change
        # is limited to representation and structured model capacity.
        legacy_rng_anchor = AvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            list(self.paired_legacy_network_layers),
            self.paired_legacy_learning_rate,
            1,
            self.ave_policy_batch_size,
            self.paired_legacy_train_steps,
            self.logger,
            self.device,
            self.gamma,
        )
        rng_after_legacy = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_after_legacy["cuda"] = torch.cuda.get_rng_state_all()
        self.ave_policy_trainer = StructuredHULHEAveragePolicyTrainer(
            self.infostate_size,
            self.action_size,
            list(self.average_policy_network_layers),
            self.average_policy_learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.average_policy_train_steps,
            self.logger,
            self.device,
            self.gamma,
            feature_encoder=self.feature_encoder,
            branch_width=self.structured_branch_width,
            replay_seed=self.feature_replay_seed + 100,
            validation_fraction=self.policy_validation_fraction,
            validation_steps=self.policy_validation_steps,
            policy_group_sampling=self.policy_group_sampling,
            policy_sampling_seed=self.feature_replay_seed,
            regret_policy_architecture=self.regret_policy_architecture,
        )
        random.setstate(rng_after_legacy["python"])
        np.random.set_state(rng_after_legacy["numpy"])
        torch.random.set_rng_state(rng_after_legacy["torch"])
        if "cuda" in rng_after_legacy:
            torch.cuda.set_rng_state_all(rng_after_legacy["cuda"])
        del legacy_rng_anchor

    def init_regret_trainers(self):
        if self.use_instantaneous_predictor:
            raise ValueError("Structured HULHE requires non-predictive UCV")
        self.regret_trainers = [
            StructuredHULHERegretTrainer(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.use_regret_matching_argmax,
                self.device,
                self.alpha,
                feature_encoder=self.feature_encoder,
                branch_width=self.structured_branch_width,
                replay_seed=self.feature_replay_seed + 200 + player,
                gradient_clip_norm=self.regret_policy_gradient_clip_norm,
                cache_frozen_targets=self.cache_frozen_regret_targets,
                uniform_regret_ties=self.uniform_regret_ties,
                regret_policy_architecture=self.regret_policy_architecture,
            )
            for player in range(self.num_players)
        ]

    def init_q_value_trainer(self):
        self.q_value_trainer = StructuredHULHEQEnsemble(
            feature_encoder=self.feature_encoder,
            branch_width=self.critic_branch_width,
            replay_seed=self.feature_replay_seed + 300,
            target_average_window=self.critic_target_average_window,
            ensemble_size=self.q_ensemble_size,
            history_size=self.feature_encoder.full_state_size,
            state_size=self.feature_encoder.policy_size,
            action_size=self.action_size,
            network_layers=(self.network_layers if self.critic_network_layers is None
                            else self.critic_network_layers),
            learning_rate=self.learning_rate,
            total_buffer_size=self.baseline_buffer_size,
            batch_size=self.baseline_batch_size,
            train_steps=self.baseline_network_train_steps,
            minimum_train_steps=self.critic_min_train_steps,
            logger=self.logger,
            regret_trainers=self.regret_trainers,
            device=self.device,
            gradient_clip_norm=self.q_gradient_clip_norm,
        )

    def init_calibration_trainer(self):
        self.calibration_trainer = StructuredHULHEResidualCalibrationTrainer(
            infostate_size=self.infostate_size,
            action_size=self.action_size,
            hidden_layers=(self.network_layers if self.calibration_network_layers is None
                           else self.calibration_network_layers),
            learning_rate=self.calibration_learning_rate,
            buffer_size=self.calibration_buffer_size,
            batch_size=self.calibration_batch_size,
            train_steps=self.calibration_train_steps,
            minimum_train_steps=self.calibration_min_train_steps,
            device=self.device,
            minimum_variance=self.calibration_minimum_variance,
            feature_encoder=self.feature_encoder,
            branch_width=self.calibration_branch_width,
        )

    def get_infostate_tensor(self, state, player=None):
        return self.feature_encoder.information_state(state, player)

    def get_history_tensor(self, state):
        return self.feature_encoder.full_state(state)

    def evaluate(self, **kwargs):
        self.logger.record("feature_encoder_id", self.feature_encoder.encoder_id)
        self.logger.record("policy_feature_size", self.feature_encoder.policy_size)
        self.logger.record("critic_feature_size", self.feature_encoder.full_state_size)
        self.logger.record("raw_open_spiel_feature_size", self.raw_open_spiel_infostate_size)
        replay_bytes = sum(trainer.buffer.nbytes() for trainer in self.regret_trainers)
        replay_bytes += self.ave_policy_trainer.buffer.nbytes()
        replay_bytes += sum(member.buffer.nbytes() for member in self.q_value_trainer.members)
        self.logger.record("primary_replay_allocated_bytes", replay_bytes)
        return super().evaluate(**kwargs)


__all__ = [
    "StructuredHULHEGroupedWideUCVEscher",
    "StructuredHULHEAveragePolicyTrainer",
    "StructuredHULHEQEnsemble",
    "StructuredHULHERegretTrainer",
]
