"""Independent completed-iteration schedules, measured in active learner time."""
from dataclasses import dataclass, asdict


@dataclass
class CheckpointSchedule:
    last_recovery_seconds: float = 0.0
    last_policy_seconds: float = 0.0

    def due(self, *, iteration, final, active_seconds, recovery_seconds,
            policy_seconds, legacy_every=None):
        legacy = legacy_every is not None and iteration % legacy_every == 0
        policy = final or legacy or active_seconds - self.last_policy_seconds >= policy_seconds
        recovery = final or legacy or active_seconds - self.last_recovery_seconds >= recovery_seconds
        return recovery, policy

    def completed(self, active_seconds, *, recovery, policy):
        if recovery:
            self.last_recovery_seconds = active_seconds
        if policy:
            self.last_policy_seconds = active_seconds

    def state(self):
        return asdict(self)
