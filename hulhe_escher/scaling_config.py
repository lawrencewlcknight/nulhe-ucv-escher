"""Fresh width-scaling experiment; production continuation defaults stay intact."""
from .config import baseline_config, smoke_config, validate_config


def scaling_config(*, small=False):
    config = smoke_config() if small else baseline_config()
    config.update(num_layers=3, num_hiddens=512,
                  average_policy_network_layers=[512, 512, 512],
                  structured_branch_width=128, critic_branch_width=96,
                  policy_group_sampling="objective_mass")
    if not small:
        config.update(ave_policy_buffer_size=12_000_000,
                      policy_validation_steps=[500, 1000, 2500, 5000, 10000, 20000])
    validate_config(config)
    return config


CHECKPOINT_HOURS = (8, 14, 20, 26, 32)
HISTORICAL_P32_URI = "gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535"
HISTORICAL_P32_SHA256 = "9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b"
COMPARISON_CAVEAT = (
    "Historical P32 used legacy uniform-group sampling; the fresh scaled run uses "
    "objective_mass from initialization, wider networks, larger policy replay and earlier "
    "validation candidates. This is a combined recipe comparison, not a capacity-only "
    "ablation. Equal active hours do not imply equal iterations, nodes or FLOPs. "
    "One training seed cannot establish generalization across training runs."
)
