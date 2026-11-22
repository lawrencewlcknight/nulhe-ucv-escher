# Fresh HULHE deep residual experiment

This experiment is **fixed-limit Hold’em**, in `hulhe-ucv-escher`. It compares
fresh deep residual training with the fresh wide32 experiment, not historical
small-network P32 and not HUNL. Both runs start from scratch with the corrected
P104 `objective_mass` policy-group sampler, seeded by `[training_seed, iteration]`.

## Architecture and controlled settings

Each of the two regret networks and the average-policy network uses:

1. Existing exact-card, strategic-feature and betting-context branches, each 128 wide.
2. Concatenate the three branches, then Linear 384 → 256 + ReLU (hidden layer 1).
3. Four blocks of `x + 0.5 * F(x)`, where F is
   `LayerNorm → ReLU → Linear 256→256 → LayerNorm → ReLU → Linear 256→256`
   (hidden layers 2–9). No activation follows the addition.
4. LayerNorm → ReLU → Linear 256→256 → ReLU (hidden layer 10).
5. Existing zero-initialized linear 256→3 output. Regrets remain signed;
   policy outputs remain logits with the existing legal-action mask.

LayerNorm uses affine parameters and epsilon 1e-5; there are nine norms.
Hidden linears retain the existing truncated-normal fan-in initialization,
with biases; residual paths are nonzero at initialization. All parameters,
including normalization gain/bias, reset during a fresh policy fit.

The architecture identifier `deep_residual_v1` and checkpoint type
`structured_hulhe_deep_residual_v1` fix these semantics. Playable reload rejects
mismatched architecture metadata. Old plain/diagnostic policies remain supported.
Full-state checkpoints persist replay, optimizers, RNGs and parallel worker state.

There are **749,827 parameters per regret/policy network**, versus 777,731 in
wide32 (3.59% fewer). The critics remain 192×2 with branches 96; calibration
remains 128×2 with branches 64. All other training settings equal `scaling_config()`:
12M policy replay, learning rates, update counts, traversal budget, validation
schedule, encoder, game and seed 20261115. The only config differences are trunk
width/depth and the explicit residual architecture selector.

This is an architecture-recipe comparison (depth, residuals and normalization),
not a depth-only ablation. Equal parameter counts do not imply equal latency,
activation memory, samples or FLOPs. One seed supports run-specific conclusions.

## Training and evaluation

- Fresh 32 active learner hours; checkpoints P8/P14/P20/P26/P32, rounded up to
  completed iterations. Policy fits, persistence and transfers are outside active time.
- `c4-standard-16`, eight collection workers, eight Torch threads, 200 GB disk.
- Full-capacity memory preflight, requiring max(8 GiB, 12% host RAM) headroom.
  Two-hour preflight limit, 40-hour training wall budget, 12-hour evaluation
  envelope, 56-hour Batch task hard stop, standard provisioning and zero retries.
- Same six-wall-hour recovery schedule, suppressed if saved within the last hour.
  Full reload and exact exported-policy verification precede stage publication.
  Old cloud recovery is pruned only by the existing verified publication path.
- Primary: **Deep-P32 versus Wide-P32**, alpha .05.
- Five secondary tests: Deep-P32/Deep-P8 and Deep/Wide at P8/P14/P20/P26,
  alpha .01 each (Bonferroni family .05).
- Independent 10,000 duplicate-pair pilot excluded from results, 100,000–1,000,000
  confirmatory pairs, targeting 80% power for 2 bb/100; seats reversed on each deal.
  Time/sample caps can prevent the requested power. Preserve incomplete and
  insufficient-time reports; completion does not guarantee significance.
- Reserve ten minutes within the evaluation envelope for both final policies on
  the same fixed development strategic-position library, with suit-invariance
  checks. Half the remaining 11h50m goes to the primary match; five equal shares
  of the other half go to secondary matches. The position audit is diagnostic,
  not a general exploitability or independent confirmatory-strength estimate.
- No Slumbot, GTO Wizard, chat schedule or persistent VM.

The wide control is pinned by its launch plan URI and source archive SHA-256.
Before evaluation, verify the control's complete configuration, seed, hardware,
checkpoint schedule, sampler receipts, policy hashes and payload metadata.
The immutable published hashes for every policy are retained in `verified-inputs.json`.
Training is independent of the wide run. If a wide checkpoint marker is absent
when training finishes, write `DEFERRED_WIDE_NOT_READY`, publish results and exit.
There is no idle polling VM and no automatic training restart. Auth/network or
checksum errors fail explicitly instead of masquerading as unfinished training.

Metrics in `scaling.json` and `training/metrics_*.json` retain iterations, nodes,
trajectories, throughput, policy fit and persistence time, replay occupancy,
regret/critic fitting statistics and policy validation CE/selected updates.
Memory and reload reports provide the corresponding resource/recovery checks.

## Prepare and submit training

From `/Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher`:

```bash
python3 tools/launch_deep_scaling_batch.py \
  --run-name hulhe-deep32-batch-20261122 \
  --wide-plan outputs/cloud-hulhe-wide32-batch-20261122/launch.json \
  --output outputs/cloud-hulhe-deep32-batch-20261122
```

This prepares a self-contained source archive and readable Batch plan, without
launching compute. Submit that exact bundle explicitly:

```bash
python3 tools/launch_deep_scaling_batch.py \
  --output outputs/cloud-hulhe-deep32-batch-20261122 \
  --submit-prepared
```

Do not regenerate or reuse a submitted name/prefix. Batch owns VM/disk cleanup;
verified full recovery, all playable policies and exact source remain in storage.
As with wide32, intermediate timed saves are local; failure between stage uploads
can lose work since the most recent uploaded stage boundary.

## Deferred evaluation only

Only if training reports `DEFERRED_WIDE_NOT_READY`, and after the wide policies
are available, prepare a separate bounded job. This fetches only the ten playable
policies and their provenance; it cannot start training or download replay states.

```bash
python3 tools/launch_deep_scaling_batch.py \
  --run-name hulhe-deep32-eval-20261124 \
  --wide-plan outputs/cloud-hulhe-wide32-batch-20261122/launch.json \
  --training-plan outputs/cloud-hulhe-deep32-batch-20261122/launch.json \
  --output outputs/cloud-hulhe-deep32-eval-20261124

python3 tools/launch_deep_scaling_batch.py \
  --output outputs/cloud-hulhe-deep32-eval-20261124 \
  --submit-prepared
```

The evaluation-only job has a 14-hour hard stop (12-hour evaluation plus setup and
transfer allowance), 40 GB disk and zero retries. Its independent output prefix
preserves the original training results. Use a new unique name if a different
date is needed. Inspect existing evaluation reports before submitting again.

## Download analysis only

```bash
mkdir -p outputs/hulhe-deep32-batch-20261122-results
gcloud storage rsync --recursive \
  --exclude '(^|/)(state_[0-9]+|state)(/|$)|^(?!.*\.json$).*' \
  gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-deep32-batch-20261122 \
  outputs/hulhe-deep32-batch-20261122-results
```

This excludes weights, replay, archives and raw hand journals. For a deferred
job, substitute its independent evaluation prefix and destination. Review
`evaluation.json`, each match's `report.json`/`plan.json`,
`strategic-audit/summary.json`, `strategic-audit/checks.json`, `scaling.json`,
`memory-gate.json` and `reload-P*.json`. Partial audits/comparisons remain explicit.

## Offline verification

```bash
python3 -m pytest tests/test_deep_scaling.py tests/test_scaling_experiment.py
python3 -m hulhe_escher.deep_scaling_experiment \
  --output /tmp/hulhe-deep32-smoke --smoke
```

Smoke uses the full residual architecture and tiny replay/update budgets, and
exercises all five save/resume/export/reload boundaries. It is not evidence of
full-scale learning or adequate production RAM. Exact continuation tests compare
uninterrupted/resumed models, optimizers, replay and RNG state. Other tests check
parameter counts, residual gradients, batch-independent inference, policy reload,
unchanged critics, sampler seeds, provenance rejection and bounded Batch plans.
