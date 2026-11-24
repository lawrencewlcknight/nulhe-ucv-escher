# Fresh 32-hour Limit Hold’em hybrid experiment

Train fresh regret and average-policy networks with a wide initial representation,
deep residual refinement and a direct wide-to-output connection. No weights,
replay, optimizer state or RNG state are inherited from earlier runs.

## Versioned network

Each player regret network and the average-policy network has:

1. The existing exact-card, context and strategic branches, each 128 wide,
   concatenated in that order (384 features).
2. Linear 384→512 + ReLU.
3. A wide residual block: `x + 0.5 * F(x)`, with F equal to
   LayerNorm512 → ReLU → Linear512→256 → LayerNorm256 → ReLU → Linear256→512.
4. Linear512→256 + ReLU.
5. Four residual blocks: `x + 0.5 * F(x)`, with F equal to
   LayerNorm256 → ReLU → Linear256→256 → LayerNorm256 → ReLU → Linear256→256.
6. LayerNorm256 → ReLU → Linear256→256 + ReLU.
7. Concatenate the wide block output (512) and deep readout (256), followed by
   a zero-initialized linear768→3 action head.

There is no activation after residual addition. LayerNorm has affine parameters
and epsilon 1e-5. Hidden layers retain Sonnet truncated-normal fan-in initialization
and zero biases. Resetting a policy fit resets every linear and LayerNorm once.
Regrets remain signed, policy scores remain logits, and existing legal-action
handling is unchanged. No dropout or BatchNorm is introduced.

There are **1,245,699 parameters per network**, versus 749,827 for Deep and
777,731 for Wide. The longest trunk path has 13 hidden linear layers, excluding
input branches and action output. Version `hybrid_residual_v1` fixes all widths,
normalization, residual scaling and concatenation order. Export loading rejects
semantic metadata mismatches. The inherited regret config's `num_hiddens=256`
describes the deep stage; the architecture selector supplies the mixed widths.

Critics/calibration, feature encoder, learning rates, update counts, traversals,
12M policy replay and validation schedule match deep32. Corrected
`objective_mass` policy-group sampling is used from initialization, with sampling
seeds `[training_seed, iteration]`. Training seed is 20261115.

## Training and safety boundaries

- 32 **active learner hours**, saving playable P8/P14/P20/P26/P32 policies.
  Boundaries round up to a completed iteration. Policy extraction, persistence,
  transfers and evaluations add wall time beyond those 32 hours.
- Google Batch in `clever-overview-399515`, `europe-west1`, using
  `c4-standard-16`, eight collection workers, eight training threads and a 200 GB
  disk. Standard provisioning, no automatic task retries or VM upsizing.
- Before training, verify checksum-pinned comparison inputs and build the pinned
  Fell Omen 2 Java bridge. No historical replay is downloaded.
- Run the existing isolated full-replay memory/fit/recovery preflight. Require
  at least max(8 GiB, 12% of RAM) observed headroom. Synthetic preflight replay
  never becomes training data. The preflight also exercises real collection and
  fitting; training reports measure actual throughput and fitting overhead.
- Two-hour preflight guard, 40-hour training wall guard, 12-hour evaluation
  envelope, **56-hour Batch ceiling** including setup and transfers.
- Six-wall-hour local recovery cadence, suppressed if a recovery was saved within
  the previous hour. Each playable checkpoint boundary verifies a full restore
  and exact agreement with the exported policy before cloud publication.
- Final P32 full state is resumable, including replay, optimizers and RNG/worker
  state. Recovery requires the exact archived source, runtime and configuration.
  Existing verified publication owns cloud recovery pruning. Periodic local
  recovery is uploaded at successful stage boundaries; interruption can lose
  work since the last uploaded boundary.

## Predeclared evaluations

The final benchmark uses fresh duplicate deals with seats swapped:

- Primary: **Hybrid-P32 versus Deep-P32**, 95% confidence interval.
- Two secondary comparisons: Hybrid-P32 versus corrected smaller P32, and versus
  Fell Omen 2, each with a 97.5% interval (Bonferroni within this secondary family).
- Corrected smaller P32 includes **both fit seeds 1701 and 1702**, equally
  weighted within each deal group; neither seed is selected and they are not
  combined into an ensemble policy.
- Each shared group comprises eight hands: one duplicate pair against Deep,
  each corrected refit and FO2. Average the two corrected-refit results within
  the group before calculating its uncertainty.
- Exclude a 10,000-group pilot. Select a common confirmation size using the
  largest of the three variance-based requirements, targeting estimated 80%
  power for 2 bb/100, bounded at 100,000–1,000,000 groups. The selected sample
  is not reduced to fit runtime. Insufficient time or incomplete confirmation
  cannot yield a confirmatory success claim. Report capped power explicitly.
- The final benchmark has nine hours of the evaluation envelope.

The remaining evaluation envelope allows 2h50 for P32/P8 and four adjacent
checkpoint comparisons, with alpha .01 each in a separate progression family,
using the existing excluded-pilot/runtime-capped evaluator. All pairs use fresh
evaluation seeds. The last ten minutes run the same fixed development strategic
library for Hybrid-P32, Deep-P32 and Wide-P32, including privacy/suit invariance
checks. Assessment positions remain locked. Audit results do not select a policy.

The primary, secondary and progression families are reported separately; there
is no claim of 5% familywise error across all eight tests together. Intervals
condition on these particular trained policies, not training-seed variability.
The corrected smaller reference used a different historical training recipe;
that comparison does not isolate architecture. Equal active hours also do not
imply equal samples or FLOPs. Strategic gaps are not whole-game exploitability.

## Prepare or launch

From the repository, omit `--execute` to prepare a reviewable local bundle only:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-hybrid32-batch-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_hybrid_scaling_batch.py \
  --run-name "$RUN" --output "outputs/${RUN}-launch" --execute
```

This submits an independent cloud job. No open local terminal is required after
submission. The launcher pins and archives the local source with an explicit
allowlist excluding secrets, outputs and Git data. Immutable output-prefix claims
prevent accidental reuse. Batch releases its VM and disk when the task ends;
published GCS artifacts remain.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name:hulhe-hybrid32' --format='table(name.basename(),status.state)'
```

## Evaluation-only rerun

If training completed but evaluation failed, use the **original hybrid training
launch.json**, never an evaluation-only plan. This downloads the five playable
policies and verified reference inputs, without training or replay restoration.
It uses a 40 GB disk and a 14-hour Batch ceiling.

```bash
EVAL="hulhe-hybrid32-eval-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_hybrid_scaling_batch.py \
  --run-name "$EVAL" --output "outputs/${EVAL}-launch" \
  --training-plan "outputs/${RUN}-launch/launch.json" --execute
```

Fixed evaluation seeds reproduce the protocol; rerunning is not an independent
replication or permission to pool results. Inspect incomplete-stage reports first.

## Download analysis only

Set RUN to the actual submitted training or evaluation job name:

```bash
mkdir -p "outputs/${RUN}-results"
gcloud storage rsync --recursive \
  --exclude '(^|/)(state_[0-9]+|state)(/|$)|^(inputs|fo2-assets)/|^(?!.*\.json$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/${RUN}" \
  "outputs/${RUN}-results"
```

This excludes model weights, replay, source archives, logs and raw hand journals.
Read `scaling.json`, `evaluation.json`, `benchmarks/report.json`,
`evaluation/*/report.json`, `strategic-audit/summary.json`, `memory-gate.json`
and `reload-P*.json`. Partial evaluation remains explicit even when training
was successful and its checkpoints were saved.

## Local validation

```bash
python3 -m pytest tests/test_hybrid_scaling.py tests/test_scaling_experiment.py \
  tests/test_deep_scaling.py tests/test_scaling_corrected.py
python3 -m hulhe_escher.hybrid_scaling_experiment \
  --output /tmp/hulhe-hybrid32-smoke --smoke
```

The offline smoke uses full-size hybrid networks with tiny replay/update budgets
and exercises all five training/save/resume/export/reload boundaries. It neither
downloads controls nor evaluates FO2. Tests separately exercise real spawned
evaluation workers with mixed architectures and a simple offline opponent.
Smoke success does not establish production RAM sufficiency or playing strength.
