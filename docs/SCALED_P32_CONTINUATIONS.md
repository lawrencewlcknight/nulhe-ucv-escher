# Wide and Deep P32 +24-hour continuations

Two independent Google Batch experiments resume the full states from the completed
fresh wide32 and deep32 runs. They preserve the architecture, feature encoder,
12M policy reservoir, all other buffers, learning rates, update schedules,
optimizers, models, training RNGs and collection-worker state.

| Experiment | Original run | Initial iteration | New policies |
|---|---|---:|---|
| Wide-P32 +24 | `hulhe-wide32-batch-20261122` | 953 | P38, P44, P50, P56 |
| Deep-P32 +24 | `hulhe-deep32-batch-20261122` | 868 | P38, P44, P50, P56 |

These labels are local to each architecture's experiment. Wide-P56 and Deep-P56
are distinct from the historical smaller-model P56. Each baseline P32 policy is
also copied unchanged into its continuation's output prefix for evaluation.

Both original runs already used corrected `objective_mass` policy-group sampling.
The continuation verifies that correction and its `[training_seed, iteration]`
sampling seed at every export. No legacy-P32 migration, network resizing or
reinitialization of the learner is performed. Average-policy extraction retains
the original recipe's usual fresh policy-fit initialization.

## Recovery and source identity

The production source contracts are pinned in
`hulhe_escher/data/scaled_continuation_sources.json`: original run URI, P32 policy
hash, source archive hash, extracted trainer fingerprint, state-manifest hash
and recovery iteration. Cloud metadata and the two small archived trainers were
verified when implementing these experiments. No full production replay was
downloaded to the developer laptop.

The job checks the requested policy, archived source and checkpoint metadata
before downloading the approximately 28.6 GB full state. It then checks the
complete state using the original archived trainer and exact saved runtime.
Training executes in that archived source directory with pinned imports, while
the new coordinator handles orchestration, publication and evaluation.

Source weights, replay shards and metadata remain unchanged. Training uses an
independent copy under a new cloud prefix. No source fingerprint is rewritten,
and no resume check is bypassed. Full state restoration includes critic targets,
optimizer state, replay/reservoir state and actor RNGs.

The four stage targets are the saved P32 active time plus 6/12/18/24 hours,
rounded up to completed iterations. This provides **24 additional active learner
hours** rather than treating initial P32's small iteration overshoot as free
continuation training. Policy fits, checkpoint verification and transfers add
wall time. All new stage exports get an independent-process full restore and
exact policy verification before cloud publication.

At P56, retain a full recovery and the original `training_source.tar.gz` for
future continuation. Existing publication uploads the new recovery before
updating pointers, then prunes only the previous recovery in the **new run**.
Original runs are protected. Six-wall-hour timed local recovery remains enabled,
with the original one-hour recent-save suppression. Intermediate local recovery
is not independently uploaded between stage boundaries; a VM loss may lose work
since the last published stage.

## Evaluation

After training completes, each job runs the established continuation suite:

- Primary: its own **P56 versus P32**, 95% confidence interval.
- Four adjacent comparisons: P38/P32, P44/P38, P50/P44 and P56/P50.
- P32/P38/P44/P50/P56 against Fell Omen 2 on shared duplicate deals. Estimate
  each later checkpoint's improvement over P32 by subtracting scores within
  each shared deal group.
- The four adjacent tests and four FO2 improvement tests form an eight-test
  secondary family: alpha .05/8, or 99.375% intervals. Confirmatory secondary
  claims require a significant positive primary result. Absolute FO2 scores
  are descriptive, with 95% intervals.
- Excluded 10,000-pair pilot; 100,000–1,000,000 confirmation pairs/groups,
  targeting estimated 80% power for 2 bb/100. Pilot throughput and variance
  determine a fixed runtime-capped sample; report capped power or incomplete
  evaluation explicitly, without significance-based stopping.
- Wide and Deep use different fresh evaluation seed offsets. Each job evaluates
  its own training progression; these jobs do not perform a direct Wide-P56
  versus Deep-P56 comparison or wait for each other.

There is one training seed (20261115). Intervals describe the fixed exported
policies, not variability across independent training runs.

## Cloud resources

Each job uses `c4-standard-16`, eight collection workers, eight training threads
and a 200 GB disk in `europe-west1`, project `clever-overview-399515`.
The training wall guard is 30 hours; evaluation receives up to 12 hours; the
Batch task ceiling is 44 hours including setup and transfer allowance.
Fell Omen 2 assets are prepared before new training. Provisioning is standard,
with zero automatic task retries to prevent silent repetition of paid training.

The existing continuation evaluator allocates 20% of its envelope to the five
head-to-head tests and the remainder to the shared-deal FO2 suite. Incomplete
evaluation makes the job fail visibly while preserving completed training.
Batch releases its VM and disk; retained GCS artifacts continue to use storage.
Launching both commands below starts two separately billed jobs.

## Launch

Omit `--execute` to prepare a local reviewable bundle without submitting compute.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher

WIDE_RUN="hulhe-wide-p32-plus24-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_scaled_continuation_batch.py \
  --variant wide --run-name "$WIDE_RUN" \
  --output "outputs/${WIDE_RUN}-launch" --execute

DEEP_RUN="hulhe-deep-p32-plus24-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_scaled_continuation_batch.py \
  --variant deep --run-name "$DEEP_RUN" \
  --output "outputs/${DEEP_RUN}-launch" --execute
```

After successful submission, both run independently of local terminals. A
prepared bundle can be submitted with `--output PATH --submit-prepared`.
Immutable prefix claims reject attempts to overwrite an existing experiment.

Headline status, from any terminal:

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name~"hulhe-(wide|deep)-p32-plus24"' \
  --format='table(name.basename(),status.state)'
```

## Download analysis only

Set `RUN` to the actual job name for either experiment:

```bash
mkdir -p "outputs/${RUN}-results"
gcloud storage rsync --recursive \
  --exclude '(^|/)(state_[0-9]+|state)(/|$)|^(?!.*\.(json|md)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/${RUN}" \
  "outputs/${RUN}-results"
```

This downloads analytical JSON/Markdown and small publication metadata, excluding
models, replay, archives, logs and raw hand journals. Start with `continuation.json`,
`provenance.json`, `sampler-verification.json`, `evaluation/report.json`,
`evaluation/continuation_report.md`, and `reload-P*.json`.

## Local checks

```bash
python3 -m pytest tests/test_scaled_continuation.py tests/test_continuation.py \
  tests/test_continuation_batch.py tests/test_p104_continuation.py
```

The new tests perform real short full-network continuations for both architectures,
verify all five stage boundaries, preserve source file hashes, check sampling seeds,
and test rejection of source/configuration/recovery drift. Cloud plans and evaluation
seed selection are checked without launching paid compute. Reduced-budget local
checks are engineering tests, not evidence of production playing strength.

## Direct P56 architecture comparison

After both continuations complete, the [P56 round robin](P56_ROUND_ROBIN.md)
compares Wide-P56, Deep-P56 and the corrected smaller P56 in a separate evaluation-only Batch job.

The [P56 + 48-hour continuations](SCALED_P56_CONTINUATIONS.md) extend these completed
wide/deep runs through P104, retaining six-hour checkpoints and corrected sampling.
