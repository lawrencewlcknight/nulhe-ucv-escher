# Wide and Deep P56 + 48-hour continuations

Two independent Google Batch jobs resume the completed Wide-P56 and Deep-P56
full recoveries, then add **48 active training hours** to reach P104. They use
exactly the original architecture, hyperparameters and archived trainer for each
variant. This is additional learning, not a restart or a migration.

## Immutable sources

| Variant | Source run | Full recovery | Playable policy |
|---|---|---|---|
| Wide | hulhe-wide-p32-plus24-20261124-112651 | state_001703 | P56.pkl |
| Deep | hulhe-deep-p32-plus24-20261124-112723 | state_001516 | P56.pkl |

The source pins are in `hulhe_escher/data/scaled_p56_continuation_sources.json`.
The completed reports confirm final reload verification. Read-only cloud checks
confirmed the latest pointers and downloaded only the small recovery manifests to
pin their SHA-256 hashes. Each full recovery is about 28.6 GB; full states are
fetched by the cloud job, not by the launcher on the laptop.

Before replay transfer the worker checks the policy, original trainer archive,
recovery iteration and recovery manifest hash. The archived source fingerprint,
production configuration, seed, worker/thread counts and sampling provenance must
match. Actual solver restoration verifies replay shards, model/target weights,
optimizers, counters and RNG state. There is no trainer-source modification.

Both sources already use **objective-mass policy-group sampling**. Every new
export must record this sampler and its expected iteration-derived sampling seed.

## Checkpoints and saved recovery

Playable policies: **P56** (unchanged baseline), **P62, P68, P74, P80, P86, P92,
P98, P104**. Checkpoints are six active training hours apart. The target is the
actual P56 cumulative active time plus 6/12/18/24/30/36/42/48 hours, avoiding
accumulation of iteration overshoots. Final iteration boundaries may overshoot
slightly. Extraction, saving, verification, uploads and evaluation add wall time.

Each stage exports a playable policy and complete training state, reload-verifies
it, then publishes in the existing ordered upload protocol. The latest full
recovery remains available for continuation; old full recovery directories may be
pruned under the existing retention policy, but all playable policies remain.
The P104 full recovery is sufficient to resume training with the preserved
`training_source.tar.gz`, `training/manifest.json` and `training/latest.json`.
Source experiments remain unchanged. Transfer retries and completed-stage retry
validation use the established continuation machinery, now with eight stages.

## Evaluation after training

For each architecture independently:

- **Primary:** P104 versus its own P56, two-sided 95% interval.
- **Eight adjacent comparisons:** each new checkpoint against its predecessor.
- **Fell Omen 2:** all nine policies on shared duplicate deals; estimate eight
  improvements relative to P56, alongside descriptive absolute scores.
- The 16 secondary tests use Bonferroni alpha **0.05/16** and **99.6875% intervals**.
  Confirmatory secondary claims require the primary test to show improvement.
- Excluded 10,000-pair/group pilot; 100,000–1,000,000 confirmation observations;
  target 80% estimated power for 2 bb/100. Binding sample/time caps are reported.
- Seat-swapped hands are averaged within each observation. FO2 changes subtract
  policy scores within each shared deal group (18 hands across nine policies).
- Fresh evaluation seed offsets: wide 3,000,000; deep 4,000,000. These differ from
  P32+24 and the P56 round robin. The training seed remains 20261115.

There is no new direct Wide-P104 versus Deep-P104 comparison in these jobs. Each
job can finish independently. Results condition on one trained policy sequence
per architecture, not independent training-seed replications.

## Resources and runtime

Each job uses **c4-standard-16**, **eight collection workers**, **eight training
threads**, and a **200 GB disk**, with no GPU and no automatic job retries.
Training has a 60-hour wall guard; evaluation has a 12-hour budget; the whole
Batch task has a 74-hour ceiling including setup and transfers. These are safety
limits, not expected runtimes. Allow roughly **52–58 hours per job**, depending on
checkpoint overhead, pilot-selected evaluation size and storage throughput.
Running both together uses two VMs; their costs add even though they finish in
parallel. Batch releases the VM/disk; GCS retains outputs and the latest recovery.

The coordinator supports both four-stage and eight-stage continuations; existing
launchers retain four stages and their original statistical design by default.
The saved trainers are restored from their original archives, so coordinator
changes do not alter the learning algorithm or invalidate recovery fingerprints.

## Prepare or launch

Prepare two independent bundles without starting cloud jobs:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
STAMP="$(date -u +%Y%m%d-%H%M%S)"
for VARIANT in wide deep
do
  RUN="hulhe-${VARIANT}-p56-plus48-${STAMP}"
  python3 tools/launch_scaled_p56_continuation_batch.py \
    --variant "$VARIANT" --run-name "$RUN" --output "outputs/${RUN}-launch" || break
done
```

To prepare and submit both jobs, use fresh names/directories:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
STAMP="$(date -u +%Y%m%d-%H%M%S)"
for VARIANT in wide deep
do
  RUN="hulhe-${VARIANT}-p56-plus48-${STAMP}"
  python3 tools/launch_scaled_p56_continuation_batch.py \
    --variant "$VARIANT" --run-name "$RUN" \
    --output "outputs/${RUN}-launch" --execute || break
done
```

Alternatively submit an already prepared bundle with `--output PATH
--submit-prepared`. The launcher uploads the local source archive with create-only
preconditions and submits Google Batch. After successful submission the terminal
can close. The jobs run independently of the laptop and of one another.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name~"hulhe-(wide|deep)-p56-plus48"' \
  --format='table(name.basename(),status.state)'
```

## Download analysis only

Set RUN to the completed job name and repeat for the other variant:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="REPLACE_WITH_ACTUAL_JOB_NAME"
mkdir -p "outputs/${RUN}-results"
gcloud storage rsync --recursive \
  --exclude='^(?!(?:(?:launch|continuation|provenance|sampler-verification|P[0-9]+|reload-P[0-9]+)\.json|evaluation/(?:.*/)?(?:report\.json|plan\.json|manifest\.json|design\.json|continuation_report\.md))$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/${RUN}" \
  "outputs/${RUN}-results"
```

This downloads analytical summaries and provenance, excluding model/replay files,
logs, per-hand journals and per-iteration metrics. Start with
`evaluation/continuation_report.md` and `evaluation/report.json`.

## Validation

```bash
python3 -m pytest -q tests/test_scaled_p56_continuation.py tests/test_scaled_continuation.py \
  tests/test_continuation.py tests/test_continuation_batch.py tests/test_p104_continuation.py
```

Coverage includes production source/variant checks, rejection before large replay
transfer, eight-stage grids and resume contracts, actual full-network restoration
for both architectures, preserved source files, ordered checkpoint publication,
P104 exports, fresh evaluation seeds and the 16-test secondary family. Local smoke
checks shorten training stages and are not evidence of poker strength.
