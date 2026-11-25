# Protected sequential Limit Hold’em: P8 plus 24 hours

Continue the completed `hulhe-sequential-4plus4-20261124-184723` experiment from
its full P8 checkpoint, iteration 170. This is 24 additional active training hours,
not a fresh run or a refit of the playable policy alone.

## Preserved learner

Use the original checksum-pinned archived trainer, model/replay state, optimizers,
random states, training seed 20261115, eight collection workers and eight central
Torch threads. Do not migrate the source or configuration. Both regret networks
and the average-policy network retain the 1,496,067-parameter protected sequential
architecture, protection coefficient 0.3, corrected objective-mass policy sampling,
four-million-row policy reservoir, and the original fitting/validation budgets.
The new coordinator only restores, schedules, publishes and evaluates.

The baseline comes from:

`gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-sequential-4plus4-20261124-184723`

Identity pins in `hulhe_escher/data/sequential_continuation_source.json` cover the
P8 policy, source archive, trainer fingerprint, recovery manifest, iteration and
actual cumulative active time (29,035.03224580901 seconds). The original source
archive and recovery metadata were checked against GCS when preparing this run.
The launcher does not download the full recovery locally; the cloud job does so.

Before replay download, reject an incorrect archive, baseline iteration or recovery
manifest. Before training, validate the exact production configuration and protected
sampler provenance, checksum all recovery shards and actually restore the full state
in a fresh interpreter. Keep the original run read-only and use a new sibling prefix.

## Schedule and artifacts

| Policy | Additional active training after P8 | Nominal total |
|---|---:|---:|
| P8 | 0 hours | 8 hours |
| P14 | 6 hours | 14 hours |
| P20 | 12 hours | 20 hours |
| P26 | 18 hours | 26 hours |
| P32 | 24 hours | 32 hours |

Target cumulative times are the actual saved P8 time plus 6/12/18/24 hours. The
original P8 overshoot is preserved, and each stage finishes its current iteration.
Active time includes collection and regret/critic/calibration fitting; it excludes
policy extraction, persistence, restoration, uploads and evaluation.

Each six-hour endpoint fits and selects a playable average policy with the existing
held-out procedure, saves full recovery, verifies it in a fresh process, and uploads
recovery before publishing the policy marker. Every next stage starts a new process
using the archived trainer with `--resume`. Protected-fitting and corrected-sampling
provenance are checked at each exported stage. Keep all five playable policies and
only the newest full recovery in the new run; preserve the original P8 separately.
P32's full state can be used for another continuation.

The original trainer is published as `training_source.tar.gz`. The new Batch
coordinator is `source.tar.gz`; these are intentionally different. Use the former
when restoring training. `resume.json` records the required paths and fingerprint.
Transfer retries and ordered checkpoint publication reuse the existing continuation
implementation. There is no automatic paid rerun of a failed experiment.

## Evaluation after training

Reuse the established continuation suite, with independent pilot/confirmation seeds
(offset 3,000,000, separate from the original sequential and wide/deep experiments):

- Primary: **P32 versus P8**, alpha 0.05 and a 95% confidence interval.
- Four adjacent comparisons: P14/P8, P20/P14, P26/P20 and P32/P26.
- Fell Omen 2: evaluate P8, P14, P20, P26 and P32 on shared duplicate deals with
  seats swapped, using the same free-fold-to-check adaptation. Report absolute
  scores and the four paired improvements over P8.
- The four adjacent matches and four FO2 improvements form one secondary family:
  alpha 0.05/8, giving 99.375% intervals. Confirmatory secondary claims require the
  primary to pass. Absolute FO2 scores are descriptive with 95% intervals.

Each match uses 10,000 excluded pilot groups, followed by a fixed plan of
100,000–1,000,000 confirmation groups, targeting 80% estimated power for 2 bb/100.
The FO2 plan also targets ±2 bb/100 precision for absolute scores. Runtime and sample
caps may prevent those targets; report achieved precision and detectable effects.
No significance-based early stopping or pooling with the earlier FO2 run. One
training seed supports run-specific conclusions, not conclusions across seeds.
This does not directly compare with historical smaller/wide/deep policies.

## Batch and runtime

One Google Batch job in europe-west1 on **c4-standard-16**, 200 GB disk, eight
collection workers/eight fitting threads, then eight evaluation workers. No GPU.
Unlike the initial run's separate evaluator, this continuation trains first and
then evaluates on the same Batch VM, matching the existing 24-hour continuations.
Fell Omen 2 assets are prepared before new training begins.

Budgets: 30 hours for training/restoration/export, 12 hours for evaluation and a
44-hour task ceiling including setup/transfers. These are limits, not a guaranteed
completion time. Failed or incomplete evaluation makes the job fail while preserving
successfully published training artifacts. Batch releases its VM/disk automatically;
GCS retains the original source run, new policies, latest recovery and reports.

## Prepare and submit

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-sequential-p8-plus24-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_sequential_continuation_batch.py \
  --run-name "$RUN" --output "outputs/cloud-$RUN"

# Explicitly submit the exact prepared source archive and job.
python3 tools/launch_sequential_continuation_batch.py \
  --output "outputs/cloud-$RUN" --submit-prepared
```

Alternatively, add `--execute` to the preparation command. Preparation alone makes
no cloud calls and creates no paid resources. Submission claims a unique GCS prefix
before launching, preventing accidental reuse of the original experiment.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name:hulhe-sequential-p8-plus24' \
  --format='table(name.basename(),status.state)'
```

For analytical results only:

```bash
RUN=hulhe-sequential-p8-plus24-YYYYMMDD-HHMMSS
mkdir -p "outputs/$RUN-results"
gcloud storage rsync --recursive \
  --exclude='(^|/)(training|original-run|archived-trainer|fo2-assets|state_[0-9]+)(/|$)|^(?!.*\.(json|md)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN" \
  "outputs/$RUN-results"
```

Read `continuation.json`, `evaluation/report.json`, `evaluation/report.md`,
`provenance.json`, checkpoint markers and `reload-P*.json`. The allowlist excludes
models, replay, logs, source archives and per-hand journals.

## Validation

`tests/test_sequential_continuation.py` checks the offline Batch plan, source
identity/configuration/sampling/protection guards, metadata rejection before large
replay downloads, fresh evaluation seeds and multiplicity, and a real full-size
network continuation with reduced buffers/budgets through all four checkpoints.
The integration check verifies restored checkpoints before publication and confirms
the source files are unchanged. Set `HULHE_SEQUENTIAL_ARCHIVE` to the downloaded
original archive to exercise the precise production trainer; otherwise the test
creates an offline archive from the repository.
