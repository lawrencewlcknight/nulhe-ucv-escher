# Sequential P32 versus corrected Small, Wide, Deep and Hybrid P32

Compare the saved protected Sequential-P32 policy with four fixed P32
comparators. This is an evaluation-only experiment: no training, refitting,
replay download, full-state restore or Fell Omen 2 assets are required.

## Saved policies and provenance

| Policy | Source experiment |
|---|---|
| Sequential-P32 | `hulhe-sequential-p8-plus24-20261125-054522` |
| Small-P32-corrected | Both archived refits, seeds 1701 and 1702, from `hulhe-sampler-p32-20261117-145836` |
| Wide-P32 | `hulhe-wide32-batch-20261122` |
| Deep-P32 | `hulhe-deep32-batch-20261122` |
| Hybrid-P32 | `hulhe-hybrid32-batch-20261124-105219` |

`hulhe_escher/data/sequential_p32_comparison_sources.json` pins every downloaded
policy and provenance file by SHA-256. The worker downloads only those files,
checks their hashes before deserialization, and validates policy identities,
corrected sampling, protected Sequential provenance, and compatible game,
encoder and action contracts. The model architectures intentionally differ.
Inputs are checked again after evaluation and stay outside the upload tree.
Source experiments remain read-only; source/output prefix overlap is rejected.

The Small comparator gives equal weight to the two archived corrected refits.
Neither fit is selected using match results. It is a score average over the two
saved policies, not an ensemble that mixes their actions or a new refit.

## Statistical design

Four equally primary two-sided comparisons, with positive scores favouring
Sequential-P32:

1. Sequential-P32 versus Small-P32-corrected.
2. Sequential-P32 versus Wide-P32.
3. Sequential-P32 versus Deep-P32.
4. Sequential-P32 versus Hybrid-P32.

Every independent deal group uses the same cards for five seat-swapped matches:
Sequential against each of the two Small refits, Wide, Deep and Hybrid. This is
**ten hands per group**. Cards stay fixed in seats and on the board while policies
swap seats. Each matchup has its own action-randomness stream. Average the two
seats for each matchup, then average the two Small matchup scores for the Small
primary comparison. There is one observation per primary comparison per group;
the correlated hands and refits are not independent observations.

- Fresh production pilot seed **927321**, confirmation seed **927322**; smoke
  seed **927921** is separate.
- Exclude the **10,000-group pilot** from all confirmation estimates.
- Target **80% estimated power for a 2 bb/100 effect** in each primary test.
  Select the largest pilot-based sample requirement across the four tests,
  bounded by **100,000–1,000,000 confirmation groups**. This is one million to
  ten million confirmation hands overall, excluding the 100,000 pilot hands.
- Four tests use alpha **0.05/4 = 0.0125** and **98.75% confidence intervals**,
  controlling familywise error at 5% by Bonferroni despite shared cards.
- Freeze the common sample size before confirmation. The pilot measures
  throughput, but runtime does not reduce the selected sample. If that sample
  cannot fit with the evaluator's runtime reserve, fail explicitly. Budget
  exhaustion during confirmation produces incomplete results, not an early
  confirmatory conclusion. There is no outcome-based stopping.
- Report the one-million-group cap when binding, achieved precision and estimated
  detectable effects at the target power. Pilot power estimates are not guarantees.
- Per-refit Small scores are descriptive. They cannot replace the equal-weight
  primary result or select a preferred comparator. Smoke, failed and incomplete
  runs cannot support confirmatory claims.

These tests condition on six saved policies from single training runs. Training
recipes, architectures, replay sizes, iterations and work per active hour differ.
The Small policies were corrected by historical refitting, while the newer
learners used corrected sampling during training. The comparison does not isolate
the effect of a single architecture change, establish training-seed generalization,
or measure exploitability. An interval crossing zero is inconclusive.

## Google Batch execution

One standard **c4-standard-16** task, **eight** evaluation workers with one Torch
thread each, and a **50 GB** disk. Evaluation has a **six-hour** budget
(21,600 seconds), a **six-hour-five-minute** process guard (21,900 seconds),
and a **seven-hour** Batch ceiling (25,200 seconds) including setup and uploads.
These are ceilings; the pilot supplies the runtime estimate. No automatic task
retry is configured. Existing bounded storage transfer retries remain enabled.
Batch releases its VM and disk after the task; GCS retains results and journals.

Prepare a reviewable local bundle without making cloud calls:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-sequential-p32-vs-all-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_sequential_p32_comparison.py \
  --run-name "$RUN" --output "outputs/${RUN}-launch"
```

Submit that exact prepared source archive and job:

```bash
python3 tools/launch_sequential_p32_comparison.py \
  --output "outputs/${RUN}-launch" --submit-prepared
```

Alternatively, add `--execute` to a preparation command with a fresh run name
and output directory. Submission uses create-only uploads and checks the
prepared source archive hash. The source archive contains allowlisted code and
metadata, excluding local outputs, model binaries, replay and credentials. The
job runs independently of the terminal after submission. Defaults are project
`clever-overview-399515`, region `europe-west1`, and the existing results bucket
and runner service account.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name~hulhe-sequential-p32-vs-all' \
  --format='table(name.basename(),status.state)'
```

## Download analytical summaries only

Set RUN to the submitted job name. This regex allows only the launch/source/status
metadata and evaluation summaries. It excludes model files, replay, archives,
logs, raw hand/deal journals and other directories.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="REPLACE_WITH_ACTUAL_JOB_NAME"
URI="gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN"
DEST="outputs/$RUN-results"
mkdir -p "$DEST"
gcloud storage rsync --recursive \
  --exclude='^(?!(?:launch\.json|sources\.json|cloud_status\.json|evaluation/(?:manifest\.json|plan\.json|report\.json|comparison_report\.md))$).*' \
  "$URI" "$DEST"
```

Start with `evaluation/comparison_report.md`, then inspect `evaluation/plan.json`
for sample selection and runtime feasibility. For a failed job, read
`cloud_status.json` first; later-stage files may be absent. `COMPLETE` is published
only after the result upload succeeds. The source policy files are never uploaded
to this run's results.

## Local validation

```bash
python3 -m pytest -q tests/test_sequential_p32_comparison.py tests/test_sequential_p32_cloud.py
```

The cloud/launcher checks exercise pinned-only input download, failed and timed-out
evaluation, publication ordering, source overlap protection, Batch limits and
retries, archive allowlisting, and submission of the prepared bundle.
