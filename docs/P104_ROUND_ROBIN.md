# P104 Wide / Deep / historical Small round robin

Evaluate three fixed, saved P104 policies head to head. This is an evaluation-only
Google Batch job: no training, policy refitting, replay download, full-state restore
or additional Fell Omen 2 evaluation.

## Saved policies

| Policy | Source experiment | Iteration |
|---|---|---:|
| Wide-P104 | hulhe-wide-p56-plus48-20261125-192126 | 3028 |
| Deep-P104 | hulhe-deep-p56-plus48-20261125-192126 | 2802 |
| Small-P104 | hulhe-p80-plus24-batch-20261120-171431 | 3489 |

Small-P104 is the original historical production export, already using corrected
objective-mass grouping. It is not an ensemble or a newly fitted policy.

`hulhe_escher/data/p104_round_robin_sources.json` pins all three policy files and
their P104 metadata by SHA-256, using the completed local analytical results.
The worker downloads only these six files and checks every checksum before loading
any policy. It verifies production metadata, iteration, actual cumulative training
time, corrected sampling and compatible game/encoder/action contracts. Architectures
intentionally differ. Inputs are checked again after evaluation, kept separate from
the uploaded results, and never modified. Output/source prefix overlap is rejected.

## Statistical design

Reuse the [P56 round-robin protocol](P56_ROUND_ROBIN.md) with new P104 identities
and independent deal seeds:

1. **Wide-P104 versus Deep-P104.**
2. **Wide-P104 versus Small-P104.**
3. **Deep-P104 versus Small-P104.**

Positive scores favour the first policy. All three comparisons are equally primary.
Each independent deal group contains six hands: all three matchups, each with seats
swapped. Cards stay fixed in seats and on the board while policies swap seats.
Each matchup uses a separate action-randomness stream. Average the two hands before
calculating uncertainty: six hands do not constitute six independent observations.

- Fresh pilot seed **928041**, confirmation seed **928042**; smoke uses 928941/928942.
- Exclude the **10,000-group pilot** from all confirmation estimates.
- Use **100,000–1,000,000 confirmation groups**, equivalent to 200,000–2,000,000
  hands per matchup and 600,000–6,000,000 hands overall, excluding the pilot.
- Plan for **80% estimated power for 2 bb/100** per matchup, using the largest
  pilot-based sample requirement across the three comparisons.
- Apply alpha **0.05/3** and **98.333% confidence intervals**, controlling the
  familywise error rate at 5%. Report sample caps and estimated detectable effects.
- Freeze the sample size before confirmation. There is no significance-based
  stopping, policy selection or sample reduction to fit runtime. If the selected
  sample cannot fit with a 20% runtime reserve, fail explicitly. Incomplete/error
  results cannot support confirmatory claims.

The historical smaller learner has a different training recipe. All policies have
approximately 104 active training hours, but their iterations, architectures and
samples collected differ. This compares the saved policies; it does not isolate an
architecture effect, estimate training-seed variability, or measure exploitability.
Poker matchups can be non-transitive, so report all three outcomes without forcing
a league ranking.

## Google Batch

One **c4-standard-16**, eight single-threaded evaluation workers and a **50 GB disk**,
in `europe-west1`. The evaluator has a three-hour budget, a three-hour-five-minute
process guard, and a four-hour overall task ceiling including setup and transfers.
These are ceilings; the pilot provides a measured runtime estimate. No automatic
paid job retry is configured. Existing bounded storage-transfer retries apply.
Batch releases its VM and disk; results remain in GCS. COMPLETE status is published
only after results upload successfully.

Prepare locally without cloud calls, then submit the exact prepared bundle:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p104-round-robin-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_p104_round_robin.py \
  --run-name "$RUN" --output "outputs/${RUN}-launch"

python3 tools/launch_p104_round_robin.py \
  --output "outputs/${RUN}-launch" --submit-prepared
```

Alternatively add `--execute` to the preparation command. Submission verifies the
prepared source archive checksum and claims a unique output prefix before launching.
The cloud job runs independently of the terminal after successful submission.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name:hulhe-p104-round-robin' --format='table(name.basename(),status.state)'
```

## Download analytical results only

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="REPLACE_WITH_ACTUAL_JOB_NAME"
URI="gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN"
DEST="outputs/$RUN-results"
mkdir -p "$DEST/evaluation"
for FILE in launch.json sources.json cloud_status.json evaluation/manifest.json \
  evaluation/plan.json evaluation/report.json evaluation/comparison_report.md
do
  gcloud storage cp "$URI/$FILE" "$DEST/$FILE" || break
done
```

This exact list excludes models, replay, logs, source archives and raw hand journals.
Start with `evaluation/comparison_report.md`. For failures, inspect `cloud_status.json`;
later-stage files may be absent.

## Validation

```bash
python3 -m pytest -q tests/test_p56_round_robin.py tests/test_p104_round_robin.py
```

Tests cover original P56 compatibility, P104 source pins, checksum checks before
deserialization, corrected sampling, actual spawned mixed-architecture paired-deal
inference, source preservation, fresh seeds, pilot exclusion, multiplicity, sample
caps, incomplete/error paths, cloud timeouts and publication order, offline Batch
packaging and create-only prepared submission. Smoke results are not poker-strength
evidence.
