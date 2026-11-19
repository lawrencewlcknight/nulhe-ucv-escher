# P56 versus corrected P32

This evaluation compares the saved P56 policy with **both** corrected P32
average-policy refits (seeds 1701 and 1702) from the completed sampler experiment.
No refit is selected using previous playing results. All three policies and the
source metadata are pinned by SHA-256 in
`hulhe_escher/data/corrected_comparison_sources.json`.

The earlier P32→P56 continuation used the original, uncorrected P32 baseline.
This experiment removes that mismatch. Both endpoints use objective-mass group
sampling, the same architecture and encoder. P32's fitting budgets were already
selected by validation (5,000 updates for both refits), not by this match.
There is one P56 export and two P32 fitting seeds; they are not independent
self-play training runs. Differences in policy-fit randomness remain. This is
not a matched multi-seed causal estimate of training time alone.

## Statistical design

- One primary test: P56's equal-weight mean win rate against both corrected P32
  policies. Positive values favour P56.
- Each independent deal group plays both matchups in both seats: four hands.
  Cards stay in their seats while policies swap. The two seat averages are
  averaged across refits **before** computing uncertainty, retaining correlation.
- Fresh seed block: pilot 619501, confirmation 619502, disjoint from the prior
  sampler and continuation comparisons. Independent action streams per matchup.
- A separate 10,000-group pilot chooses the fixed confirmation sample size using
  variance and measured throughput. Pilot outcomes never enter the final estimate.
- Confirmation: 100,000–1,000,000 groups (400,000–4,000,000 hands), targeting
  80% estimated power for a 2 bb/100 effect, two-sided alpha 0.05.
- One primary 95% Student confidence interval. Separate refit scores and intervals
  are descriptive only; no multiple primary tests or winner selection.
- Time/sample caps and detectable effect are reported explicitly. A completed
  capped test can still provide a valid interval, but may miss the power target.
  Incomplete runs and smoke tests cannot claim confirmatory improvement.
- This is a direct head-to-head evaluation. It does not repeat Fell Omen 2 matches
  or infer a change in Fell Omen 2 winnings from the head-to-head result.

## Cloud execution

One Google Batch task: `c4-standard-16`, eight evaluation workers, 50 GB disk.
Only three playable policies and two metadata files are downloaded; no replay,
self-play collection, training, or FO2 installation is needed. Evaluation has a
2-hour budget, a 125-minute child-process guard, and a 3-hour Batch task limit
including startup and upload. The pilot measures actual throughput; these are
limits, not a promised runtime. Batch releases the VM/disk after the task.
No automatic retries launch another paid evaluation.

From the repository:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p56-vs-p32c-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_corrected_comparison.py \
  --run-name "$RUN" \
  --output "outputs/$RUN-launch" \
  --execute
```

Omit `--execute` to prepare and inspect the immutable source bundle and job spec.
The launcher uploads local source, including these implemented files; a Git push
is not required for that upload. Source runs are read-only. Input checksum,
metadata and architecture checks fail before dealing any cards.

Headline status, runnable from any terminal:

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name:hulhe-p56-vs-p32c' --format='table(name.basename(),status.state)'
```

## Download only analytical results

Set `RUN` to the submitted job name:

```bash
RUN="REPLACE_WITH_JOB_NAME"
DEST="/Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher/outputs/$RUN-results"
URI="gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN"
mkdir -p "$DEST/evaluation"
gcloud storage cp "$URI/launch.json" "$URI/cloud_status.json" "$URI/sources.json" "$DEST/"
gcloud storage cp "$URI/evaluation/manifest.json" "$URI/evaluation/plan.json" \
  "$URI/evaluation/report.json" "$URI/evaluation/comparison_report.md" "$DEST/evaluation/"
```

`comparison_report.md` is the human-readable result; `report.json` contains the
primary and descriptive statistics. Raw pair journals stay in cloud storage for
reproducibility and are not included in this download. Failed runs may lack a
plan/report; inspect `cloud_status.json`, `evaluation.log`, and `batch-run.log`.
