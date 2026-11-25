# P56 Wide / Deep / corrected Small round robin

Evaluate the three fixed, saved P56 policies against each other. No training,
policy refitting, full-state restore, replay download or Fell Omen 2 assets are
needed. The existing continuation evaluations already cover Fell Omen 2.

## Pinned policies

| Policy | Source experiment | Iteration |
|---|---|---:|
| Wide-P56 | hulhe-wide-p32-plus24-20261124-112651 | 1703 |
| Deep-P56 | hulhe-deep-p32-plus24-20261124-112723 | 1516 |
| Small-P56 | hulhe-p32-plus24-batch-20261118-1105 | 1943 |

`hulhe_escher/data/p56_round_robin_sources.json` pins each playable policy and its
P56 metadata by SHA-256. These identities come from the completed local analytical
reports; the cloud worker verifies all six files before deserialising policies.
It checks iteration, cumulative training time, corrected objective-mass sampling,
production metadata, and compatible game/encoder/action contracts. Architectures
intentionally differ. Inputs are checked again after evaluation and never uploaded
or overwritten. The smaller P56 is the established corrected export, not a new
refit of historical P32.

## Statistical design

Three equally primary comparisons, positive scores favour the first policy:

1. Wide-P56 versus Deep-P56.
2. Wide-P56 versus Small-P56.
3. Deep-P56 versus Small-P56.

Each independent deal group contains six hands: all three matchups, each with
seats swapped. Cards stay fixed in the seats and board; policies swap seats.
Each matchup's two results are averaged before computing uncertainty. Shared
cards across matchups do not turn the six hands into six independent samples.
Matchups have separate action-randomness streams.

- Fresh production pilot seed **926561**, confirmation seed **926562**.
- Excluded pilot of **10,000 groups**.
- Confirmation sample **100,000–1,000,000 groups**, or 200,000–2,000,000 hands per
  matchup and 600,000–6,000,000 hands overall, excluding pilot hands.
- Pilot variance targets **80% estimated power for 2 bb/100 per matchup**.
- Use the largest required sample across the three matchups, subject to the
  declared one-million-group cap. Report binding caps and estimated detectable
  effects; power estimates are not guarantees.
- Three two-sided tests use alpha **0.05/3** and **98.333% confidence intervals**,
  providing Bonferroni familywise error control at 5% despite shared deals.
- Freeze the sample size before confirmation. No significance-based stopping or
  selection of whichever policy looks best during the run.
- The pilot also measures throughput. If the selected sample cannot fit with a
  20% runtime reserve, fail explicitly rather than silently shrinking it. Partial
  or failed confirmation runs cannot claim confirmatory significance.

These tests compare three fixed saved policies. Different training histories,
replay sizes, samples per training hour and architectures remain confounded.
They do not estimate training-seed generalization or exploitability. Poker
matchups may be non-transitive; a forced league ranking is not the primary result.

## Google Batch execution

One `c4-standard-16` task, eight single-threaded evaluation workers, 50 GB disk.
Evaluation has a three-hour stage budget, a three-hour-five-minute process guard,
and a four-hour total task limit including setup/uploads. No automatic task retries.
Existing storage transfer retries apply. Batch releases its VM/disk on completion;
GCS results remain. Sources are read-only and output/source overlap is rejected.

Allow roughly **45–90 minutes** for setup and typical confirmation sizes, based on
previous local-policy match throughput; the pilot provides a measured estimate.
The four-hour limit is an emergency ceiling, not the expected duration.

Prepare locally without launching:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p56-round-robin-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_p56_round_robin.py \
  --run-name "$RUN" --output "outputs/${RUN}-launch"
```

Submit an independent cloud job (use a fresh run name and output directory):

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p56-round-robin-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_p56_round_robin.py \
  --run-name "$RUN" --output "outputs/${RUN}-launch" --execute
```

The launcher archives the local allowlisted source, uploads it with create-only
preconditions, and submits the Batch job. It does not depend on an open terminal
or a GitHub checkout during execution. Defaults are project
`clever-overview-399515`, region `europe-west1`, and the existing results bucket
and runner service account.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name~hulhe-p56-round-robin' --format='table(name.basename(),status.state)'
```

## Download analytical summaries only

Set RUN to the submitted job name. This explicit list excludes models, logs and
raw hand journals, retaining all files needed to interpret the comparison:

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

Start with `evaluation/comparison_report.md`. For a failed job inspect
`cloud_status.json` first; a failure before confirmation may not have every file.
Complete-pair journals remain in cloud storage for reproducibility. Final status
is published after results so COMPLETE is not visible ahead of its reports.

## Local validation

```bash
python3 -m pytest -q tests/test_p56_round_robin.py
```

Tests cover checksums before pickle loading, production identities, corrected
sampling, compatible mixed architectures, actual spawned paired-deal inference,
source preservation, three-test sample planning, pilot exclusion, capped power,
partial/error/budget paths, cloud publication ordering and source archive guards.
