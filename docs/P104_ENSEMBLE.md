# P104 twenty-network policy ensemble

This independent Google Batch experiment trains **20 new average-policy networks**
from the frozen P104 replay. It collects no self-play and does not modify P104,
the production learner, or any continuation experiment.

The source is `hulhe-p80-plus24-batch-20261120-171431`, iteration **3,489**,
policy SHA-256 `7266950fc5fd4adfa37319e41bc69264e938e98c46754874c99e23483e442ad2`.
Only policy replay shards and recovery metadata are downloaded. Source policy,
replay checksums, encoder, configuration and corrected trainer identity are checked.

## Training and playable export

- Keep P104's architecture, learning rate and batch size. Use corrected grouped
  objective-mass sampling, including its original loss scaling.
- Initialization seeds: **1045001–1045020**. Separate minibatch seeds:
  **1046001–1046020**. Each member gets a fresh initialization and optimizer.
- Select each member's fitting budget using the historical canonical 5% validation
  partition, weighted KL, and budgets **500, 1,000, 2,500, 5,000, 10,000, 20,000**.
  Train one path per member; ties select the smaller budget.
- Save the selected budget and initial-weight hash before refitting. Restart from
  that same initialization and fresh Adam; fit **all replay** for exactly the
  selected number of updates. Do not select or stop using the now-included
  validation groups. These are not independently held-out data after refitting.
- Retain all 20 final networks with equal weights. No seed/member selection,
  bootstrapping away scarce replay groups, or selection using poker match results.

`fit/ensemble.json`, `fit/experiment.json`, the 20 budget receipts, and the 20
`member_*.pkl` files form a portable, checksum-verified bundle. Paths are relative;
keep these files together. Members are individually playable with
`LoadedHULHEPolicy`; load the ensemble with:

```python
from hulhe_escher.game import load_hulhe_game
from hulhe_escher.p104_ensemble import EnsemblePolicy

policy = EnsemblePolicy(load_hulhe_game(), "path/to/fit/ensemble.json")
probabilities = policy.action_probabilities(state)
```

The ensemble encodes the observing player's state once, evaluates all 20 models
using vectorized inference, applies legal-action masking and softmax separately
to each model, then averages **probabilities**. It does not average logits, weights
or selected actions, nor select one member for an entire hand. Evaluation samples
an action from this probability distribution. Small floating-point differences
from vectorization are permitted and tested against separate model inference.

## Fresh paired-deal evaluation

For every deal group, play three matches with seats swapped:

1. Ensemble versus saved P104.
2. Ensemble versus pinned Fell Omen 2.
3. Saved P104 versus the same Fell Omen 2.

Each group therefore contains **six hands**. Use the same nine cards for all
matches and both seats, independent of betting. The two FO2 matchups also share
action random streams. Subtract the two FO2 scores **within each deal group**
to estimate improvement; do not treat them as independent samples.

The three predeclared tests are ensemble versus P104, ensemble's improvement over
P104 against FO2, and ensemble's absolute score against FO2. Bonferroni allocates
alpha **0.05/3** per two-sided test: **98.33% intervals** provide familywise 95%
coverage under the interval assumptions. Additional 95% intervals and P104's
absolute FO2 score are descriptive. A positive lower adjusted bound supports a
win/improvement; an interval crossing zero is inconclusive.

An excluded **10,000-group pilot**, seed **1049701**, estimates variance and actual
20-network throughput. Choose the largest required sample across the three tests,
targeting **80% estimated power for 2 bb/100**, with **100,000–1,000,000** confirmation
groups. Confirmation uses independent seed **1049702**. The count is fixed before
confirmation; there is no significance monitoring or early success stop. If the
one-million maximum caps the power target, reports explicitly show that limitation.
These units follow the repository convention: bb is the big blind/small bet.

The pilot writes `evaluation/plan.json` with measured throughput, expected runtime,
and a 25% runtime allowance. If the entire selected count will not fit the remaining
stage budget, the job fails before confirmation with
`INSUFFICIENT_TIME_FOR_SELECTED_SAMPLE`, preserving fits and pilot results. It never
silently lowers the sample to fit time. Incomplete matches cannot produce a
confirmatory conclusion. Fresh deal-group intervals condition on this fixed
ensemble and replay; they do not measure uncertainty across self-play runs or
establish exploitability/Nash convergence.

Versus saved P104, this experiment tests the **combined ensemble and all-replay
export change**. It does not isolate ensembling from the export change. Isolating
those effects would require an additional matched single-refit comparison.

## Compute and runtime

One **c4-standard-16**, **200 GB disk**, eight fitting threads, and eight evaluation
workers with one Torch thread each. The 20 members fit sequentially with shared
replay preparation, avoiding 20 parallel replay copies. No GPU or collection
workers are needed. There are no automatic whole-job retries; storage transfers
retain their existing retries.

Planning estimate: **4–8 hours**, excluding Batch queueing, subject to the measured
pilot. Task emergency ceiling: **12 hours**, including setup. Coordinator budget:
11 hours after startup, including input preparation; fitting guard 2.5 hours;
evaluation guard up to 9 hours, clipped to remaining coordinator time. These
guards do not promise completion or substitute for sample-size requirements.

Receipts, member policies and analytical reports upload periodically; the final
upload also preserves journals and logs. Only the results directory is uploaded,
not the downloaded source replay. Batch releases VM and disk; GCS outputs remain.

## Launch

This command submits a paid independent cloud job. Omit `--execute` to produce a
local review bundle without contacting Google Cloud. Submission captures the local
allowlisted source; it does not require downloading code from GitHub at runtime.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p104-ensemble-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_p104_ensemble.py \
  --run-name "$RUN" --output "outputs/$RUN-launch" --execute
```

After successful submission, the terminal may close. Check from any terminal:

```bash
gcloud batch jobs list \
  --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment=hulhe-p104-ensemble' \
  --format='table(name.basename(),status.state)'
```

## Download analytical results only

No model files, journals or logs are included in this download list:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="REPLACE_WITH_ACTUAL_JOB_NAME"
DEST="outputs/$RUN-results"
URI="gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN"
mkdir -p "$DEST"
for FILE in launch.json cloud_status.json provenance.json summary.json report.md
do
  gcloud storage cp "$URI/$FILE" "$DEST/$FILE" || break
done
```

For a failed/incomplete job, `summary.json` may not exist. Read `cloud_status.json`,
then download the available `fit/experiment.json`, `evaluation/plan.json` and
`evaluation/report.json`. They preserve fitting progress, throughput projections,
and incomplete status without requiring any model or log downloads.

## Local validation

```bash
python -m pytest tests/test_p104_ensemble.py tests/test_p104_diagnostics.py \
  tests/test_policy_sampling_experiment.py -q
```

Smoke tests retain all 20 members but use tiny networks, replay, update budgets
and deal counts. A check/call opponent is permitted only in smoke mode; such
results are never confirmatory. Tests check source preservation, independent
initialization, frozen budgets, full-replay fitting, portable checksums, actual
spawned evaluation, probability averaging, group covariance, budget refusal,
publication and the Batch launch contract.
