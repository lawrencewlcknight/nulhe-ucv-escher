# Two independent P104 diagnostic experiments

Both jobs pin P104 from `hulhe-p80-plus24-batch-20261120-171431`, iteration 3,489,
SHA-256 `7266950fc5fd4adfa37319e41bc69264e938e98c46754874c99e23483e442ad2`.
They read the source and write only to separate output prefixes. They neither
modify the production learner nor touch the P104→P128 continuation.

Each job runs independently in Google Batch on c4-standard-16, with eight CPU
threads/workers, a 200 GB disk and no automatic whole-job retry. Storage retries
remain active. Batch releases its VM/disk; outputs remain in Cloud Storage.

## Experiment 1: policy extraction

Expected elapsed time: approximately 2–3 hours, excluding Batch queueing.
Emergency task ceiling: four hours, including setup and transfers. The coordinator
allows at most 3.5 hours after startup, with bounded subprocesses and time reserved
for publication. Estimates are not completion guarantees.

Only the policy replay shards and full-state metadata are downloaded. There is
**no new self-play**. Restore the current architecture, encoder, learning rate and
objective-mass sampling. Use fitting seeds **1042701 and 1042702**.

For each seed:

1. Reset a copy of the policy model. Use the original stable canonical 5% validation
   partition. Train on the other groups and record validation-weighted KL at
   **500, 1,000, 2,500, 5,000, 10,000 and 20,000** updates. Choose the smallest
   budget with the lowest validation KL (exact ties prefer fewer updates).
2. Save this selected training-only model as the **control**. Also save the best
   model from the original 2,500/5,000/10,000/20,000 grid as a descriptive reference.
   Both are snapshots of the same training path, not extra independent fits.
3. Freeze the budget in a receipt. Restart from the identical initial weights and
   a fresh matched Adam optimizer; train on **all grouped replay** for exactly that
   budget. Never select this refit using data now included in training.

The four primary playable exports are `control_<seed>.pkl` and
`all_replay_<seed>.pkl`. Two `legacy_<seed>.pkl` files preserve the original-grid
reference. Report initial-weight hashes, partitions, sampling, selected budgets,
fitting time, weighted KL, action L1/argmax errors, and street/frequency breakdowns.
Source P104 is also measured. Former validation metrics after all-replay fitting
are **in-sample**; they do not establish independent generalization.

### Fresh-deal confirmation

- Primary: each refit versus its matched control on fresh duplicate deals with
  seats swapped. Average the two seed-specific pair scores within each shared
  deal group; that group is the independent statistical observation.
- Pilot: 10,000 groups, excluded from confirmation. Select a fixed confirmation
  count between **400,000 and 1,000,000 groups**, targeting 1 bb/100 at 80% estimated
  power. Report one primary 95% interval. Do not stop on significance.
- Secondary: both controls, both refits and P104 versus the same pinned Fell Omen 2
  build (free-fold-to-check adaptation), on new shared duplicate deals. Use a
  separate 10,000-group pilot and **100,000–1,000,000 groups**, targeting 2 bb/100
  for the two-seed average refit-minus-control improvement.
- The FO2 improvement has a fixed-sequence gate: it is confirmatory at 95% only
  after a significant positive primary result. Otherwise it is descriptive.
  Absolute scores and individual-seed results are always descriptive. A capped
  power target is explicitly reported; an unaffordable minimum is never reduced.
- Head-to-head seeds are 1048701/1048702; FO2 seeds are 1048801/1048802
  (pilot/confirmation). No lucky seed or candidate is selected from match results.
- Intervals condition on this replay and these two fitted policies. They do not
  estimate training-seed variability, exploitability or universal superiority.

Fitting, primary evaluation, FO2 and rivers run sequentially, with guards of
30, 90, 45 and 40 minutes respectively, shortened if the total budget is nearly
exhausted. Playing-strength confirmation takes priority over river diagnostics.
A missing or incomplete required stage fails the job and retains partial results.

## Experiment 2: regret-learning investigation

Expected elapsed time: approximately 30–90 minutes; the task ceiling is two hours.
The coordinator budget is 90 minutes after startup. This job restores the **full**
P104 recovery and verifies its archived corrected trainer and original runtime.

It captures exactly **one isolated production-sized iteration**, 3,490, using
its original eight collection workers. All regret rows for each player are saved,
with frozen targets, before/after predictions and model/optimizer snapshots. Critic
captures retain the existing bounded sample. No resumable continuation is written.
The source checkpoint remains unchanged. The archived replay does not retain
trajectory identifiers; reports explicitly mark them unavailable rather than
inventing independent samples or cluster-valid uncertainty estimates.

For exactly equal encoded inputs and legal masks, report the empirical identity:

```
row-weighted target MSE = within-group target variation + error predicting the sampled group mean
```

Report streets, repeated groups, singletons, multiplicity and IID sample-mean
noise sensitivity before/after the production fit. Production post-fit errors
are in-sample. A noisy sampled mean is not a known true regret target.

Conditional fitting requires:

- At least 30 repeated preflop groups and 100 repeated rows for each player;
- At least one player with post-fit mean-target error contributing at least 10%
  of repeated-row error and at least one exceeding 1.25 times the IID mean-noise
  sensitivity scale;
- Nonempty preflop support in all fitting partitions.

This is an explicitly exploratory **spending gate**, not a significance test or
proof of inadequate capacity. Correlation can invalidate the IID noise scale.

If the gate passes, use two seeds (**1043901, 1043902**), fresh Adam at 0.0003,
clipping at 10, at most 500 updates per player and budgets **0, 100, 250, 500**:

1. `row_current`: additional fitting of P104 networks to raw frozen rows.
2. `group_mean_current`: same architecture, sampling training groups proportional
   to multiplicity and fitting their means. This preserves expected row-MSE
   gradients; the removed within-group variance is a constant in the objective.
3. `expanded_group_mean`: only if the first two arms fail to reduce mean validation
   error by at least 10%. Add a zero-output residual correction, verifying exact
   prediction equality at initialization, then use weighted group-mean fitting.

Use target-independent 60/20/20 group-level training/validation/audit splits.
Duplicates cannot cross partitions. These holdouts apply to the diagnostic fit;
source models may have historical exposure. All streets contribute training
examples. Validation selects budgets and the arm with lowest two-seed mean
preflop mean-target MSE; simpler arms win ties and source wins ties with fitting.
Freeze selection before audit or river inspection. **Retain both fitting seeds.**

The selected `.pt` artifacts are diagnostic instantaneous regret strategies only.
They are neither replacement average policies nor resumable production checkpoints.
If fitting is skipped or update zero remains best, compare the unchanged source
strategy and explicitly record that no fitting improvement was established.
Capture, fitting and river subprocess guards are 15, 20 and 30 minutes, subject to
the overall budget. Any promising result supports a later short continuation test;
it does not automatically change production training.

## Shared river diagnostics

Use the same **90 development river positions** from
`hulhe-positions-p56-20261119-143942`, library digest
`ee1e877034f84335282d87e620f1f0295a046c2efde0427cdc93cbe4f42c065e`.
The learned opponent and its conditional ranges remain frozen at P56 for all
candidates. Also evaluate check/call, value and pressure opponents with strong,
uniform and weak ranges. Earlier inspected development positions never select
models; assessment positions remain locked.

Report conditional value and remaining-response-gap changes, local-gap changes,
root-action effects with baseline continuation fixed, fold/call/raise probability
changes, improved/regressed counts and encounter/family summaries. These values
are **bb per decision**, not whole-game bb/100. There are no confidence intervals
on correlated position averages. The policy arm compares each refit with its
matching control and both with P104; the regret arm compares both selected seeds
with the P104 instantaneous regret strategy. All face the same frozen P56 opponent.

## Launch separately

These commands submit paid cloud jobs. Omit `--execute` to create local bundles
only. Use unique job names and output directories. Recommended order: policy
experiment first, regret experiment afterward; concurrency is optional.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p104-policy-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_p104_diagnostic.py --mode policy \
  --run-name "$RUN" --output "outputs/$RUN-launch" --execute
```

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p104-regret-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_p104_diagnostic.py --mode regret \
  --run-name "$RUN" --output "outputs/$RUN-launch" --execute
```

Defaults: project `clever-overview-399515`, region `europe-west1`, output bucket
`gs://clever-overview-399515-fhp-escher-results/hulhe`, service account
`fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com`.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment~"hulhe-p104-(policy|regret)"' \
  --format='table(name.basename(),status.state)'
```

## Download only analytical summaries

Set `RUN` to the actual job name. Both jobs publish `summary.json`, `report.md`,
`provenance.json`, and `cloud_status.json`. The summary includes policy/refit
metrics or regret decomposition and fit results, fresh-match results where
applicable, and all river comparison summaries. Models, captures, replay, logs,
and per-hand journals are excluded by this explicit download list.

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

For an incomplete job, download `cloud_status.json` first. Partial analytical
reports are under `fit/experiment.json`, `head/report.json`, `fo2/report.json`,
`capture/report.json`, `diagnostic/report.json`, and `rivers/report.json` as
applicable. A failed required stage is never labelled a completed experiment.
