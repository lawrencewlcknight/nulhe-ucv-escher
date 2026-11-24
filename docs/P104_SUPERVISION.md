# P104 regret-supervision audit

This independent Google Batch experiment tests whether noisy or systematically
misaligned regret supervision is a useful next target for improving UCV-ESCHER.
It freezes P104 (iteration 3,489), including both players' instantaneous regret
strategies, the two critics and calibration model. It evaluates prospective
iteration 3,490 without fitting, advancing the production run, or changing replay.

The source is the completed `hulhe-p80-plus24-batch-20261120-171431` run.
The policy checksum and corrected trainer fingerprint are pinned in
`hulhe_escher/p104_protocol.py`. Only the recovery manifest, metadata (which
contains all model weights) and P104 policy are downloaded. Replay shards are
not needed. Source integrity is checked again after the audit.

## Design

1. Collect 20,000 frozen diagnostic traversals, alternating both seats and both
   critic folds. These follow the production traverser proposal and opponent
   instantaneous strategy. They populate no training buffers.
2. Select 96 positions using encounter hashes, before observing any target or
   error: 16 flop, 32 turn, 48 river; 48 development, 24 validation, 24 untouched
   assessment positions. Each street/partition balances seats. Connected flop
   and full-board suit orbits remain in one partition. Existing protected
   assessment flop orbits are reserved for assessment.
3. At each position enumerate compatible opponent hands. Compute both the
   opponent-reach posterior and the actual frozen production-visitation
   posterior, including the hero's earlier sampling proposals separately for
   each critic fold. Report their distance, effective hand count and action-value
   differences. A full state with one fixed opponent hand is **not** treated as
   the entire information state.
4. Compute independent references: enumerate every hidden hand and remaining
   betting branch on rivers; on flop and turn use 4,096 terminal-payoff rollouts
   per legal action. A mixture over the three posteriors with bounded importance
   ratios estimates all three expectations. Shared random numbers across forced
   actions/ranges retain covariance. Folding values are known exactly.
5. For each fold sample 32 independent opponent hands from its production
   posterior, then eight independent continuations per hand. Compare:

   | Arm | Change at the selected root decision |
   |---|---|
   | Current | Native calibrated proposal and recursive UCV estimator |
   | Average four | Mean of four independent current estimates, including four hidden draws |
   | All legal actions | Evaluate each legal root action, with native UCV continuation below it |
   | Uniform | Uniform legal root proposal, with correct importance correction and native continuation below it |

   P104 already fixes beta to one, so a beta-one ablation is recorded as an alias
   of current rather than repeated. No candidate changes downstream policies,
   network sizes, replay sizes, or fitting settings.

6. Select the lowest encounter-weighted equal-CPU MSE on validation only. Write
   `selection.json` before calculating any assessment targets. Evaluate only that
   candidate and current on assessment. Current wins exact ties.

All-action expansion is confined to the selected decision; it does not expand
the entire sampled game tree. Four-sample averaging reuses the current sample
bank for a paired comparison. Its independent-sample variance is analytically
one quarter at four times the cost, hence identical equal-CPU MSE. It cannot win
selection just because a small grouped variance estimate happens to look good.

## Measurements and interpretation

For each legal action, separate within-full-state continuation variation from
between-hidden-hand variation. Estimate mean-target error relative to the
independent reference, with uncertainty from both sets of samples. Negative
between-hand estimates are retained diagnostically before nonnegative truncation.
The repeated continuations are not counted as independent hidden draws.

The primary measurement estimates MSE at the CPU budget required for 256
independent current targets at the same position. It combines estimated variance
at that budget with a noise-corrected, nonnegative squared-bias estimate. Also
report variance times CPU, paired changes in mean target, reference precision,
resolved action-ranking/sign errors, critic discrepancy, importance-correction
tails and local hypothetical next-target strategy changes. The latter are
**not fitted policies, equilibrium values or evidence of improved playing strength**.

Assessment intervals resample connected board components and paired hidden-hand
blocks, with Gaussian reference-mean uncertainty and shared range/fold covariance.
They are conditional on frozen P104. Squared-bias estimation remains difficult
with limited samples, so an unresolved result is explicitly inconclusive.
They condition on measured process-CPU costs; local scheduling/timing variation
is not separately bootstrapped, and target sampling excludes reference setup.
Reports retain street, seat, fold, encounter weights and coarse decision-family
labels; summary.json contains all per-position measurements.

A recommendation for a short matched continuation requires:

- at least 25% estimated equal-CPU MSE reduction;
- a positive lower endpoint of the primary 95% reduction interval;
- at least eight independent board components;
- each assessed legal-action mean agreeing with its reference to within 0.05 bb,
  including its pointwise 95% uncertainty;
- no street with more than 10% deterioration in its point estimate.

The bias and street rules are conservative screening criteria, not simultaneous
confidence guarantees. Failure to resolve a mean difference is not proof of
unbiasedness. Distinguish a sampling-posterior discrepancy from estimator bias:
estimator means are compared with the **matching production-posterior reference**.
The frozen audit cannot establish whether information-state weighting over past
training iterations is correct, or whether those older targets were accurate.

There is no Fell Omen 2 match because this experiment trains no candidate policy.
There is also no redundant rerun of the old 90-position river suite: its networks
would be unchanged. The new held-out river positions provide exact local checks.
A successful finding should lead to a separate short matched continuation and
fresh policy evaluation before changing production training.

## Compute and guards

Google Batch: `c4-standard-16`, eight processes with one Torch thread each,
200 GB balanced Hyperdisk, standard provisioning, no automatic task retries.
The Batch task has an eight-hour emergency limit, the coordinator seven hours,
and the audit at most 6.5 hours. Source/setup/publication consume the remaining
time. A six-position development pilot uses independent random streams and
projects the full workload conservatively before the main measurement begins.
If it will not fit, the experiment stops incomplete and publishes the pilot.
It never silently cuts reference samples or assessment size.

Allow roughly **three to six hours provisionally**; cloud pilot measurements,
not this estimate or reduced local smoke tests, determine feasibility.
Batch deletes its worker VM/disk; results remain in Cloud Storage. Partial
reports are uploaded periodically and the final upload uses existing retries.

## Launch

From any terminal:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p104-supervision-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_p104_supervision.py \
  --run-name "$RUN" \
  --output "outputs/${RUN}-launch" \
  --execute
```

Omit `--execute` to build/review the bundle without uploading or submitting.
Submission archives local source with a checksum; it does not depend on a later
Git checkout or an open terminal. Use a new run name for each submission.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name~hulhe-p104-supervision' --format='table(name.basename(),status.state)'
```

## Download analytical reports only

Set `RUN` to the submitted job name. This explicit file list excludes models,
logs, source bundles and raw diagnostic arrays. `summary.json` includes every
completed position's analysis; the other files document selection and feasibility.

```bash
RUN="YOUR-SUBMITTED-JOB-NAME"
DEST="/Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher/outputs/${RUN}-results"
URI="gs://clever-overview-399515-fhp-escher-results/hulhe/${RUN}"
mkdir -p "$DEST/audit"
gcloud storage cp "$URI/cloud_status.json" "$URI/provenance.json" "$DEST/"
for FILE in summary.json report.md protocol.json panel.json selection.json runtime_projection.json; do
  gcloud storage cp "$URI/audit/$FILE" "$DEST/audit/"
done
```

Incomplete jobs may not have selection or a runtime projection. Read their
cloud status and partial summary first. Optional per-position `.npz` files are
raw analytical measurements (not networks), needed only to recompute statistics.
