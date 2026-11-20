# P80 position audit and paired P56 comparison

Replicate the completed `hulhe-positions-p56-20261119-143942` diagnostic at
P80, iteration **2,711**, then compare P80 and P56 on the frozen P56 development
positions. This is a diagnostic Google Batch job, separate from P80→P104 training.

The dedicated launcher pins these inputs:

| Input | Source / SHA-256 |
| --- | --- |
| Full P80 recovery | `hulhe-p56-plus24-batch-20261119-144002/training/state_002711` |
| P80 policy | `822e1d499ce2688f362e46595220b341dbff6adf8546e9ca65a2e459f4020e8c` |
| P56 audit | `hulhe-positions-p56-20261119-143942` |
| P56 policy | `35b5348443678a2738d8ae52b1b313370739be90b1e0c298d3f73afd72adc30e` |
| Frozen P56 library | `ee1e877034f84335282d87e620f1f0295a046c2efde0427cdc93cbe4f42c065e` |

Both exported policies already use corrected objective-mass sampling of grouped
average-policy targets. The isolated upstream iteration uses P80's exact archived
trainer, runtime, configuration, optimisers, replay and RNG state. It introduces
no new network architecture, sampling correction or policy refit. Source files
are read-only; no resumable production checkpoint is created or replaced.

## Replicated diagnostic

Retain the [original protocol](POSITION_INVESTIGATION.md):

- Collect 20,000 P80 self-play hands and 20,000 hands against the pinned Fell
  Omen 2 bridge. These measure encountered positions, not confirmatory win rates.
- Select 128 river and 128 earlier-street positions before scoring their errors.
  Use the same event sampling, family frequencies and inclusion weights.
  Score development only; the assessment partition remains locked.
- Scan retained policy replay for exact selected-state matches and broader
  family coverage. Report objective-weighted KL/L1 by family and historical
  training/validation split. Missing exact matches are not imputed.
- Compare the exported policy and instantaneous regret strategy. On development
  rivers, enumerate compatible hidden hands and the remaining betting tree
  against frozen P80 with its history-conditioned range, plus nine synthetic
  opponent/range panels. Run the fixed strategic regression suite separately.
- Check 24 upstream river probes, using 64 traversals per critic fold, against
  independently integrated conditional action advantages.
- Run one isolated ordinary iteration and capture actual regret/critic targets
  and prediction errors before and after fitting, capped at 4,096 rows per learner.

The encounter seed and selection rules remain those of the earlier audit for
replication. P80's new encounter distribution is reported separately from the
fixed-position comparison. Legacy analysis fields named `p32_opponent` and
`regret_against_p32` are retained for compatibility; `learned_opponent_label`
and the policy hash identify P80 in the new audit and P56 in the archived audit.

## Like-for-like comparison

Evaluate both exports on the **185 frozen P56 development positions (90 rivers)**, keeping
the P56 learned opponent and its history-conditioned posterior identical for
both candidates. Recompute all nine synthetic panels using the same ranges.
P56 baseline scores must reproduce the archived analysis within numerical
tolerance before results are accepted. The comparison never unlocks assessment.

Report remaining-river response-gap reductions, local gaps, improved/regressed
counts and source/seat/family breakdowns, with both inclusion-weighted and
unweighted averages. Also isolate the current action mixture using fixed P56
continuation values. Earlier streets report action changes without claiming an
oracle improvement. The fixed regression suite scores P56, P80 and P80's
instantaneous regret strategy.

These are descriptive conditional **bb per decision**, not bb/100,
exploitability or independent population tests. Correlated positions do not
receive invented confidence intervals. The audit does not attribute changes to
a single cause, infer unseen-state visitation from reservoir absence, or select
a replacement policy. Historical validation was already used in policy fitting.

## Compute and launch

One `c4-standard-16`, eight collection/upstream workers, one analysis thread,
150 GB disk. Keep the existing stage budgets: 30-minute encounters, three-minute
regression, 50-minute analysis, 30-minute paired comparison and 20-minute upstream
probe work budget within a 25-minute restore/iteration process guard. The task
has a four-hour emergency limit and no automatic whole-job retries. Aim for
roughly one to two hours, subject to cloud measurement; incomplete stages are
reported rather than silently shrinking the audit. Batch releases its VM/disk.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-positions-p80-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_p80_position_audit.py \
  --run-name "$RUN" \
  --output "outputs/$RUN-launch" \
  --execute
```

The defaults use project `clever-overview-399515`, region `europe-west1`, the
existing runner service account and `gs://clever-overview-399515-fhp-escher-results/hulhe`.
Omit `--execute` for a local review bundle only. Use a fresh run name and output
directory. Submission uploads the current coordinator source; it does not alter
the archived production trainer.

Status from any terminal:

```bash
gcloud batch jobs list \
  --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment=hulhe-positions AND labels.checkpoint=p80' \
  --format='table(name.basename(),status.state)'
```

## Analytical download

Set `RUN` to the submitted name. This excludes replay, model payloads, raw
encounter journals and optional NPZ target captures.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
mkdir -p "outputs/$RUN-results"
gcloud storage rsync --recursive --exclude='^(?!.*\.(json|md|html)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN" \
  "outputs/$RUN-results"
```

Read `report.md`, `summary.json`, `comparison/report.md`, `comparison/report.json`,
`analysis/replay_coverage.json`, `upstream/probes/report.json`,
`upstream/iteration/report.json` and `regression/summary.json`.
`comparison_inputs.json`, `provenance.json` and the launch manifest record input
identities; baseline reproduction errors are in the paired comparison report.
