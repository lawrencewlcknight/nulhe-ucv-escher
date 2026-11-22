# Captured regret-target diagnostic

This investigation uses the **existing P32 captures** from
`hulhe-positions-p32-20261118-181957`, taken during iteration 1133, and the
original P32 model. It performs no new self-play, changes no production training
configuration and does not resume the P56→P80 experiment.

## What is measured

For each player separately, group exactly equal encoded information states and
require identical legal masks. For a deterministic prediction, the observed
squared error decomposes exactly into:

```
row-weighted MSE = within-state target variance + error predicting each sampled state mean
```

The calculation uses legal actions only. Reports separate streets, repeated
states, singletons and multiplicity bins (1, 2–4, 5–9, 10+). The same frozen
labels are compared with captured predictions before and after the original
production fit. Duplicate replay row indices and inconsistent predictions or
masks are rejected. All 4,096 captured rows per player are used.

**This is an empirical identity, not proof of an irreducible noise floor.**
The sample mean itself is uncertain. We also report the estimated variance of
the sample mean under an IID assumption, and the signed difference between
mean-target MSE and that estimate. Traversal labels can be correlated, and
post-fit predictions have seen the labels; these are sensitivity diagnostics,
not unbiased post-training risk estimates or confidence intervals. Singletons
have no within-state variance estimate and cannot establish accurate targets.

## Conditional fitting comparison

An exploratory budget gate permits fitting when both players have at least
30 repeated preflop groups and 100 repeated rows, and either player's post-fit
empirical mean-target MSE accounts for at least 10% of repeated-row error.
This gate is deliberately labelled a heuristic, not a statistical test for
insufficient capacity. A residual comparable to the IID mean-noise scale is
particularly weak evidence of a capacity problem. `--fits off` runs only the
decomposition; `--fits force` bypasses this budget gate explicitly.

Two fitting arms, each with seeds 3901 and 3902:

- **warm_current:** refit copies of the source regret networks.
- **expanded_current:** retain the source networks and add a trainable residual
  correction network. The correction has three 128-wide branches, a 384-wide
  trunk, two residual blocks and a 256-wide head with zero-initialized output.
  Its initial predictions must match the source exactly, so the capacity
  comparison does not require relearning the source strategy from scratch.

Both arms use fresh Adam optimizers, learning rate 0.0003, gradient clipping at
10, up to 500 updates per player, and uniform-row batches up to 1,024. All
captured streets contribute training targets, protecting against fitting only
preflop and ignoring river examples. Loss is masked three-action MSE. The
existing production optimizer is not continued; this is a supervised diagnostic.
The source weights are those **before** the captured production iteration:
post-fit predictions were saved, but post-fit network weights were not.

Exact information-state groups are assigned by a target-independent hash to
60% training, 20% validation and 20% audit. Duplicate states never cross these
partitions. They are held out from these refits, not necessarily from the
source model's historical training. Training, validation and audit errors are
reported by street and multiplicity.

Validation chooses among updates 0, 100, 250 and 500. Update zero allows the
source model to remain preferable when fitting hurts generalization. Select the
arm with lowest mean preflop validation mean-target MSE across the two seeds;
ties prefer the original architecture. Export the first predeclared seed,
never the luckier seed. Selection is written before audit or river inspection.
Parameter counts, fitting time and selected update budgets are reported.
If update zero wins, the source model remains the selected candidate; the
experiment explicitly records that additional fitting did not improve validation.

## River decision check

Evaluate the selected diagnostic **instantaneous regret strategy** and the
source instantaneous regret strategy on every development river position in
the existing common-position library. Assessment positions remain locked.
The unchanged source average policy supplies the opponent and its exact
history-conditioned hidden-hand posterior for both candidates. This also
applies to roots originally encountered against Fell Omen 2: these scores are
not new matches against Fell Omen 2.

Report action probabilities, exact conditional values, local decision gaps,
remaining-response gaps and privacy/suit checks, with encounter-weighted
paired summaries separately for self-play and FO2-encountered positions.
Positive conditional value change is better; negative local-gap change is
better. These are conditional **bb**, not bb/100, exploitability or independent
playing-strength evidence. River outcomes are inspected after candidate
selection and cannot select a different model. A lower training loss alone
never counts as a strategic improvement.

When the selected strategy is exactly unchanged, the tool verifies matching
architecture, tensors, encoder and regret-matching settings, then checks actions
and invariants on every common river root. Paired value/gap changes are exactly
zero by strategy identity, so expensive hidden-hand integration is not repeated.
Absolute values are not invented or reported in that case. Captured-row and
training-partition overlap counts are recorded for each river root.

The saved `selected_regret.pt` contains diagnostic models only. It is neither
an average-policy export nor a full resumable training checkpoint.

## Run locally

Supply the downloaded four `regret_*_before.npz` / `regret_*_after.npz` captures
and `report.json` from `upstream/iteration`, the original policy, and the
`encounters` directory containing `library.json` and `report.json`:

```bash
python -m hulhe_escher.regret_diagnostic \
  --captures outputs/hulhe-positions-p32-20261118-181957-results/upstream/iteration \
  --policy outputs/hulhe-plus24-20261116-063535/training/policy_001132.pkl \
  --encounters outputs/hulhe-positions-p32-20261118-181957-results/encounters \
  --state outputs/policy-sampler-source-p32 \
  --output outputs/regret-target-diagnostic \
  --threads 4
```

The state directory needs only the original manifest and checksum-verified
metadata, not replay shards. Omit `--state` and use `--fits off` for the small
analysis alone. If fitting is warranted but metadata is absent, the tool writes
`NEEDS_METADATA` and exits nonzero rather than claiming the experiment completed.
Every run requires a new output directory.

## Google Batch

The launcher downloads only the existing captures, encounter library, policy,
and source metadata. It pins the policy and capture-report hashes, verifies
source identities, and never downloads the full replay or recollects data.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-regret-targets-p32-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_regret_diagnostic.py \
  --run-name "$RUN" --output "outputs/$RUN-launch" --execute
```

Omit `--execute` to inspect the local bundle. The job uses c4-standard-16, eight
fitting threads, **zero collection workers**, a 50 GB disk, no automatic job
retry, a one-hour diagnostic guard, and a two-hour emergency task ceiling
including setup/transfers. This is a bounded small-data investigation, not an
hour-based training run. Actual fitting and river timings are recorded; an
unfinished stage is labelled incomplete. Existing storage retries remain active.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment=hulhe-regret-targets' \
  --format='table(name.basename(),status.state)'
```

Results are in `diagnostic/report.md`, `diagnostic/report.json`,
`diagnostic/selection.json`, and `cloud_status.json`. Partial river output is
preserved on timeout. Source runs are read-only; the diagnostic can support a
later training-specification decision but makes no automatic production change.
