# P32 common-position investigation

This is a diagnostic Google Batch job, not another long training experiment. It
reuses the immutable full P32 recovery and its exported policy, generates frozen
policy encounters, traces selected decisions through replay and networks, and
checks the local traversal estimator against independently integrated river values.
No production configurations, source checkpoints, or live jobs are changed.

## Fixed protocol

1. Play **20,000 hands of P32 self-play and 20,000 hands of P32 versus Fell Omen 2**.
   Alternate P32 seats in the latter; record only P32 decisions. Self-play records
   both seats. Use the existing pinned Java FO2 bridge and explicit free-fold-to-check
   adaptation. Record adaptation counts and verify the asset build before/after.
   These hands identify encounters, not a confirmatory win-rate test.
2. Select **128 river and 128 earlier-street positions** before evaluating their
   errors. Earlier quotas are 42 preflop, 42 flop and 44 turn. Within each street,
   split quotas across matchup and seat, selecting uniformly over decision events
   with deterministic bottom-k hashes. Repeated information states remain separate
   events. Record full family frequencies over all encounters and the selected
   events' inverse inclusion weights. Worker merge order cannot change selection.
3. Freeze and hash the visible-information-only library. Roughly 25% of card-orbit
   groups are locked assessment. Entire flop suit orbits, including later-street
   extensions, stay together; preflop uses starting-hand orbits. Existing suite
   assessment boards are also protected. The normal job scores development only.
4. Scan the complete retained policy replay, using the production exact grouping
   and objective weighting. For exact selected-state matches, report row count,
   grouped target, accumulated objective weight, policy KL and historical split.
   Report replay coverage and weighted KL/L1 by coarse family and historical
   training/validation split. No approximate neighbour is substituted for a missing
   exact target. A family is street/seat/made-hand category/facing-bet/raise availability;
   preflop uses pocket-pair versus unpaired. It deliberately groups many different
   situations, not equivalent information sets.
5. Compare P32 policy and instantaneous regret policy on identical positions,
   recording raw regret outputs. Enforce valid probability/privacy/hole-order
   contracts and report suit invariance. Earlier streets have no invented answer
   key. Retain the existing development strategic suite as a separate regression
   panel, with its own uniform/synthetic range assumptions.
6. On each development river root, enumerate all **990 compatible opponent hands**
   and the full remaining betting tree. Score all nine existing synthetic panels.
   Additionally use frozen P32 as the opponent, conditioning its hidden hand range
   on the entire observed opponent betting history. Evaluate each earlier action
   with only cards then visible; uniform chance likelihoods and hero action
   likelihoods cancel conditional on the hero's known cards. Zero support is
   explicitly unsupported. Integrate hidden hands before selecting a hero response.
   Compare both exported P32 and the instantaneous regret strategy against this
   same frozen P32 opponent.
7. Freeze up to **24 development river probes**: half with the largest conditional
   P32 action gaps, half low-gap controls. This selection is made before inspecting
   traversal estimates. Restore the full checkpoint in an isolated process with
   the **archived original trainer, exact runtime, optimizers, replay and RNGs**.
   For these probes, use frozen instantaneous regret strategies for both players,
   derive a fresh opponent posterior, and compare exact action advantages with
   **64 independently sampled traversals per critic fold**. Unsupported histories
   are recorded rather than replaced by uniform ranges.
8. Run **one ordinary production iteration**. Capture actual regret/critic training
   targets and before/after predictions on up to 4,096 training rows per learner,
   as in the earlier bottleneck investigation. Also record cumulative regret
   predictions before/after on the fixed probes and their exact occurrence counts
   in that iteration's regret replay. Save no resumable continuation checkpoint.

## Interpretation and statistical limits

The local estimator comparison starts from a **unit root**, just as this solver's
`dfs` does: no prefix reach weight is supplied. Draw hidden opponent hands from
the frozen regret opponent's history-conditioned posterior. Convert exact values
from big blinds to the solver's utility normalization, then subtract the current
strategy's value to obtain expected regret increments. This validates a local
conditional increment; it does **not** validate the distribution/weighting of roots
reached by a full self-play traversal.

Report means, standard errors, and approximate t intervals with a Bonferroni
allowance across selected probes, folds and three possible actions. Traversals
are independent within a frozen root/fold; extreme tails can still impair t
interval coverage. Flags need inspection/replication, not an automatic bias claim.
Replay, critic-controller and calibration writes are disabled during probes;
RNGs, counters, active fold and diagnostic aggregates are restored even on error.

The synthetic probe target is discounted/clipped previous target-network output
plus the independent expected local increment. It is separate from actual training
targets captured during the iteration. Sequential player updates can change
production supervision relative to the frozen pre-iteration probe. A changed
probe prediction is not evidence that the probe itself was trained on.

The source policy's historical validation pool was used for selection: it is not
an untouched independent test. The new assessment partition is held out from this
diagnostic-driven tuning, not guaranteed absent from historical self-play.
Never compare instantaneous and historical average policies as if they must agree.

Conditional response gaps against frozen P32 or synthetic models are **bb per
decision**, not bb/100, equilibrium errors or exploitability. Family frequency
times conditional gap is only a triage ranking. Do not sum overlapping decisions
as a whole-game loss estimate. No statistical confidence intervals are invented
for correlated audit positions or family means.

Use patterns to guide the next investigation:

| Pattern | Interpretation to investigate |
|---|---|
| Frequent family, few retained rows | Collection/replay coverage; reservoir absence does not prove no visitation |
| High policy error on historical training rows | Optimization or capacity |
| Low training error, higher historical-validation error | Generalization/representation; check small group counts |
| Local estimator mean disagrees with independent reference | Conditioning/implementation, then additional replication |
| Wide estimator intervals | Noisy supervision or weak variance reduction |
| Large residuals on actual training targets after fitting | Regret/critic fitting limits; bootstrapped critic labels are not ground truth |

## Runtime and cost controls

One `c4-standard-16`, 150 GB Hyperdisk Balanced, eight collection workers and the
source's eight training workers, CPU only. Exact audit and local probes use one
Torch thread to avoid tiny-matrix overhead. Target duration is one to two hours,
not a measured cloud guarantee.

- Encounter stage: 30-minute work budget, 31-minute process guard.
- Fixed regression suite: 3-minute work budget, 4-minute process guard.
- Replay and common-position audit: 50-minute work budget, 51-minute process guard.
- Upstream probes: 20-minute probe budget inside a 25-minute process guard that
  includes restore and one training iteration.
- Google Batch task emergency limit: four hours, including installation/downloads.
- No automatic retries of the complete expensive task. Storage transfers retain
  existing retries. Batch releases its VM and disk when the task finishes.

The first one and four learned-opponent river checks produce timing projections.
Timing does not silently shrink the benchmark or change selection based on errors.
Unfinished diagnostics are marked **INCOMPLETE**, preserved and uploaded; genuine
errors are **FAILED**. Stage failures stop later dependent work.

## Build or launch from any terminal

The launcher packages the current local source, so it does not require a Git push
to run. Review/commit the code before running if desired. Remove `--execute` to
only create a local reviewable bundle. An existing output directory/run prefix is
never overwritten. Original P32 inputs are read-only.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher

RUN="hulhe-positions-p32-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_position_investigation.py \
  --project clever-overview-399515 \
  --region europe-west1 \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --run-name "$RUN" \
  --source-uri gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535 \
  --expected-policy-sha256 9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b \
  --iteration 1132 \
  --output "outputs/$RUN-launch" \
  --execute
```

Headline status, independent of shell variables:

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment=hulhe-positions' \
  --format='table(name.basename(),status.state)'
```

Analytical results only (set `RUN` to the submitted job name):

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
mkdir -p "outputs/$RUN-results"
gcloud storage rsync --recursive --exclude='^(?!.*\.(json|md|html)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN" \
  "outputs/$RUN-results"
```

The main output is `report.md`/`report.html` with `summary.json`. Detailed analytical
files are `encounters/report.json`, `encounters/library.json`,
`analysis/report.json`, `analysis/replay_coverage.json`, `analysis/timing.json`,
`analysis/upstream_positions.json`, `upstream/probes/report.json`,
`upstream/iteration/report.json`, `upstream/iteration/family_errors.json` and
`regression/summary.json`.
NPZ target samples are optional downloads; source checkpoints are never uploaded
to the result prefix. Cloud status, provenance and per-stage resource use are saved.

## Small local checks

Use a Python environment with repository dependencies. A small encounter run can
use `--opponent check_call` (explicitly labelled smoke) instead of FO2:

```bash
python -m hulhe_escher.position_investigation.collect \
  --policy /absolute/path/policy.pkl --opponent check_call --hands 100 \
  --workers 1 --river 8 --earlier 12 --max-seconds 120 \
  --output outputs/position-encounter-smoke

python -m hulhe_escher.position_investigation.analyse \
  --state /absolute/path/state_directory --policy /absolute/path/policy.pkl \
  --encounters outputs/position-encounter-smoke --max-seconds 300 \
  --upstream-positions 2 --output outputs/position-analysis-smoke
```

For a later frozen candidate assessment, the analysis CLI accepts explicit
`--unlock-assessment` and a new output directory. The current diagnostic traces
the original policy against its own recovery; it is not a candidate-comparison
launcher. Freeze the next intervention before consuming the assessment partition.
