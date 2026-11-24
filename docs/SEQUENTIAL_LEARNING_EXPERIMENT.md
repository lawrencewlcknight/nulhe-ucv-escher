# Protected sequential Limit Hold’em: four plus four hours

This is a fresh eight-active-hour self-play experiment using the proposed protected
sequential network for both player regret networks and the average-policy network.
It does not resume P104 or use the frozen-target diagnostic's fitted weights.

The protocol follows `LEARNING_EXPERIMENT.md` in both the limit and no-limit repos:
P4, a full save and process restart, four more active hours, P8, fixed-policy
comparisons and an independent external benchmark. The saved original Limit run
(`outputs/results-4plus4-20261115/experiment.json`) selected eight collection
workers and eight central Torch threads. We retain those settings, seed 20261115,
and c4-standard-16 rather than repeating hardware selection on this experiment.

## Architecture and learning

Each regret network and the average-policy network has 1,496,067 parameters:
three 128-wide feature branches, a 384-wide entrance, then four sequential stages.
Each stage contains a shared 384→192→384 residual transform and a specialist
384→224→384 residual transform. LayerNorm/ReLU precede linear layers, residual
scale is 0.5, and a 768→3 output reads the entrance and final representations.
All four stages execute for every example; there is no street gating. Showdown
is terminal and does not have an action-selection specialist.

Every fit starts with 200 unrestricted updates, included in its existing update
budget. Subsequent updates train the shared weights and the specialist for the
sampled street. Other specialists have `grad=None`, preventing Adam momentum from
moving them, but gradients still flow through their activations. The brief warm-up
is repeated per fit; it is not a separate period of self-play. Smoke fits shorten
it to leave at least one protected update.

Protection coefficient is **0.3**, a predeclared experimental choice rather than
an established optimum. Each protected update draws an additional quarter-batch
(512 for a 2,048 primary batch) from other streets, using training data only:

- Regret: supervised rehearsal against the same current frozen regret targets.
- Average policy: KL(reference || current), with a frozen reference initialized
  after warm-up and refreshed every 50 protected updates. The reference exists
  only during this fit and is not a production inference ensemble.

Primary streets are sampled in proportion to objective mass; examples are sampled
with replacement conditional on that street. Regret rows have equal mass.
Policy groups use accumulated replay weight and a constant loss factor preserving
the corrected weighted objective's scale. There is no second weighting by replay
mass. Protection uses the objective conditional on the other streets. If replay
contains only one street, the unavailable auxiliary update is skipped and counted.
This changes minibatch variance and adds a protection objective; it is an
experimental learning change, not an identical-algorithm speed optimization.

Keep baseline collection volume, replay capacities, regret/critic update budgets,
discounting and critics/calibration. In particular, policy replay remains four
million rows and regret fitting remains 750 updates. Regret learning rate stays
0.001; average-policy learning rate is 0.0003. Policy fitting resets its model and
optimizer as before, runs up to 20,000 updates, and chooses among 2,500/5,000/10,000/
20,000 using the existing information-set-disjoint validation partition. Validation
examples remain excluded from fitting, including protection. This experiment does
not also introduce an all-replay policy refit. Gradients are clipped to norm 10.

Per-fit reports record warm-up, selected street counts, protection updates/loss,
and the policy sampling seed. Only the opt-in `protected_sequential_v1`
configuration uses these changes; existing plain/deep/wide/hybrid runs retain
their fitting paths.

## Checkpoints and comparisons

Four hours means active collection and regret/critic/calibration fitting. Policy
extraction, saving, full recovery verification, uploads and evaluation are excluded.
Each stage finishes its current iteration, so actual active time can exceed four
hours. P8 is four additional hours after P4, just as in the original experiment.
Both playable policies are preserved. The newest full recovery replaces the older
one only after successful publication; P8 can resume later with the archived source,
identical runtime/configuration and worker contract.

Both full checkpoints are restored in a fresh interpreter and checked against
playable policy weights before publication. The second training stage separately
uses `--resume`. Recovery includes replay, models, optimizers and random states.

The existing evaluation runners are reused without changing their statistical design:

- Primary: P8 vs P4, duplicate deals with policies swapping seats. Exclude a 10,000
  pair pilot; freeze 100,000–1,000,000 confirmation pairs, aiming for 80% power for
  2 bb/100. Report a 95% interval across pair means. Runtime caps may reduce power.
- Controls: P4/P8 vs random and check/call, 1,000 pilot and 10,000 confirmation pairs
  each, descriptive only. Local evaluation has a two-hour budget, 80% for primary.
- Fell Omen 2: both checkpoints play both seats on shared deals, using the original
  free-fold-to-check adaptation. Exclude a 10,000-group pilot; freeze 100,000–1,000,000
  confirmation groups. Report each absolute score and the paired P8-minus-P4 change.
  Overall confirmatory improvement remains gated on the primary test passing.
- No significance-based early stopping. An incomplete evaluation is reported as
  incomplete and makes its Batch worker fail, even when training was successful.

This supports comparison with the original Limit run's P4/P8 scores and learning
progress. It is not a direct match against historical P4/P8 policies. Architecture,
protected fitting, policy learning rate, corrected sampling and subsequent code
fixes differ from the original recipe, so it is not a capacity-only ablation.
Equal active hours are not equal samples or FLOPs. A single training seed cannot
establish reproducibility across training seeds, and no-limit scores are for a
different game rather than a directly comparable playing-strength scale.

## Cloud execution

Two independent Google Batch jobs in europe-west1, with no automatic retries:

| Role | Resources | Limits |
|---|---|---|
| Train and local evaluation | c4-standard-16, 200 GB, 8 actors / 8 central threads | 2h full-capacity preflight, 12h training wall budget, 2h local evaluation, 18h task ceiling |
| Fell Omen 2 | c4-standard-4, 50 GB, 4 workers | 26h including setup/waiting, 28h task ceiling |

The evaluator begins waiting immediately and can run the P4 pilot while P8 trains.
It does not compete for the trainer's CPUs. A full-capacity memory/grouping/recovery
probe uses the new architecture before production collection begins. Batch releases
VMs/disks on completion; checkpoints and analytical reports remain in GCS. These
are ceilings, not estimates or guaranteed power. Launching is explicit.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-sequential-4plus4-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_sequential_learning.py \
  --run-name "$RUN" --output "outputs/cloud-$RUN"

# Submit the exact prepared source and plans after reviewing launch.json.
python3 tools/launch_sequential_learning.py \
  --output "outputs/cloud-$RUN" --submit-prepared
```

The launcher uploads an allowlisted source archive, excluding secrets, outputs,
models and `.git`. The two jobs use its verified checksum. A unique GCS prefix is
claimed before submission. If submission of the trainer fails after the evaluator
was created, that evaluator is deleted. If a submit call has an ambiguous network
failure, inspect Batch before trying again with another name.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name:hulhe-sequential-4plus4' --format='table(name.basename(),status.state)'
```

Analytical downloads only (substitute the run name):

```bash
RUN=hulhe-sequential-4plus4-YYYYMMDD-HHMMSS
mkdir -p "outputs/$RUN-results"
gcloud storage rsync --recursive \
  --exclude='(^|/)(training|inputs|state_[0-9]+)(/|$)|^(?!.*\.(json|md)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN" \
  "outputs/$RUN-results"
python3 -m hulhe_escher.experiment_report --experiment "outputs/$RUN-results"
```

This excludes model files, replay, logs and per-hand journals; checkpoint markers
and consolidated reports retain the metrics needed for analysis. The coordinator
also publishes per-iteration JSON metrics separately in its analytical subtree;
training-state directories remain excluded.

## Local verification

```bash
python -m pytest tests/test_sequential_learning.py tests/test_sequential_frozen.py -q
python -m hulhe_escher.sequential_experiment --smoke \
  --output outputs/sequential-learning-smoke
```

Smoke retains full network size with tiny replay and fitting budgets. It exercises
collection, protected regret/policy fitting, full recovery, actual process restart,
policy loading, all local matches and a check/call proxy for the external paired
benchmark. It never launches cloud resources or downloads FO2 assets and cannot
support a poker-strength claim.
