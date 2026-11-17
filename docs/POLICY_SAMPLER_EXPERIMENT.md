# Frozen-replay policy sampler experiment

This experiment needs **no new self-play training**. Refit average policies on
the saved P32 reservoir, changing only grouped-minibatch sampling. The playable
P32 policy alone is insufficient: we also need its saved policy replay. Nothing
in this experiment updates regret networks, critics, replay or the live run.

## Preregistered comparison

Two arms, with matched initial weights for fit seeds 1701 and 1702:

* `uniform`: existing uniform sampling without replacement within each batch,
  using the original per-group loss weights.
* `objective_mass`: independent draws with replacement proportional to each
  group's accumulated replay weight, using a constant compensating loss factor.

For training groups with weights `w_i`, the existing objective is
`sum(w_i * loss_i) / number_of_training_groups`. With
`p_i = w_i / sum(w)`, each sampled loss is multiplied by the **mean training-group
weight**. Do not multiply by `w_i` again. The validation partition matters to this
normalization. If the entire training set fits in a batch, both arms compute the
exact full-batch objective.

Keep the source architecture, Adam learning rate (P32: 0.003), batch size (2048),
gamma (2), 20,000-update budget, existing 5% information-set-disjoint validation
split and checkpoint-selection schedule unchanged. Both arms start afresh,
matching the source's reset-each-fit contract. Do not introduce a lower learning
rate, bigger network, full-data final refit or other changes in this experiment.

Report weighted cross-entropy, target entropy, KL and action-probability L1 error
for training and validation, overall and by street; report fit times and sampled
preflop counts/weight variability. Validation is used to select updates, so it
is not an independent test set and its improvement alone is not a strength claim.
All fit seeds go into gameplay evaluation; do not select the most favorable seed.

## Required artifacts and resources

Verified P32 source:

* `gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535/training/state_001132`
* Matching policy: `training/policy_001132.pkl` under the same run prefix.
* Policy SHA-256: `9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b`.

Only four replay arrays are needed: 4,000,000 x 417 features, three-action soft
targets, three-action masks and iteration weights. These occupy 6,784,000,512
bytes (about 6.8 GB decimal), plus about 19 MB metadata. The loader verifies
SHA-256, shapes, encoder, iteration, source configuration and saved policy
weights, then memory-maps replay read-only. It does not allocate critic buffers,
start collection workers or restore the whole solver. The downloaded subset is
explicitly **not a complete resumable training checkpoint**.

Prefer a separate CPU machine with 32 GB or more RAM and at least 12 GB free disk.
The existing c4-standard-16 class is suitable; use eight Torch threads and run
fits sequentially. Do not compete for CPU with the ongoing continuation. No GPU
is required. Four policy fits should cost far less than 24 hours of self-play,
but wall time must be measured. The 2-hour evaluation cap below is separate
from fitting and transfer time; fitting uses a fixed update budget for fairness.

## Prepare and fit

### One-command Google Cloud launch

The launcher needs only Python 3 and authenticated `gcloud` locally. It uploads
an allowlisted source archive and starts a separate `c4-standard-16` in
`europe-west1-b`. The VM installs dependencies, verifies the P32 policy, downloads
only policy replay, prepares FO2, runs all four fits and evaluates them with
eight workers. Results are uploaded to the run's own prefix every two minutes
(raw hand journals on final upload), with retries for transient transfer errors.
The final status is published after the results. The VM shuts down on exit,
with a six-hour safety cap and a 50 GB boot disk. The stopped disk remains
billable until explicitly deleted after results have been verified.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-sampler-p32-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_policy_sampler.py \
  --project clever-overview-399515 \
  --zone europe-west1-b \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --run-name "$RUN" \
  --source-uri gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535 \
  --expected-policy-sha256 9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b \
  --iteration 1132 \
  --output "outputs/cloud-$RUN" \
  --execute
```

Omit `--execute` to prepare a local launch bundle without creating a VM. Use a
fresh run name/output directory for a subsequent launch. The source snapshot
includes the current working copy; never package unrelated unfinished changes.
Automatic shutdown alone does not imply success: inspect `cloud_status.json`
and `evaluation/report.json`. Installation logs are in `startup.log` and serial
console output. The six-hour cap is a safety limit, not an expected runtime.

Headline VM status from any terminal:

```bash
gcloud compute instances list --project=clever-overview-399515 \
  --filter='name~hulhe-sampler-p32' --format='table(name,status)'
```

### Manual execution

Run inside this repository with its installed Python environment. These commands
download only the policy subset, not hundreds of analytical/training files:

```bash
python -m hulhe_escher.policy_sampling_experiment prepare \
  --source-state gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535/training/state_001132 \
  --output outputs/policy-sampler-source-p32

gcloud storage cp \
  gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535/training/policy_001132.pkl \
  outputs/policy-sampler-source-p32/policy_001132.pkl

python -m hulhe_escher.policy_sampling_experiment fit \
  --state outputs/policy-sampler-source-p32 \
  --policy outputs/policy-sampler-source-p32/policy_001132.pkl \
  --output outputs/policy-sampler-p32 \
  --seeds 1701 1702 --threads 8
```

Use `prepare --metadata-only` to inspect the download plan without fetching the
large arrays. Existing verified array downloads are reused. Only load trusted
checkpoint metadata: as with the recovery loader, it uses PyTorch/pickle.

If P32's policy is already local, pass that existing file to `--policy` instead
of copying it again. Output directories must be new; completed experiments are
not overwritten. Each fit is saved immediately. A failed partial experiment
is marked `FAILED` and is not eligible for gameplay evaluation.

Outputs include `experiment.json` and four playable checkpoints:
`uniform_seed1701.pkl`, `objective_mass_seed1701.pkl`, and the same for seed 1702.
They retain source training provenance and add explicit refit metadata, including
initial-weight hashes, source-policy hash and implementation hashes. These are
policy-only exports, not full training checkpoints.

## Gameplay evaluation

Install/verify the Fell Omen 2 assets through the existing setup instructions.
Then run:

```bash
python -m hulhe_escher.policy_sampling_evaluate \
  --experiment outputs/policy-sampler-p32 \
  --output outputs/policy-sampler-p32-evaluation \
  --workers 8 --pilot-pairs 10000 \
  --minimum 100000 --maximum 1000000 --max-seconds 7200
```

For each shared deal, play both seats for:

1. Corrected versus uniform policy within each fit seed.
2. Every policy against Fell Omen 2, using shared action-randomness streams
   within a matched refit pair.

Average across seats, then across the two fit seeds **within each deal group**.
The two predeclared tests are mean head-to-head score and mean improvement
against Fell Omen 2. Use Bonferroni-adjusted 97.5% intervals (`alpha=.025` each).
The independent statistical observation is a deal group, not each hand or each
fit seed. With two fit seeds, each group contains twelve hands.

An excluded 10,000-group pilot estimates variance for detecting a 2 bb/100 effect
with approximately 80% power. Fix the confirmatory sample size before seeing
confirmatory results. Use 100,000 to 1,000,000 groups, constrained by measured
throughput and the time cap. Report caps/incomplete runs explicitly; do not claim
a powered null result when the budget is insufficient. No significance-based
early stopping. Absolute FO2 scores are descriptive.

The FO2 adapter uses the same explicit free-fold-to-check handling as prior
experiments. Results are conditional on these fixed policies and this one
training replay. Two refit seeds do not constitute two independent self-play
runs. Better replay fit need not improve play; disagreement between the two
gameplay endpoints should be reported rather than selecting the favorable one.

## Verification and interpretation

```bash
python -m pytest tests/test_policy_sampling_experiment.py tests/test_replay_grouping.py -q
```

Tests check exact expected loss/gradient equivalence on a held-out partition,
sampling frequencies, unchanged legacy RNG/model updates, checksum rejection,
selective replay loading, matched initialization, playable exports, and the
end-to-end evaluator on tiny synthetic replay. `fit --smoke` and evaluation
against `check_call`/`random` are explicitly non-confirmatory.

This experiment answers whether sampler correction improves policy extraction
from learning already completed. It does not establish its effect on a different
network, different learning rate, or future self-play training. The production
default remains uniform until the experiment supports a deliberate change.
