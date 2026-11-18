# P32 Limit Hold'em bottleneck investigation

A separate Google Batch diagnostic job reuses P32 learning. Production defaults and
ongoing continuation jobs are unchanged. It downloads exactly the full immutable
`training/state_001132` recovery from
`gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535`.
The policy must have SHA-256
`9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b`.
The original checkpoint is never written, pruned or republished.

## Questions and controlled comparisons

Six supervised fits use the same frozen targets, batch size, update budget,
encoder and objective-mass sampler. Two declared seeds (2701, 2702) are used for
all three configurations:

| Arm | Policy model | Learning rate |
| --- | --- | --- |
| Corrected reference | Existing structured branches 96, trunk 256–256 | 0.003, from source |
| Optimisation candidate | Identical reference architecture and initial weights | 0.001 |
| Capacity candidate | Branches 128; 384-wide projection; two residual blocks, each two 384-wide layers; 256-wide head | 0.001 |

The capacity candidate has 892,291 parameters, deterministic ReLU layers and
identity-initialised residual branches. There is no batch normalisation or dropout.
This tests a particular modest residual capacity increase, not depth independently
of width. Regret and critic production architectures are not enlarged here.

Each fit runs 20,000 updates, checking 2,500/5,000/10,000/20,000 as in the source
contract. There is no automatic longer fit, hyperparameter sweep or retraining on
the full dataset after selection. Saved policies preserve the chosen update.

The original canonical-feature 5% holdout is divided by an independent fixed hash
into roughly 2.5% selection validation and 2.5% untouched audit. The remaining
95% trains. Group-identical states cannot cross partitions. All arms and seeds
share the exact split; split hashes and counts are recorded.

Select the architecture/rate with the lowest **mean validation weighted KL over
both seeds**. Resolve exact ties in favour of the reference, then the smaller
low-rate model. Export the **first predeclared seed** of that arm for evaluation;
do not choose a lucky seed. Freeze `selection.json` before computing audit scores
or running matches. Audit is for reporting, not switching candidates. The source
P32's audit loss is descriptive: P32 may already have trained on these groups.

For each split, report weighted KL, target entropy, action L1 error and argmax
disagreement overall, by street, by retained replay frequency (1, 2–9, 10–99,
100+ raw occurrences), and by their intersection. Frequency is not weight mass.
Rare/empty cells are shown explicitly. These metrics compare with replay targets;
low KL does not establish that the targets themselves are strategically optimal.

## Exact-position and upstream probes

Six positions compare instantaneous regret strategy, historical replay target
(if present), P32 and all six fitted policies. Four are exact constraints:
private royal flush or a royal flush on the board, for both seats, facing a capped
river bet. Exhaustive enumeration of all 990 possible opponent hole-card pairs
checks that calling strictly dominates folding. AA and 72o preflop are unscored
contextual comparisons; no equilibrium action is invented. Missing replay groups
are reported as unavailable. Instantaneous regret and historical average policies
need not agree even in a correctly functioning solver.

In a separate process, restore the full checkpoint using its archived trainer,
original source fingerprint, runtime, worker configuration, optimizers and RNGs.
Run exactly **one normal production iteration** with 8 collection workers. Capture
up to 4,096 replay rows for each regret learner and each critic immediately before
that learner fits, preserving sequential player-update and critic-update order.
Save features, actions/masks, frozen labels and before/after predictions in small
NPZ artifacts. Report target tails, errors, collection throughput, fit durations,
importance corrections and estimator centering. Sample selection uses private RNGs.

The before/after errors are on training replay; they are not independent test
errors. Critic TD targets are bootstrapped labels, not ground truth. A large
residual can reflect approximation, noise or a wrong target and must be interpreted
with the exact-position probes and policy audit.

After that isolated iteration, run 64 repeats at three fixed full states (preflop,
flop, river), for both critic folds: **384 additional rollouts**. At the root, form
the normal estimate and a zero-root-control estimate using identical sampling
probabilities and identical downstream returns. Compare advantage variances,
paired means/standard errors, residuals and correction tails. Replay writes are
disabled and RNG/counters restored. This is exploratory conditional variance at
three full states, not a full no-control training ablation, information-set value
oracle, or proof of general variance reduction.

The probe writes no resumable next checkpoint. Refits always read original P32
replay, never replay altered by the isolated iteration.

## Fresh playing-strength test

Evaluate only the selected first-seed policy against P32 and Fell Omen 2; also
re-evaluate P32 against FO2 on the same fresh deals. This supports two predeclared
claims: candidate-vs-P32 and improvement against FO2. Use seat-swapped duplicate
deals and paired shared-deal differences, with Bonferroni-adjusted **97.5% intervals**
(alpha .025 each). FO2 uses the existing free-fold-to-check adaptation and its
build fingerprint is checked before and after play.

A fresh excluded 5,000-deal-group pilot estimates variance and speed. Fix the main
sample before playing it: 100,000–250,000 groups, targeting 80% power for 2 bb/100,
subject to a one-hour evaluation guard. Report actual planned detectable effect,
caps and incomplete results; never stop on significance. A complete 100,000-group
run has 600,000 main hands across the three matches. All inference is conditional
on this single P32 replay and selected policy, not independent self-play seeds.

## Compute and runtime

Use one Google Batch task on a **c4-standard-16** in `europe-west1`, eight
collection workers/evaluation workers, eight fitting threads and a 150 GB
Hyperdisk Balanced boot disk. No GPU. The earlier four current-network
fits took 8.3 minutes including diagnostics. Six fits, including two roughly
five-times-larger models, plus one iteration and fewer matches than that sampler
study should ordinarily fit in roughly **one to two hours including setup**.
This is an estimate, not a measured guarantee for the larger candidate.

There is no overall two-hour cutoff. Work is finite; guards catch stalls:
15 minutes for restore/one-iteration/probes, one hour for six refits, one hour for
matches (with process cleanup allowance), and a four-hour Batch task limit,
including setup. Time waiting for Batch capacity is additional.
Any exceeded stage is marked failed/incomplete, never silently called successful.
Collection timings measure the existing regret/critic networks. Policy inference
microbenchmarks do not establish collection throughput for larger regret networks.
A candidate's better supervised error alone does not establish better strength per
training hour. Results retain the measurements needed to assess that tradeoff.

Results upload periodically and on exit. Batch releases its VM and boot disk when
the task ends; retained GCS artifacts still incur storage charges. There are no
automatic task retries, avoiding a repeated paid investigation after a failure.
Transfers retain bounded retries, including four attempts to upload final bootstrap
logs. A forced termination may prevent the final upload; Cloud Logging and earlier
periodic uploads provide partial diagnostics. Large inputs remain separate from
uploaded results. Full replay is not uploaded again; bounded target captures and
playable refits are retained.

The runtime is shared with the corrected continuation Batch launcher: pinned
Debian 12 container, Batch COS host image, Python 3.11.2 and CPU-only Torch 2.7.0.
The source archive is checksum-verified before execution. Scientific stages,
seeds, splits, objective-mass sampling and evaluation design are unchanged.

## Launch

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-bottleneck-p32-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_bottleneck.py \
  --project clever-overview-399515 --region europe-west1 \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --run-name "$RUN" \
  --source-uri gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535 \
  --expected-policy-sha256 9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b \
  --iteration 1132 --output "outputs/cloud-$RUN" --execute
```

Omit `--execute` to build a local bundle without uploading or submitting anything.
The bundle contains `job.json`, `batch.sh`, `source.tar.gz` and `launch.json`.
The launcher uses an explicit source allowlist; credentials, `.env`, outputs and
`.git` are excluded. With `--execute`, it uploads the immutable bundle and submits
using `gcloud batch jobs submit`. Existing objects cannot be overwritten. Use a
fresh run name and local output directory for each submission attempt.
Once submission succeeds, execution is independent of the local terminal.
The configured project, Batch API, service account and bucket access must already
be available, as for the continuation Batch job. No local training environment is
needed to submit: the launcher uses Python's standard library and authenticated
`gcloud`.

Headline status from any terminal (Batch jobs use generated VM names, so the old
Compute Engine name filter will not identify this experiment):

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment=hulhe-bottleneck' --format='table(name.basename():label=JOB,status.state:label=STATUS)'
```

Download small analytical reports only (set `RUN` to the actual run name in a new
terminal):

```bash
mkdir -p "outputs/$RUN-results"
gcloud storage rsync --recursive --exclude='^(?!.*\.(json|md|csv|log)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN" "outputs/$RUN-results"
```

Inspect `cloud_status.json`, `summary.json`, `fit/experiment.json`, `fit/probes.json`,
`upstream/report.json`, and `evaluation/report.json`. NPZ target captures, policy
PKLs and per-hand JSONL journals are excluded from the small-results command.
`batch-run.log` and `batch-exit.json` diagnose setup/exit failures; task output also
goes to Cloud Logging. Confirm Batch reports `SUCCEEDED` and the investigation's
`cloud_status.json` reports `COMPLETE`; an incomplete evaluation is reported in
`evaluation/report.json` even if the coordinator completed normally.
The four-stage smoke/integration tests use synthetic replay, real OpenSpiel and
an offline opponent. Implementation itself does not launch a paid job.
