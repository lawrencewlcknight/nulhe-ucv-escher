# Limit Hold'em — HULHE UCV-ESCHER

A research repository for Unbiased Control-Variate ESCHER in **heads-up
fixed-limit Texas Hold'em (HULHE)**, with **Fell Omen 2** as the selected challenge
opponent. The learner comes from FHP Experiment 36/38 through the HUNL baseline,
with **up to 2,500 updates per critic per fit**. See [the baseline evidence](docs/BASELINE.md).

The game interface uses **OpenSpiel 1.6.3 `universal_poker`**:

* Two players, 52 cards, two hole cards each and all four betting streets.
* Blinds 50/100; fixed bets 100/100/200/200 on preflop/flop/turn/river.
* Four bets per street. The preflop big blind counts as the first bet, so the
  OpenSpiel `maxRaises` setting is `3 4 4 4`.
* Actions: `0=fold`, `1=check/call`, `2=bet/raise`, subject to the legal mask.
* Independent, rake-free hands with nonbinding bankrolls. Returns are chips;
  divide by 100 for small bets (big blinds), or 200 for big bets.

The strategic encoder supplies **417 policy inputs** and **605 critic inputs**.
It preserves exact cards and betting history, and adds hand strength, draw
potential, board texture, pot odds and betting interactions. Three network
branches process exact cards, strategic card descriptors and betting context.
See [the representation review](docs/STRATEGIC_FEATURES.md) for exact versus sampled
statistics, costs and design choices. The production training specification uses
750 regret updates per player, replay-scaled 250–2,500 critic updates and
100–2,000 calibration updates, frozen regret-target caching and gradient norm
clipping at 10. Regret replay is 250,000 rows per player;
historical policy replay is four million rows.
Regret and average-policy networks use 96-wide input branches and a 256/256
hidden trunk; critics use 96 and 192/192; calibration retains 64 and 128/128.
The six online networks total 914,321 trainable parameters. See the
[capacity specification](docs/BASELINE.md#current-hulhe-network-capacity).
See [the game-interface specification](docs/GAME_INTERFACE.md) for evidence,
seat conventions and the complete feature layout.

Training reuses encoded features for each unchanged native OpenSpiel state,
separately for each encoder and observing player. Weak references release completed
states; a full-history token invalidates entries after mutation. Returned arrays
remain independently mutable, and serialization omits the temporary cache.
Binary suit ordering uses packed bits, inference uses explicit tensor devices,
and frozen targets use contiguous replay views while preserving tail padding.
Both Hold'em variants replace repeated rank sorting with an exact, read-only
8,192-mask lookup (832 KiB per process). Features, model dimensions and training
configuration are unchanged. `tests/test_training_efficiency.py` checks all rank
masks, cache behavior, and exact replay/RNG/model/Adam equivalence over repeated
fits, including production minibatches. Existing source-fingerprint checks for
training continuation still apply. Worker pipelining and persistent replay-hash
caches remain deferred until VM profiling supports them.

**Status:** fixed-limit self-play, checkpoint/resume and local Fell Omen 2
matches are implemented. See [the opponent interface](docs/FELL_OMEN2.md) for
setup, lightweight tests and the known upstream free-fold incompatibility.
No playing-strength result is claimed. See [the development plan](docs/HULHE_PLAN.md).

## Install

Use Python 3.10–3.12 (3.12 is the development reference):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,parallel]'
```

Core dependencies are pinned in `pyproject.toml`. The parallel extra adds Ray;
sequential training does not need it. The source repository and copied-file
hashes are recorded in [provenance.json](docs/provenance.json). Upstream attribution
and the original licence-status note remain in [UPSTREAM.md](vr_deep_cfr/UPSTREAM.md).
The installed command is `hulhe-train`; `python -m hulhe_escher.train` is equivalent.
Use a separate virtual environment from the HUNL project because the inherited
shared package names (`unbiased_escher`, `adaptive_escher`, `vr_deep_cfr`) overlap.

## Continue training from a checkpoint

To test policy sampling without another self-play run, see the
[frozen-replay sampler experiment](docs/POLICY_SAMPLER_EXPERIMENT.md). It compares
matched refits of saved policy replay and evaluates them against each other and
Fell Omen 2; the production default is unchanged.

Full recovery checkpoints support continued training. Re-run the original
training command with the same output directory and options, add `--resume`, and
set `--iterations` to the desired **total** iteration count. For example, after
training the eight-worker baseline to iteration 100:

```bash
python -m hulhe_escher.train --iterations 200 --workers 8 --threads 8 \
  --policy-seconds 21600 --recovery-seconds 21600 \
  --seed 0 --output outputs/baseline-seed0 --resume
```

The runner automatically loads the recovery directory referenced by `latest.json`
and continues at the next iteration. It restores all online/target networks,
Adam states, persistent replay, critic target history, counters, random-number
states (including workers), and recovery/policy scheduling. Fresh regret batches
and temporary caches are rebuilt as they are during uninterrupted training.

Keep the whole run directory: `manifest.json`, `latest.json`, and the latest
`state_*` directory with all its shards. Shards are verified before loading.
Use the same source, configuration, Python/dependency versions, seed, worker
count, thread count and scheduling options. Only the total iteration limit may
change. If the target has already been reached, resume performs no further work.

`policy_*.pkl` files are inference exports; they do not contain the replay and
optimizer state required for full training continuation. Every policy export
also produces a full recovery snapshot, subject to latest-only recovery retention.

## Verify and run

```bash
python -m pytest -q
python -m hulhe_escher.train --smoke --iterations 2 --output outputs/smoke
python -m hulhe_escher.train --smoke --iterations 3 --output outputs/smoke --resume
RUN_RAY_TESTS=1 python -m pytest -q -m ray
```

Smoke reduces replay, minibatch and optimiser budgets; it is not evidence of
playing strength. It uses the full baseline network sizes. Tests exercise real OpenSpiel and PyTorch. Ray tests start
real local worker processes and are opt-in.

The production learning budget is in
[`hulhe_escher/data/baseline.json`](hulhe_escher/data/baseline.json).
An explicit training invocation using the original eight-worker arrangement is:

```bash
python -m hulhe_escher.train --iterations 100 --workers 8 --threads 8 \
  --recovery-seconds 21600 --policy-seconds 21600 --seed 0 --output outputs/baseline-seed0
```

This is a substantial workload: 10,000 trajectories **per player per iteration**,
large replay buffers, and up to 20,000 policy updates per fit. Benchmark memory
and throughput on the intended training machine before choosing a long run.
The command uses the current machine and does not submit any cloud jobs.

Automatic recovery checks run every **21,600 wall-clock seconds (six hours)**,
on a fixed grid from run start. A check skips saving if a successful recovery
checkpoint was completed less than **3,600 seconds (one hour)** ago. This includes
recovery snapshots saved alongside policy fits. At exactly one hour, saving is
eligible again. A skipped check advances to the next six-hour slot; it does not
retry when the one-hour window expires. For example, a save at 5.5 hours suppresses
the six-hour check, with the next check at 12 hours.

Checks run after completed iterations, so long iterations can delay them.
Downtime counts as wall time; missed slots cause at most one check on resumption.
The grid, last successful save time and skipped checks survive restart. A skipped
check updates only the small latest-pointer metadata, without rewriting replay.
Policy fitting remains every **21,600 active learner seconds**, excluding policy
fitting, persistence and downtime. Policy-triggered, explicit iteration and final
saves remain unconditional.

`--checkpoint-every N` adds recovery-only saves every N iterations (default 0).
`--policy-every N` adds policy fits every N iterations (production default 0;
smoke default 1). Set `--policy-every 0` for time-only policy scheduling in smoke.
`--recovery-seconds 0` disables timed recovery; `--checkpoint-seconds` is an alias
for `--recovery-seconds`. Policy fits also save full recovery state, including the
selected policy and matching optimizer. The final requested iteration always
fits a policy and saves recovery state. Both schedule clocks survive resume;
recovery-only saves never fit a policy. No background timer interrupts learning.

Both variants cache frozen regret targets once per fit, clip regret gradient
norms at 10, and skip empty regret batches without modifying learner state.
Collection and critic bootstrapping use uniform probabilities among tied maximum
regrets when all legal regrets are non-positive; unequal negative regrets still
select the highest value. Tiny positive regrets are normalized without a floor
that would reduce total probability. Terminal masks produce zero continuation.

The historical policy reservoir holds four million rows. Policy fitting resets
the model, runs all 20,000 updates, and selects the best iteration-weighted
held-out cross-entropy at 2,500/5,000/10,000/20,000 updates. A stable, game-neutral
hash holds out 5% of canonical feature groups; identical features cannot appear
in both partitions. The selected model and matching Adam state are restored.
Tiny replays without both partitions use all groups and report validation
unavailable. This selection is within one fit, not extra self-play experiments;
held-out policy error is not exploitability.

See the [shared training specification](docs/TRAINING_ALIGNMENT.md).

Checksummed, sharded continuation states contain networks, optimisers, replay,
temporal critic history and RNGs, including actor RNGs. Recovery saves do not
require a policy fit. Only complete outer iterations are saved. `--resume` checks
the source fingerprint, runtime, seed, configuration and worker arrangement;
`--iterations` is the desired **total**, not additional iterations. Use trusted
checkpoint files only; their PyTorch/pickle metadata is not an untrusted format.
Only the newest recovery checkpoint is retained. After a new save passes checksum
verification and `latest.json` is updated, all older recovery checkpoints in that
run directory are deleted. Failed saves preserve the previous recovery checkpoint.
Playable `policy_*.pkl` files and metrics are retained separately. Allow room for
the old and new recovery checkpoints during saving. Cloud orchestration remains
future work.

## Layout

| Directory | Purpose |
|---|---|
| `hulhe_escher/` | Game, four-street encoder, configuration, CLI and checkpoints |
| `unbiased_escher/` | UCV estimator, structured learners, compact replay, parallel collection |
| `adaptive_escher/` | Persistent frozen-target critic and target cache |
| `vr_deep_cfr/` | Inherited neural cumulative-regret and replay foundations |
| `tests/` | Fixed-limit rules, information boundaries, cache equivalence, resume and actors |
| `docs/` | Baseline evidence, provenance and next implementation steps |

The HUNL source history is retained in this independent repository.
Ignored outputs, checkpoints, environments and build caches
were not copied. The source HUNL repository is untouched.

The default encoder is `hulhe_strategic_limit_v5`. Policy checkpoint format 2
supports v5 and the original fixed-limit v2/v3/v4 encoders for evaluation.
Full continuation checks the exact game and encoder: **start fresh training for
v5**. Old v2/v3/v4 replay cannot be reused. HUNL/preparation-copy policies remain rejected.
The production training-specification update also requires a fresh run: old
continuations have a different source/configuration contract. Saved playable
policies still load with their original encoder and network dimensions.

## Play a checkpoint against Fell Omen 2

With a JDK installed, prepare the pinned opponent once and run a small match:

```bash
python -m hulhe_escher.fo2_setup
RUN_FO2_TESTS=1 python -m pytest -q -m fo2
python -m hulhe_escher.evaluate --checkpoint outputs/smoke/policy_000003.pkl \
  --pairs 20 --seed 20261114 --output outputs/fo2-evaluation
```

This runs 40 hands with paired deals and exchanged seats. The default rejects
incompatible opponent actions and logs failures; it does not silently alter
the strategy. See [FELL_OMEN2.md](docs/FELL_OMEN2.md) before longer benchmarks.

## Four-plus-four-hour learning experiment

See [the learning experiment protocol](docs/LEARNING_EXPERIMENT.md) for P4/P8
training and restart, statistically planned head-to-head and Fell Omen 2
comparisons, memory/throughput preflight and the reviewable Google Cloud launcher.
The launcher prepares files by default; only `--execute` creates paid VMs.

### Continue P8 for another 24 active training hours

`tools/launch_continuation.py` prepares a reviewable launch bundle. `--execute`
submits one `c4-standard-16` VM. Training uses eight collection workers and eight
Torch threads inherited from the original P8 contract. The VM performs training
and evaluation sequentially, so no evaluator VM waits idle during training.

The continuation restores the **full** P8 recovery, not just `P8.pkl`. It runs
four stages ending at approximately P14, P20, P26 and P32. Targets are measured
from P8's actual active time (about 8.04 hours), at completed iteration boundaries;
policy fitting, persistence, uploads and evaluations do not consume the 24-hour
active training budget. A playable policy and full recovery are uploaded at each
stage. All five playable policies are kept; only the newest continuation recovery
is retained. The original P8 run and recovery remain untouched.

The archived original trainer executes in a separate directory with the original
source fingerprint, runtime, configuration, replay, optimizers and actor RNGs.
The coordinator does **not** bypass the strict resume contract or modify the
learning algorithm. Initial and final full recoveries are loaded into a fresh
solver process and checked against their playable exports. `training_source.tar.gz`
in the new cloud run contains the original trainer needed to resume P32 later;
`source.tar.gz` contains the new coordinator/evaluator. See `resume.json` for the
recovery locations. Downloading only analytical results is insufficient to resume.

After the final reload check succeeds, evaluation runs:

* Primary: P32 vs P8, duplicate deals with seats swapped, 95% interval.
* Secondary: P14/P8, P20/P14, P26/P20 and P32/P26 head-to-head, plus each new
  policy's FO2 performance minus P8's performance. These eight tests use
  Bonferroni-adjusted 99.375% intervals, with formal improvement claims gated on
  the primary test passing.
* FO2: all five policies play the same deals in both seats against the pinned
  **free-fold-to-check adaptation**, with independent action streams. Absolute
  scores have descriptive 95% intervals. A group contains ten hands.

Each evaluation uses a disjoint pilot (10,000 groups/pairs), freezes sample size
before confirmation, targets a 2 bb/100 effect at 80% estimated power, and bounds
confirmation between 100,000 and 1,000,000 groups/pairs. FO2 sample planning uses
the largest requirement across the four paired improvements and descriptive
precision targets. Incomplete runs cannot produce confirmatory claims. Sample
caps and estimated detectable effects are recorded; one training seed does not
establish exploitability or generality across seeds.

The trainer has a 30-hour wall budget; evaluation has a separate 12-hour budget
(20% reserved for the five head-to-head tests). The VM stops on completion/failure
and has a 44-hour hard limit including setup and publication. Stopped disks remain
until explicitly deleted. Progress is published every two minutes; full hand
journals are uploaded at the end. Results are in `continuation.json`,
`evaluation/report.json` and `evaluation/continuation_report.md`.

Example (omit `--execute` to prepare only; use a fresh run name/output directory):

```bash
python3 tools/launch_continuation.py \
  --project clever-overview-399515 --zone europe-west1-b \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --source-uri gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-4plus4-20261115-104829-repair1 \
  --expected-p8-sha256 8713598dd4e670aa937255eb6700e42a29c60df660ec6fd4a44e740fa25d72ca \
  --run-name hulhe-plus24-UNIQUE-TIMESTAMP \
  --output outputs/cloud-hulhe-plus24-UNIQUE-TIMESTAMP \
  --execute
```

For a local smoke integration test without cloud services, run
`python -m pytest tests/test_continuation.py`. It trains a small full-architecture
checkpoint, restores four successive stages, verifies the final recovery and runs
all comparisons against an offline test opponent. Such results are labelled smoke
and cannot support strength claims.

To recover a continuation stopped while publishing a completed stage, the
coordinator accepts `--resume-checkpoint P14 --expected-resume-sha256 HASH` (also
P20/P26/P32). Use its retained workspace and original baseline/source arguments.
It requires a failed run whose latest full state exactly matches that completed
policy, verifies a fresh full-state restore, repairs cloud publication, and then
executes only the remaining stages on the original cumulative time grid. It
rejects an intermediate recovery that has no matching completed policy. Previous
failure reports and stage logs are retained under numbered retry names. Deploy
the updated coordinator/uploader separately from the immutable archived trainer;
do not change the archived training source or bypass its resume contract.

### Continue P32 to P56 (and subsequent 24-hour blocks)

For the **P104→P128 Google Batch experiment**, use the
[P104 continuation guide](docs/P104_CONTINUATION.md). It restores the verified
P104 full recovery, retains corrected policy sampling, and exports P110, P116,
P122 and P128. The primary P128/P104 comparison uses at least 400,000 duplicate
pairs, with the existing adjacent comparisons and Fell Omen 2 benchmark.

For the **P80→P104 Google Batch experiment**, use the
[P80 continuation guide](docs/P80_CONTINUATION.md). It restores the verified P80
full recovery, retains the corrected policy sampler, and exports P86, P92, P98
and P104 with adjacent comparisons and the Fell Omen 2 benchmark.

For the prepared **P56→P80 Google Batch experiment**, including verification of
the inherited objective-mass policy sampler, use the
[P56 continuation guide](docs/P56_CONTINUATION.md). Its launcher pins the completed
P56 checkpoint and runs the six-hour comparisons and Fell Omen 2 benchmark.
The commands below describe the older standalone-VM workflow.

Pass `--baseline-hours 32` to start from P32 and export P38, P44, P50 and
P56. The primary comparison becomes P56/P32; the four adjacent comparisons and
four FO2 improvements all use the shifted labels and P32 baseline. The statistical
design, VM size, worker counts, training settings and active-time budget stay the
same. Evaluation uses a fresh deterministic seed block for each continuation.
P32 is evaluated again on those fresh deals alongside the four new policies.

For baselines after P8, the downloader uses `training_source.tar.gz` from the
previous run, **not** its coordinator `source.tar.gz`. The original trainer and
full recovery contract remain unchanged through every generation. The old
experiment stays intact in its own cloud prefix. The baseline SHA-256 must match
the named policy, which must also match the latest full recovery pointer. Before
training, the complete state is checksum-verified and restored in a fresh solver;
the same check runs on the final P56 recovery.

```bash
python3 tools/launch_continuation.py \
  --project clever-overview-399515 --zone europe-west1-b \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --source-uri gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535 \
  --baseline-hours 32 \
  --expected-baseline-sha256 9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b \
  --run-name hulhe-p32-plus24-UNIQUE-TIMESTAMP \
  --output outputs/cloud-hulhe-p32-plus24-UNIQUE-TIMESTAMP \
  --execute
```

Omit `--execute` to prepare a launch without creating a VM. The old
`--expected-p8-sha256` option remains an alias for compatibility. Future blocks use
`--baseline-hours 56`, then `80`, etc., with the corresponding source run and hash.
When retrying a failed stage, retain `--baseline-hours 32` and use the actual
completed checkpoint, for example `--resume-checkpoint P38`.

For compact analytical downloads, copy just `continuation.json`,
`evaluation/continuation_report.md`, `evaluation/report.json` and
`evaluation/design.json`. Bulk recursive downloads also include large hand
journals and training logs.

For the isolated P32 policy-capacity and upstream-target investigation, see
[the bottleneck investigation](docs/BOTTLENECK_INVESTIGATION.md). It compares six
frozen-replay refits, instruments one restored iteration and evaluates one
validation-selected policy on fresh paired deals. Its Google Batch launcher uses
one c4-standard-16 task, a four-hour emergency limit and automatic VM/disk cleanup.

For lightweight strategic-position evaluation, see the
[Limit Hold’em strategic audit](docs/STRATEGIC_AUDIT.md). It compares playable
policies on a frozen development/assessment library, computes exact river
response gaps and provable dominated folds, and checks hidden-card privacy and
suit invariance. Preflop/flop/turn action diagnostics are reported separately.
Run `python -m hulhe_escher.strategic_audit run --baseline uniform --smoke
--output outputs/strategic-smoke` for an offline smoke check.

For a Google Batch investigation of common decisions, replay coverage,
generalization and local regret estimates, see
[the P32 position investigation](docs/POSITION_INVESTIGATION.md). It freezes a
visitation-based benchmark from P32 self-play and Fell Omen 2 encounters, audits
history-conditioned river values, and instruments one isolated training iteration.

The [P80 position audit](docs/P80_POSITION_INVESTIGATION.md) repeats the common-position
diagnostic at P80 and compares P80 with P56 on the frozen P56 development library,
holding the P56 opponent and ranges fixed. It runs as a separate Google Batch job.

The [two P104 diagnostic experiments](docs/P104_DIAGNOSTICS.md) independently test
all-replay policy extraction (including earlier stopping checkpoints and fresh
paired matches) and investigate regret-target noise with conditional fitting.
Both use frozen P56 river panels and preserve the P104 source and continuation.

The [sequential specialist frozen-target checks](docs/SEQUENTIAL_FROZEN_CHECKS.md)
test the proposed 1.50M-parameter network on saved P104 policy replay and captured
regret targets. Two fitting seeds compare unrestricted training, selective street
updates and two protection strengths, with held-out errors and interference checks.
This bounded Batch diagnostic performs no new self-play or production changes.

The [protected sequential four-plus-four learning experiment](docs/SEQUENTIAL_LEARNING_EXPERIMENT.md)
trains this architecture from scratch for eight active hours, with P4/P8 policies,
a real process restart, verified full recovery, P8-versus-P4 and control matches,
and independent Fell Omen 2 evaluation. Protected selective fitting applies to
both regret and average-policy networks. Preparation is offline; cloud submission
is explicit, using separate trainer and evaluator Google Batch jobs.

The [sequential P8 plus 24-hour continuation](docs/SEQUENTIAL_P8_CONTINUATION.md)
restores that run's full P8 with its unchanged archived trainer and exports
P14/P20/P26/P32. A single Batch job trains first, then evaluates P32 against P8,
each adjacent checkpoint pair, and all five policies against Fell Omen 2.


The [Wide/Deep-P32 versus corrected smaller P32 experiment](docs/SCALING_CORRECTED_COMPARISON.md)
evaluates both final scaled policies against both saved corrected P32 refits,
using shared duplicate deals and two multiplicity-adjusted primary tests.

The [P104 regret-supervision audit](docs/P104_SUPERVISION.md) uses repeated
independent samples and exact river references to diagnose target noise,
sampling ranges and equal-compute estimator improvements without retraining.

The [P104 twenty-network ensemble experiment](docs/P104_ENSEMBLE.md) fits 20
independent all-replay average policies, averages their action probabilities, and
compares the ensemble with saved P104 and Fell Omen 2 on fresh paired deals.
Its excluded pilot measures throughput and fixes the confirmation sample size.

For a small follow-up using the already captured regret targets, see the
[captured regret-target diagnostic](docs/REGRET_TARGET_DIAGNOSTIC.md). It separates
within-state target variation from sample-mean prediction error, conditionally
compares additional fitting with a function-preserving capacity expansion, and
checks common river decisions without new self-play or full replay downloads.

The [P56 versus corrected P32 comparison](docs/P32_CORRECTED_VS_P56.md) is an
evaluation-only Google Batch experiment using both saved corrected P32 refits,
fresh duplicate deals, and a predeclared equal-weight primary comparison.

## Fresh architecture comparisons

The [wide32 experiment](docs/WIDE32_SCALING.md) trains 512×3 regret/policy trunks.
The [deep32 experiment](docs/DEEP32_SCALING.md) substitutes approximately
parameter-matched 256×10 pre-activation residual trunks, preserving the rest of
the fresh 32-hour recipe and evaluating directly against matching wide policies.

The [hybrid32 experiment](docs/HYBRID32_SCALING.md) combines a 512-wide residual
stage with four 256-wide residual blocks and a direct wide-to-output connection.
It starts fresh for 32 active hours, exports P8/P14/P20/P26/P32, and compares its
final policy with Deep-P32, both corrected smaller-P32 refits and Fell Omen 2.
All three run as bounded Google Batch jobs; preparation does not submit compute.

The [Wide and Deep P32 +24-hour continuations](docs/SCALED_P32_CONTINUATIONS.md)
restore each architecture's original full checkpoint and archived trainer, export
P38/P44/P50/P56, and evaluate progress against its own P32 and Fell Omen 2.

The [Hybrid P32 +48-hour continuation](docs/HYBRID_P32_CONTINUATION.md) restores
Hybrid-P32 with its original archived trainer, exports P38 through P80 every six
active hours, and evaluates progress against its own P32 and Fell Omen 2.
