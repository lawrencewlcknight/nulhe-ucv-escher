# Four-plus-four-hour limit Hold'em learning check

This is one production training seed, evaluated at P4 and P8. It checks engineering
correctness and early playing-strength improvement. It does not measure
exploitability or establish improvement across independent training seeds.

## Training and continuation

The experiment uses the existing fixed-limit game, encoder, full network widths,
replay capacities and learning budgets. It trains for four active hours, finishes
the current iteration, fits the average policy, and saves P4 plus full recovery
state. A new process restores buffers, optimizers and random states and performs
four additional active hours before exporting P8. Actual durations, iterations,
policy hashes and source fingerprints are recorded. Each average-policy fit runs
20,000 updates with the existing held-out checkpoint selection.

Active time includes collection and regret/critic/calibration fitting. It excludes
preflight, policy extraction, saving, restart, uploads and evaluations. Iteration
boundaries may overshoot four hours. The training/preflight wall budget is 12 hours;
local evaluation has an additional two hours. Failure to finish either training
stage is reported as failure, not a completed eight-hour run. Only the newest
full recovery is retained, with both playable policy exports preserved.

Install Python 3.10–3.12 and `pip install -e '.[parallel,dev]'`.

```bash
# Offline integration check: full networks, tiny replay and update budgets.
# Always use a new output directory.
python -m hulhe_escher.experiment --smoke --stage-seconds .001 \
  --training-wall-seconds 180 --evaluation-seconds 180 \
  --output outputs/learning-smoke

# Production configuration, eight active hours, on a suitable VM.
python -m hulhe_escher.experiment --output outputs/learning-4plus4
```

The smoke experiment's eight confirmatory pairs are plumbing checks only. The
combined report explicitly suppresses strength claims from smoke policies.
`train --additional-training-seconds N` can also be used directly with `--resume`;
it means additional active time in the current process, not lifetime time. It is
not part of the immutable continuation contract. Keep the iteration ceiling high
enough. Ordinary training without this flag preserves existing cadence behaviour.

## Primary: P8 against P4

Actions are sampled from frozen average policies without training exploration.
Each fully predealt nine-card deal is played twice. Cards stay assigned to their
seats, policies swap seats, and later betting cannot change the board. Private
opponent cards are never passed to policy inference. Deal and action random
streams are separate and keyed by pair ID, independent of worker scheduling.

The independent observation is P8's mean profit over the two hands, in big blinds.
One big blind/small bet = 100 chips; a later-street big bet = 200 chips.

* 10,000 pilot pairs, excluded from confirmation.
* Target 80% power for a 2 bb/100 effect, two-sided alpha 0.05.
* Required pairs = ceil((1.96 + 0.8416)^2 * pilot variance / 0.02^2).
* Freeze 100,000–1,000,000 pairs in `plan.json` before any confirmation data.
  Pilot throughput also limits the chosen count to 80% of remaining runtime.
* A Student 95% interval across pair means entirely above zero supports improvement;
  entirely below zero supports regression; otherwise the result is inconclusive.

No significance-based early stopping is permitted. Pilot and confirmation seeds
are disjoint. The plan records estimated power-related detectable effect, required
versus selected count and caps. If even the minimum will not fit, confirmation is
not started. An unfinished sample has no confirmatory claim. Zero observed variance
is treated as degenerate, not certainty. Power and interval precision are planning
approximations based on pilot variance, not guarantees.

The primary receives 80% of the two-hour evaluation budget. P4 and P8 also play
uniform-random and check/call controls (1,000 pilot pairs and 10,000 final pairs per
match), sharing the remaining 20%. Controls are descriptive, not extra primary tests.

```bash
python -m hulhe_escher.learning_evaluate --a outputs/learning-4plus4/P8.pkl \
  --b outputs/learning-4plus4/P4.pkl --workers 8 --output outputs/head-to-head
```

The original `hulhe_escher.evaluate` FO2 smoke interface remains available unchanged.

## Fell Omen 2: paired checkpoint comparison

Run this independently on the small evaluator VM. It waits for hash-verified policy
publication and begins P4's pilot while training continues. P8's pilot uses the
same deal IDs. Independent action streams are used for each checkpoint.

```bash
python -m hulhe_escher.fo2_setup
python -m hulhe_escher.benchmark --experiment outputs/learning-4plus4 \
  --output outputs/learning-4plus4/benchmarks/fo2
```

This runner explicitly uses **Fell Omen 2, free-fold-to-check adaptation**. The
pinned upstream tables sometimes assign probability to folding when a check is
free; that mass is transferred to check, preserving raise probability. This is
not a certification of the original tournament binary. Asset hashes, upstream
commit, bridge/build fingerprint and adaptation counts are recorded. Other illegal
probabilities, missing tables, protocol errors or timeouts fail the evaluation.
See [adapter fidelity](FELL_OMEN2.md). Do not mix strict and adapted results.

Each checkpoint plays both seats against FO2 on each deal. The independent unit
for improvement is the four-hand group: P8's pair-average profit minus P4's.
The run uses 10,000 pilot groups and 100,000–1,000,000 additional confirmation
groups, equivalent to 200,000–2,000,000 confirmation hands **per checkpoint**.
The frozen count is the largest requirement for:

1. 80% power for a 2 bb/100 paired improvement at alpha 0.05;
2. P4's approximate 95% score-interval half-width of 2 bb/100;
3. P8's approximate 95% score-interval half-width of 2 bb/100.

Runtime and sample caps can prevent these targets. The report shows achieved
intervals and pilot-estimated precision. There is no significance-based stopping
or automatic pooling of interrupted runs. Journals retain complete pair results,
deal IDs, both seat payoffs and adaptation counts; the manifest's seeds, policy
hashes and pinned implementation reproduce detailed hands without logging every
large action trace. Pilot journals are separate. Failed/partial samples are not
reported as completed confirmation.

Absolute scores and their 95% intervals are descriptive. The paired improvement
is a secondary test: an overall confirmatory improvement claim requires the
primary P8-vs-P4 test to pass first. This fixed testing order uses one external
opponent, so no two-opponent Bonferroni adjustment is needed. A less negative FO2
score can show improvement; beating FO2 is not required.

The benchmark has a 24-hour deadline including setup/waiting in cloud mode.
Four single-thread Python workers each own one 256-MB-heap Java opponent process.
Benchmark failures do not terminate training. No network poker service or account
credentials are required after checksum-pinned asset setup.

## VM selection and resource checks

Default: on-demand `c4-standard-16`, 16 vCPUs/60 GiB, in `us-central1-a`, 200 GiB
Hyperdisk Balanced. Dense replay allocation is approximately 15.847 decimal GB.
Use a separate `c4-standard-4`, 4 vCPUs/15 GiB, 50 GiB Hyperdisk Balanced for FO2.
Local head-to-head uses eight workers after training has exited. No GPUs are used.

Preflight, bounded to 30 minutes within the training budget, first runs a synthetic
full-replay memory probe (up to ten minutes). It touches all dense replay pages,
groups four million distinct policy rows, runs 500 policy updates and saves,
verifies and restores full recovery data. The probe requires at least 48 GiB host
RAM. It fails closed without successful monitoring or 15% host headroom. It does
not certify actor peaks or full critic-target caches; production resource reports
remain necessary. Its generated state is temporary and is never training data.

Then preflight compares (collection workers, central PyTorch threads) = (8,4),
(12,4), (8,2), (8,8), with one thread per collection actor. Two warm iteration times
are compared after an initial iteration. Replacing the default requires a 10% gain
in both warm iterations. Cases use full network widths/batches but small replay,
1,024 traversals and 500 extraction updates. Cases are individually bounded and
all data is discarded. The selected configuration is frozen across P4/P8. Collection
and central fitting remain separate phases; independent learners fit sequentially.

If memory preflight fails, inspect the report and prepare a new run with
`--training-machine c4-highmem-16` (124 GiB). There is no automatic paid resize.
`--skip-preflight` deliberately bypasses these probes for a previously verified VM.
This is a practical hardware check, not proof of the global cost/performance optimum.

## Prepare and launch

```bash
# Creates files only; no cloud calls, resources, live matches or credentials.
python tools/launch_learning_check.py --project YOUR_PROJECT \
  --bucket gs://YOUR_PRIVATE_BUCKET/hulhe --service-account YOUR_SERVICE_ACCOUNT \
  --run-name learning-4plus4-001 --output outputs/cloud-plan
```

Review `launch.json` and both startup scripts. To submit, use the same options with
`--execute` and a new local output directory. Prerequisites: enabled Compute/Storage
APIs, available C4 capacity/quota, an existing private bucket, permission to create
VMs and use the service account, and object read/write/delete access for that
account in the run prefix. The launcher creates no buckets, IAM grants or secrets.
Use a unique remote run name.

The archive includes allowlisted source, Java bridge source and asset manifests;
it excludes `.env`, `.git`, output data and the downloaded FO2 cache. The external
VM installs Java and fetches checksum-pinned assets. Recovery uploads complete
before policy publication markers; old cloud recovery is pruned only after the
replacement and pointer upload. The trainer's final sync excludes benchmark files
so it cannot overwrite the independent evaluator's report.

VM hard stops are 15 hours (trainer) and 26 hours (evaluator); startup scripts also
shut down on completion or failure and refuse to silently rerun on manual restart.
Stopped VMs retain disks: verify artifacts and delete resources to end storage
charges. Hard stops can interrupt uploads, so recovery disks are retained.

Planning compute ceilings: 15 * $0.79068 + 26 * $0.19767 = **$17.00** with the
standard trainer; **$20.78** with the $1.042712/hour high-memory fallback. Storage,
network and taxes are extra. This is not an enforced total billing cap or a live
quote. [Google Cloud pricing](https://cloud.google.com/products/compute/pricing/general-purpose?hl=en)
and [runtime limits](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime).
No production VM performance or paid cloud execution has been certified locally.

## Report and alignment with no-limit

```bash
gcloud storage rsync --recursive gs://YOUR_PRIVATE_BUCKET/hulhe/learning-4plus4-001 outputs/downloaded-run
python -m hulhe_escher.experiment_report --experiment outputs/downloaded-run
```

Read `learning_report.md` and `learning_report.json`. Training diagnostics include
finite metrics/weights, changed policy parameters, experience by player/street,
critic versions, replay occupancy, phase timings, extraction/save costs and process
memory. Fresh-regret coverage counts retained rows; production capacities exceed
the maximum rows generated per iteration. These are engineering evidence, separate
from strategic improvement. Confidence intervals describe evaluation-hand
uncertainty for this seed, not variation across independent training runs.

Alignment: both variants use four-plus-four active hours, a real process restart,
the same policy extraction, isolated evaluation, pilot-then-fixed confirmation,
duplicate-deal primary tests, descriptive controls, gated external evidence,
independent evaluator VMs, source-only cloud bundles and bounded runtimes.
Deliberate differences are the 2 bb/100 limit precision target (no-limit: 10),
locally controlled duplicate FO2 deals (remote no-limit providers lack this facility),
one external opponent, smaller replay memory and the 60-GiB default training VM.
