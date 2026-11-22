# Fresh HULHE width-scaling experiment

This is **fixed-limit Hold’em** in `hulhe-ucv-escher`. Start fresh, train for
32 active learner hours, and compare the final policy with historical HULHE P32.
Do not resize or resume the P104 checkpoint: a fresh initialization gives the
larger reservoir a valid history from iteration one.

## Fixed specification

| Parameter | Historical configuration | New configuration |
|---|---:|---:|
| Regret trunk, each player | 256 × 2 | 512 × 3 |
| Average-policy trunk | 256 × 2 | 512 × 3 |
| Regret/policy feature branches | 96 | 128 |
| Policy replay reservoir | 4 million rows | 12 million rows |
| Critic trunk / branches | 192 × 2 / 96 | unchanged |
| Calibration trunk / branches | 128 × 2 / 64 | unchanged |
| Policy validation update candidates | 2500/5000/10000/20000 | add 500/1000 |
| Average-policy sampling | legacy uniform at P32 | P104 `objective_mass` correction |

Each new regret/policy network has **777,731** parameters, versus 180,867.
Total online network parameters rise from 914,321 to 2,704,913; target/optimizer
copies add memory. The policy has 5% stable canonical-group validation, a
20,000-update maximum, and a fresh initialization at every policy extraction.
The selected validation checkpoint supplies both model and optimizer state.

Retain 10,000 traversals/player/iteration; regret 750 updates; critics 250–2500
updates/member and calibration 100–2000, scaled with replay occupancy; batch
size 2048; policy learning rate .003 and other learning rates .001. Retain two
critics, four-target averaging, alpha 2.3, gamma 2, beta 1, sampling floor .2,
and no instantaneous predictor. Encoder and game rules are unchanged.

The experiment config explicitly persists `policy_group_sampling=objective_mass`.
The sampler draws groups in proportion to objective mass, with constant
importance factors (no double weighting), seeded by `[training_seed, iteration]`
as in the P104 archived correction. Every export must record the required mode
and seed; otherwise the experiment fails. Legacy configuration defaults remain
unchanged for old workflows. The new independent `critic_branch_width` setting
prevents widening the critic accidentally when policy/regret branches widen.

## Time, checkpoints and comparisons

- Seed 20261115; fresh models, replay, optimizers and RNG state.
- Export P8, then P14/P20/P26/P32 at six-hour intervals. These labels belong to
  this new run prefix, not the older P8/P32 artifacts.
- Targets are cumulative active times, rounded up at an iteration boundary.
  Collection and learner updates count; policy fitting, persistence, reloads
  and uploads do not. This is **32 active hours, not 32 elapsed hours**.
- Each stage exits and restarts from its complete recovery; every export also
  passes an independent-process full reload and exact playable-policy check.
- Local timed recovery checks every six wall hours, suppressed by a recovery
  saved in the preceding hour. Verified stage recoveries are uploaded before
  cloud pointers and policy markers. Previous cloud recovery is then pruned.
  The latest stage recovery, all playable policies and exact training source
  remain in storage. A failure between stages can lose work since the latest
  uploaded stage boundary; intermediate timed saves are local to the Batch VM.
- Primary comparison: **new P32 versus historical P32**, alpha .05. Historical
  policy SHA-256: `9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b`.
- Secondary comparisons: new P32/P8 and the four adjacent checkpoint pairs,
  alpha .01 each (Bonferroni family .05).
- Duplicate deals with swapped seats; independent 10,000-pair pilot excluded
  from results; 100,000–1,000,000 confirmatory pairs, targeting 80% power for
  2 bb/100. Half the 12-hour evaluation budget goes to the primary comparison;
  the other half is split equally across secondary comparisons. Pilot variance
  and throughput determine the fixed confirmatory sample. Sample/time caps can
  prevent the requested power; reports retain required/selected sizes and
  incomplete status. Completion does not guarantee a statistically significant
  result. No Slumbot, GTO Wizard or other external opponent is run here.

Historical P32 used the old sampler. Consequently this tests the **combined
revised recipe**, not network size in isolation. A future fresh smaller-network
control using the corrected sampler and the same validation schedule would
isolate scaling more cleanly. Report iterations, nodes, trajectories, fit time
and throughput alongside playing strength: equal hours are not equal samples
or FLOPs. One training seed also limits conclusions about training variability.

## VM choice and memory gate

Default: **c4-standard-16**, 16 vCPUs / 60 GB advertised RAM, eight collection
workers and eight Torch threads. This retains the earlier CPU/parallelism
configuration. Dense learner replay alone is approximately **29.415 GB
(27.4 GiB)**; grouping, actors, caches, optimizer states and reloads add to it.
The network weights themselves are a small part of memory use.

The smallest supported candidate is **c4-highmem-8**, eight vCPUs / 62 GB,
with four workers and four Torch threads. It is a lower-cost-per-hour option,
but likely collects fewer samples in 32 hours, weakening an equal-time comparison
with the earlier 16-vCPU run. A 31/32 GB VM leaves insufficient headroom. These
are planning judgments, not measured full-scale fit or throughput claims.
Machine specifications: https://docs.cloud.google.com/compute/docs/general-purpose-machines

Both options run a disposable, full-capacity memory preflight before production:
real collection/fits with the selected workers, touch all learner replay pages,
group/fit full policy replay, save and reload all state. Measured host memory
must retain at least max(8 GiB, 12%) headroom. The subprocess is destroyed before
the seeded experiment starts, so synthetic data cannot enter training. The probe
does not prove every future peak; sampling every .5 seconds may miss transients.
If it fails, **stop before training**, inspect the memory report, and explicitly
choose a new job on `c4-highmem-16` (124 GB). Never silently upgrade or restart.

The Batch task has a 200 GB Hyperdisk Balanced boot disk, standard provisioning,
zero retries, a two-hour preflight limit, 40-hour training wall budget, 12-hour
evaluation budget and a 56-hour hard stop including setup/upload allowance.
Batch manages and releases the VM/disk. No chat schedule or persistent VM is
created. All checkpoint transfers are bounded and verified by the existing
storage transport; a mandatory final analytical sync reports transfer failures.

## Prepare and submit

Run from `/Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher`:

```bash
python3 tools/launch_scaling_batch.py \
  --run-name hulhe-wide32-batch-20261122 \
  --machine c4-standard-16 \
  --output outputs/cloud-hulhe-wide32-batch-20261122
```

The selected run uses `c4-standard-16`, eight collection workers and eight Torch
threads. This only prepares an immutable source archive, readable configuration, launch
plan, Batch JSON and bootstrap script. Review them, then submit explicitly:

```bash
python3 tools/launch_scaling_batch.py \
  --output outputs/cloud-hulhe-wide32-batch-20261122 \
  --submit-prepared
```

For the smaller VM, prepare a **new** bundle/name with `--machine c4-highmem-8`.
Do not edit a submitted bundle, reuse its prefix or resubmit an expensive run.
The cloud prefix claim is immutable: partial submission failures require
inspection before further action.

## Download analysis only

```bash
mkdir -p outputs/hulhe-wide32-batch-20261122-results
gcloud storage rsync --recursive \
  --exclude '(^|/)(state_[0-9]+|state)(/|$)|^(?!.*\.json$).*' \
  gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-wide32-batch-20261122 \
  outputs/hulhe-wide32-batch-20261122-results
```

This fetches JSON plans, progress/efficiency metrics, memory/reload receipts and
evaluation statistics. It excludes policy/model files, replay states, source
archives and raw hand logs. Results are in `scaling.json`, `evaluation.json`,
`evaluation/*/report.json`, `evaluation/*/plan.json` and the P*.json markers.

## Local verification

`python3 -m hulhe_escher.scaling_experiment --output /tmp/hulhe-wide32-smoke --smoke`
exercises the full network architecture with tiny replay/update budgets, all
five export/restart boundaries and full-state reload checks. It is plumbing
validation, not evidence of learning or full-capacity memory fit. Production
cloud publication is disabled in smoke mode.

The scaling tests also compare a resumed run with an uninterrupted run,
including model/optimizer/replay/RNG equality; check independent critic capacity,
sampler selection and seeds; check cumulative time targets and failure guards;
and validate Batch limits and analytical-only publication.
