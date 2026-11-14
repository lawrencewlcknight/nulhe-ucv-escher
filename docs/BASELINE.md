# Inherited UCV-ESCHER baseline

This document describes the current HULHE specification and its inherited evidence.
The project uses the [fixed-limit interface](GAME_INTERFACE.md). The HUNL-specific
section below is historical and does not describe the current game.
Package references below use the new namespace.

## Current HULHE network capacity

The November 2026 capacity review selects the following sizes for fresh runs.
Each network has separate exact-card, strategic-card and betting-context input
branches, followed by the shared hidden trunk and output layer.

| Network | Width of each input branch | Shared hidden layers | Parameters per online network |
|---|---:|---|---:|
| Regret (two players) | 96 | 256, 256 | 180,867 |
| Average policy | 96 | 256, 256 | 180,867 |
| Critics (two cross-fit members) | 96 | 192, 192 | 151,491 |
| Calibration | 64 | 128, 128 | 68,738 |

This is three hidden layers on each input-to-output path. Total online trainable
capacity is **914,321 parameters**, up from 911,377 with encoder v4; frozen targets, temporal
snapshots and optimiser state are additional. V5 adds four policy inputs and eight critic inputs.
Smoke uses the same architecture with smaller sampling and fit budgets.

`num_hiddens` and `num_layers` now control the regret trunk independently of
`critic_network_layers`, `calibration_network_layers` and the existing
`average_policy_network_layers`. `structured_branch_width` controls regret,
policy and critic branches; `calibration_branch_width` keeps calibration at 64.
All these dimensions are validated before solver construction.

The rationale is a moderate capacity increase for four-street Hold'em while
keeping the frequently evaluated calibration model small. This is a practical
starting point, not a demonstrated optimum. Relevant comparisons are ESCHER's
[released 256/128 defaults](https://github.com/Sandholm-Lab/ESCHER/blob/main/ESCHER.py),
[VR-Deep's 3x128 flop-Hold'em experiment](https://arxiv.org/html/2511.08174v1),
and [Deep CFR's deeper structured architecture](https://proceedings.mlr.press/v97/brown19b/brown19b-supp.pdf).
Flop-only results do not validate full-Hold'em capacity.

Start a fresh output directory after this change. Full continuation requires
matching source and configuration and cannot resize trained weights or Adam
state. Existing playable policy checkpoints retain their saved dimensions and
remain loadable for evaluation without conversion.

## Decision

Use the **FHP Experiment 36 learner, continued unchanged in Experiment 38**:
the structured, cached, synchronous parallel UCV implementation with **2,500
TD updates for each of two critic members per outer iteration**. This is 5,000
critic optimiser steps in total, compared with 20,000 at the old 10,000-per-member
setting. It is the best-supported reduced-budget starting point, not a proven
optimum for no-limit Hold'em.

The extraction uses FHP source commit
`c2d7a5cf5c1bfeb98393fb185457b68c7854ce13`. The historical source has unrelated
working-tree edits in an evaluation launcher; none were needed or copied.
The inherited implementation traces through Experiment 9's frozen-target cache,
Experiment 7's parallel collector, and Experiment 2's structured networks and
compact float32 replay. Reusable checkpoint code was lifted out of experiment
packages into `hulhe_escher`; runtime imports do not depend on historical experiments.

## Why 2,500 rather than 500

The completed **Leduc Experiment 51**, run `exp51-budget-20261101-220804`, compared
10,000, 2,500, 500 and zero critic updates. There were three paired seeds, 600
iterations and 12 million trajectories per run. The primary endpoint was mean
exact-average exploitability at iterations 450, 500, 550 and 600, in native
`NashConv / 2` units. Positive differences below mean worse than 10,000 updates.

| Updates per critic | Mean difference | 98.333% paired interval | Mean critic fit time | Mean learning time |
|---:|---:|---:|---:|---:|
| 10,000 | Reference | — | 17.447 h | 46.822 h |
| **2,500** | **+0.001656** | **[-0.000961, +0.004272]** | **4.906 h** | **37.218 h** |
| 500 | +0.000764 | [-0.012794, +0.014323] | 1.266 h | 36.951 h |
| 0 | +0.051951 | [+0.019309, +0.084592] | Approximately 0 | 29.948 h |

The 2,500-update arm met the study's exploratory acceptable-loss margin of 0.005;
500 did not because its uncertainty was much wider. Zero-Q was consistently
worse. The three-seed study does not establish equivalence or transfer to HUNL.
Runtime differences in other components prevent interpreting these totals as
isolated critic speedups; the 500-update arm barely improved mean total time
over 2,500 in this run. Its smaller critic cost alone is insufficient reason to
choose it as our baseline.

The compact result snapshot is in [critic_budget_evidence.json](critic_budget_evidence.json).
The original timing source is `analysis/costs_by_seed.csv` in the same run.
Experiment 50 studied temporal averaging; Experiment 52 studied advantage-label
noise. Neither is the critic-update-budget study. Four-fit averaging remains in
the inherited baseline for continuity; we do not claim that Experiment 50 proved
it beneficial.

## Learning specification

| Component | Selected setting |
|---|---|
| Algorithm | Outcome-sampling UCV-ESCHER with corrected all-action estimates and DCFR+ cumulative-regret fitting |
| Control coefficient | Fixed beta = 1 |
| Critics | Two cross-fit members; trajectory assignment determines held-out collection control |
| Critic fitting | Persistent online networks/Adam; 2,500 updates per member per iteration; batch 2,048; LR 0.001; gradient norm cap 10 |
| Target critics | Frozen during collection/fit; parameter average over the last four completed fits (available history during warmup) |
| TD target cache | Rebuilt once per eligible fit; no cache reused across target/replay/iteration changes |
| Sampling | Learned residual-standard-deviation proposal, mixed with 20% uniform legal-action probability mass |
| Calibration | Enabled; 2,000 updates per iteration; batch 2,048; LR 0.001; minimum variance 1e-5 |
| Regret fitting | One network per player; 1,000 updates per player per iteration; batch 2,048; LR 0.001; alpha 2.3; persistent cumulative networks; frozen targets cached once per fit |
| Non-positive regret fallback | Uniform among exactly tied maximum legal regrets; unequal negative regrets retain argmax; positive regret matching unchanged |
| Instantaneous predictor | Disabled; prediction gate fixed at zero |
| Average policy | Grouped soft-target cross entropy; iteration weighting gamma 2; reset each fit; 20,000 updates with best validation-checkpoint selection; batch 2,048; LR 0.003 |
| Policy fit cadence | Every six active learner hours and final endpoint; independent of recovery saves |
| Recovery cadence | Every active learner hour and final endpoint, at completed outer iterations |
| Collection | 10,000 trajectories per player per iteration, 20,000 total; alternating player updates |
| Networks | Three branches per network; regret/policy 96-wide branches and 2×256 trunk; critics 96-wide branches and 2×192 trunk; calibration 64-wide branches and 2×128 trunk |
| Regret replay | Capacity 250,000 per player, reset for each player collection batch |
| Policy replay | 2,000,000-row historical reservoir |
| Critic replay | 1,000,000 transitions total, divided into two 500,000-row circular buffers |
| Calibration replay | 1,000,000 rows |
| Storage | Dense compact float32 replay, integer bookkeeping; no inherited FHP-only feature codebook |
| Parallel execution | Eight synchronous traversal actors as production starting point; deterministic budget partition; cached actor snapshots; central sequential learner fits |
| Development execution | One worker/one PyTorch thread by CLI default; real parallel integration test uses two actors |

**Leduc and FHP policy-fitting cadences differ.** Experiment 51 retained 20,000
policy updates every iteration, whereas the selected FHP system only fits its
policy at checkpoints. The runner restores training RNG state after each policy
fit. Its independent recovery/policy schedules exclude fitting, persistence and
downtime from the clock. Schedule state is saved with full continuation. An
explicit `--checkpoint-every` retains the older coupled iteration trigger.

## Production training specification — November 2026

These settings select one production learner, without a configuration sweep.
The regret budget increases modestly from 750 to 1,000 updates; the FHP-derived
critic and calibration budgets, learning rates, network sizes, sampling floor,
fixed beta and four-fit target averaging are retained. This is a reasoned
specification, not measured evidence of improved Hold'em strength.

The game permits at most 23 decisions per hand. At 10,000 traversals, even
counting every decision as a player-regret example gives at most 230,000 rows.
The 250,000-row buffer retains the complete default collection batch. Reassess
this bound before increasing traversals. Doubling historical policy replay while
shrinking the two regret allocations changes dense replay array capacity from
13,303,000,000 to 12,455,000,000 bytes. Grouping, fitting copies, workers, models
and checkpoint buffers are additional memory.

Each regret fit caches the previous frozen network's discounted/clipped outputs
plus the sampled advantages. The cache consumes no RNG, is rebuilt for every
fit, and is never checkpointed or reused after labels/targets change. Its size is
12 bytes per occupied regret row (three float32 actions). Critic TD targets and
collection both use uniform ties among maximum non-positive regrets; the
critic fallback uses original signed regrets, not their zero-clamped values.

Policy fitting groups complete canonical feature vectors before splitting them.
A deterministic BLAKE2b hash assigns 5% of groups to validation, keeping identical
information sets in one partition and membership stable across fits. Validation
uses iteration-mass-weighted soft-target cross-entropy. The training optimizer
never sees held-out groups during that fit. Checkpoints at 2,500, 5,000, 10,000
and 20,000 updates are compared; exact ties prefer the earlier checkpoint.
The selected model and corresponding Adam state are restored. This is model
selection within one fit, not early termination or additional self-play runs.
Small replays lacking either partition use all groups and explicitly mark
validation unavailable. Metrics expose counts, all validation scores, selected
update, and regret-cache cost. Validation is not an exploitability estimate.

Fresh training is required because configuration and source fingerprints changed.
Existing policy files remain playable. No long training or opponent-strength
experiment is part of this implementation.

## HUNL-specific changes in this initial repository

The learning mechanism and budgets transfer; the FHP trained weights and its
flop-only feature contract do not. New policies start from fresh initialisation.

* Two players, 52 cards, two private cards each, board progression `0 3 1 1`,
  no-limit betting, 200bb stacks, 50/100 blinds, no rake.
* OpenSpiel 1.6.3's FCPA abstraction: fold, check/call, pot-sized raise, all-in.
  There is no fixed-limit raise cap. This is a restricted betting abstraction
  of no-limit poker, not support for every legal chip-sized raise.
* The new encoder has **287 policy features and 339 critic features**. Cards
  occupy separate hole/flop/turn/river channels with global suit canonicalisation.
  Betting history is retained from OpenSpiel, chip amounts are divided by 20,000,
  and the context includes player identity, street and legal-action mask.
* Critics additionally receive both hole-card sets. Policy/calibration/regret
  inputs never contain the other player's cards. Public turn/river order is
  preserved even where OpenSpiel's native public-card mask loses it.
* The source's additional relative-strength and betting-interaction descriptors
  are **not blindly ported**: they assumed a flop-only fixed-limit game. The new
  representation is a deliberate initial HUNL implementation, not the identical
  231-dimensional FHP representation. Reintroducing useful four-street strength
  features and testing their compute/quality tradeoff is planned.
* Continuation and policy formats are renamed and reject old FHP checkpoints.
  Full states are sharded with checksums and include actor/replay RNGs. Fits and
  checkpoints occur only at completed outer-iteration boundaries.

## Efficiency priorities

The cache avoids repeating frozen target inference thousands of times per fit.
Reduced critic updates, bounded actor scratch replay, compact learner arrays and
checkpoint-only policy fits are retained immediately. No historical paired
policy diagnostic, exact game-tree enumeration, old experiment orchestration or
automatic cloud launch is retained.

The next profiling pass must include collection, feature construction, critic
cache building, calibration, regret fitting, policy grouping/fitting, persistence,
peak resident memory and total wall time. Do not infer full-game training cost
from the tiny smoke run. Grouping may save less work in full Hold'em because
repeated information sets are rarer. Only then test 500 critic steps, more
selective policy extraction, richer bet menus, or accelerator changes against
quality per total compute. Neither training nor opponent API benchmarking has
been run at production scale in this repository.
