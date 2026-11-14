# Limit Hold'em — HULHE UCV-ESCHER

A research repository for Unbiased Control-Variate ESCHER in **heads-up
fixed-limit Texas Hold'em (HULHE)**, with **Fell Omen 2** as the selected challenge
opponent. The learner comes from FHP Experiment 36/38 through the HUNL baseline,
with **2,500 updates per critic per fit**. See [the baseline evidence](docs/BASELINE.md).

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
1,000 regret updates per player and retains the 2,500-update critics and
2,000-update calibration learner. Regret replay is 250,000 rows per player;
historical policy replay is two million rows.
Regret and average-policy networks use 96-wide input branches and a 256/256
hidden trunk; critics use 96 and 192/192; calibration retains 64 and 128/128.
The six online networks total 914,321 trainable parameters. See the
[capacity specification](docs/BASELINE.md#current-hulhe-network-capacity).
See [the game-interface specification](docs/GAME_INTERFACE.md) for evidence,
seat conventions and the complete feature layout.

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
  --recovery-seconds 3600 --policy-seconds 21600 --seed 0 --output outputs/baseline-seed0
```

This is a substantial workload: 10,000 trajectories **per player per iteration**,
large replay buffers, and up to 20,000 policy updates per fit. Benchmark memory
and throughput on the intended training machine before choosing a long run.
The command uses the current machine and does not submit any cloud jobs.

Recovery saves occur every hour of active learner time, and playable policies
are fitted every six hours. Both are written at the requested final iteration.
Policy validation uses a stable 5% information-set holdout, checks updates
2,500/5,000/10,000/20,000 and retains the best model and matching optimizer.
It follows one fitting trajectory, not separate training experiments. All 20,000
updates execute; selection can retain an earlier checkpoint. Tiny smoke replays
without both split partitions fit all groups and report validation unavailable.
The optional `--checkpoint-every N` also triggers both operations every N
iterations; smoke defaults to every iteration. Policy fitting, persistence and
downtime do not advance the active learner clock, and boundaries can overshoot
by one outer iteration. The two schedule clocks are restored on resume.

Checksummed, sharded continuation states contain networks, optimisers, replay,
temporal critic history and RNGs, including actor RNGs. Recovery saves do not
require a policy fit. Only complete outer iterations are saved. `--resume` checks
the source fingerprint, runtime, seed, configuration and worker arrangement;
`--iterations` is the desired **total**, not additional iterations. Use trusted
checkpoint files only; their PyTorch/pickle metadata is not an untrusted format.
All checkpoint directories are retained; provision disk space for the chosen
cadence. A retention policy and cloud runner remain future work.

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
