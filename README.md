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

The strategic encoder supplies **412 policy inputs** and **595 critic inputs**.
It preserves exact cards and betting history, and adds hand strength, draw
potential, board texture, pot odds and betting interactions. Three network
branches process exact cards, strategic card descriptors and betting context.
See [the representation review](docs/STRATEGIC_FEATURES.md) for exact versus sampled
statistics, costs and design choices. Learning objectives and budgets are unchanged.
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
playing strength. Tests exercise real OpenSpiel and PyTorch. Ray tests start
real local worker processes and are opt-in.

The inherited learning budget is in
[`hulhe_escher/data/baseline.json`](hulhe_escher/data/baseline.json).
An explicit training invocation using the original eight-worker arrangement is:

```bash
python -m hulhe_escher.train --iterations 100 --workers 8 --threads 8 \
  --checkpoint-every 50 --seed 0 --output outputs/baseline-seed0
```

This is a substantial workload: 10,000 trajectories **per player per iteration**,
large replay buffers, and 20,000 policy updates per checkpoint. Benchmark memory
and throughput on the intended training machine before choosing a long run.
The command uses the current machine and does not submit any cloud jobs.

Each checkpoint saves a playable policy and a checksummed, sharded continuation
state containing networks, optimisers, replay, temporal critic history and RNGs,
including actor RNGs. Only complete outer iterations are saved. `--resume` checks
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

The HUNL source history is retained in this independent local repository. No
remote is configured. Ignored outputs, checkpoints, environments and build caches
were not copied. The source HUNL repository is untouched.

The default encoder is `hulhe_strategic_limit_v3`. Policy checkpoint format 2
supports both this encoder and the original fixed-limit v2 encoder for evaluation.
Full continuation checks the exact game and encoder: **start fresh training for
v3**. Old v2 replay cannot be reused. HUNL/preparation-copy policies remain rejected.

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
