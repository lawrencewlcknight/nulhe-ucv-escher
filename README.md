# HUNL UCV-ESCHER

A clean research repository for training Unbiased Control-Variate ESCHER in
heads-up no-limit Texas Hold'em. The learner is extracted from the efficient
FHP Experiment 36/38 implementation, with **2,500 updates per critic per fit**.
See [the baseline specification](docs/BASELINE.md) for the choice and evidence.

The initial game has two players, a full 52-card deck, preflop/flop/turn/river,
50/100 blinds, and 20,000-chip stacks reset every hand. The initial **FCPA action
abstraction** allows fold, check/call, pot-sized raise and all-in when legal.
This is a runnable development baseline, not a trained competitive model.
It does not yet support arbitrary opponent bet sizes or Slumbot/GTO Wizard API
play. Those require the action/state adapter work in [the development plan](docs/HUNL_PLAN.md).

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

## Verify and run

```bash
python -m pytest -q
python -m hunl_escher.train --smoke --iterations 2 --output outputs/smoke
python -m hunl_escher.train --smoke --iterations 3 --output outputs/smoke --resume
RUN_RAY_TESTS=1 python -m pytest -q -m ray
```

Smoke reduces replay, minibatch and optimiser budgets; it is not evidence of
playing strength. Tests exercise real OpenSpiel and PyTorch. Ray tests start
real local worker processes and are opt-in.

The inherited learning budget is in
[`hunl_escher/data/baseline.json`](hunl_escher/data/baseline.json).
An explicit training invocation using the original eight-worker arrangement is:

```bash
python -m hunl_escher.train --iterations 100 --workers 8 --threads 8 \
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
| `hunl_escher/` | Game, four-street encoder, configuration, CLI and checkpoints |
| `unbiased_escher/` | UCV estimator, structured learners, compact replay, parallel collection |
| `adaptive_escher/` | Persistent frozen-target critic and target cache |
| `vr_deep_cfr/` | Inherited neural cumulative-regret and replay foundations |
| `tests/` | HUNL rules, information boundaries, cache equivalence, resume and actors |
| `docs/` | Baseline evidence, provenance and next implementation steps |

No old FHP experiment runners, results, checkpoints, cloud launchers, evaluation
suite or Git history were copied. The original FHP repository is untouched.
