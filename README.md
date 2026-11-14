# Limit Hold'em — HULHE UCV-ESCHER

A preparation repository for developing Unbiased Control-Variate ESCHER in
**heads-up fixed-limit Texas Hold'em (HULHE)**, copied from `hunl-ucv-escher`
at commit `cf456b3b1a8b41bd5f35fd6dd75059d9e13a8b44`.
The learner comes from the efficient FHP Experiment 36/38 implementation,
with **2,500 updates per critic per fit**.
See [the baseline specification](docs/BASELINE.md) for the choice and evidence.

**Status: repository preparation only. Fixed-limit rules are not implemented yet.**
This copy changes project identity, imports, CLI and checkpoint namespaces;
the learning algorithm, budgets, encoder behavior and game rules remain inherited.
The executable game still has two players, 52 cards, preflop/flop/turn/river,
50/100 blinds and 20,000-chip stacks reset every hand. The FCPA abstraction
allows fold, check/call, pot-sized raise and all-in when legal.

The first development step is to replace these rules and the corresponding
encoder/action contract with fixed-limit Hold'em. See [the development plan](docs/HULHE_PLAN.md).
Current smoke runs exercise the inherited no-limit game; they are not limit
training or limit benchmark results. Run manifests explicitly record this status.

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
| `tests/` | Inherited no-limit rules, information boundaries, cache equivalence, resume and actors |
| `docs/` | Baseline evidence, provenance and next implementation steps |

The HUNL source history is retained in this independent local repository. No
remote is configured. Ignored outputs, checkpoints, environments and build caches
were not copied. The source HUNL repository is untouched.

HULHE policy, model, encoder and continuation identifiers have their own namespace;
existing HUNL checkpoints are not interchangeable. Fixed-limit conversion must
version these contracts again where their shapes or semantics change.
