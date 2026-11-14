# Verification

The current suite validates the fixed-limit game and the learner boundary.
The opt-in opponent suite tests the actual pinned Java core and strategy tables.
These are integration checks, not a playing-strength benchmark.

Use the pinned dependencies and Python 3.10–3.12.

```bash
python -m pytest -q
RUN_RAY_TESTS=1 python -m pytest -q -m ray
python -m hulhe_escher.fo2_setup
RUN_FO2_TESTS=1 python -m pytest -q -m fo2
python -m hulhe_escher.train --smoke --iterations 2 --output outputs/smoke
python -m hulhe_escher.train --smoke --iterations 3 --output outputs/smoke --resume
```

The suite enumerates every within-street betting sequence against independent
fixed-limit rules, including the big-blind option, check-raises and the four-bet
cap. It also covers the longest hand, exact win/fold/tie payoffs, zero-sum
settlements, suit symmetry, hidden-card isolation, public turn/river order,
285/337-dimensional features, three-action unbiased estimation,
sampling support, checkpoint policy probabilities, cached/uncached learner
agreement and exact continuation through replay, optimisers and critic history.
The opt-in Ray test checks real worker collection and exact parallel continuation.

Smoke fits actual networks and writes actual checkpoints using reduced budgets.
It does not establish strategy quality, full-budget throughput, cloud deployment
readiness or integration with a fixed-limit benchmark opponent.

## Fixed-limit interface validation — 2026-11-14

* Sequential suite: 20 passed, with the opt-in Ray test skipped.
* Real two-actor collection and exact parallel continuation: 1 passed separately.
* Version 0.2.0 wheel built and installed; the installed CLI completed two smoke
  iterations and resumed to iteration three from outside the source checkout.
* Old no-limit encoder ids, checkpoint versions and game contracts are rejected.
* Reference-rule tests enumerate every legal betting sequence within each street;
  the longest full hand checks every street's cap and the 2,400-chip payoff bound.

The learner and numerical budgets are unchanged. This validates the game and
state interface, not playing strength or interoperability with a running opponent.

## Historical preparation-copy validation — 2026-11-14

The following results describe commit `f025fe7`, before fixed-limit conversion.

* Sequential suite: 10 passed; the opt-in Ray test was skipped in this run.
* Separate real Ray collection/resume test: 1 passed (two local actors).
  Ray required execution outside the macOS process-inspection sandbox.
* Built the `hulhe_ucv_escher` wheel and installed it in a separate temporary
  environment. Ran the installed `hulhe-train` from outside the repository for
  two smoke iterations, then resumed to iteration three successfully.
* Checked wheel package names and bundled baseline configuration, source-file
  provenance hashes, and the manifest's explicit inherited-no-limit status.
* Compared every Python source against the source repository after namespace
  substitution: only package/CLI documentation and game-status metadata differ.
  All learner code and numerical baseline settings are preserved.

Validation used Python 3.12, NumPy 1.26.4, PyTorch 2.7.0, OpenSpiel 1.6.3 and
Ray 2.51.2. The available test environment had pytest 9.0.2; the declared dev
extra remains pytest 8.x as inherited. No full-budget training was run.

## Opponent integration validation — 2026-11-14

See [FELL_OMEN2.md](docs/FELL_OMEN2.md) for the bridge's reference comparisons,
strict free-fold rejection, explicit policy adaptation and reproducibility tests.
A 40-hand strict match against a tiny learned checkpoint completed without
substitutions; setup and training are not part of the timed match loop.

* Full suite with `RUN_FO2_TESTS=1`: 34 passed, 1 skipped in 9 seconds. The
  unchanged learner's opt-in Ray test was not rerun for this integration.
* Built and installed the version 0.3.0 wheel in a separate environment. From
  outside the checkout, the installed `hulhe-evaluate` reproduced all 40 logged
  hands exactly, using the same checkpoint, assets and seed.
* The source and installed match loops took approximately 1.6 and 1.2 seconds,
  respectively. Both used strict rejection and made 94 opponent decisions.
* Verified that the wheel includes the Java bridge sources and pinned asset
  manifest, without bundling upstream binaries or strategy archives.
