# Verification

The current suite validates the fixed-limit game and the learner boundary.
The opt-in opponent suite tests the actual pinned Java core and strategy tables.
These are integration checks, not a playing-strength benchmark.

## Encoder v5 validation — 2026-11-15

* Offline suite: **53 passed, 8 skipped**; skips are opt-in Ray/Fell Omen 2 tests.
* Real two-worker collection and exact continuation: **1 passed** separately.
* Independent enumeration verifies general nuts-removal and exact flop backdoor
  completions. All 1,326 preflop holdings retain every original v4 value; the
  complete 169-class cache survives eviction from the general hand cache.
* Frozen v4 metadata and nonzero saved-policy predictions are preserved. V5
  continuation passes with cold caches; continuation from older layouts is rejected.
* Cross-repository audit: 30,000 ranked hands, 800 random card cases with 51 shared
  channels, all 169 preflop equity classes, and 612 decisions across 60 matched
  full hands agree. Shared fixtures pass independently in each repository.
* Single matched 100-hand/800-decision timing: v4/v5 policy-plus-critic encoding
  took 4.134/4.366 ms per decision on the first pass and 0.208/0.242 ms on the
  repeated pass. These local measurements include within-hand reuse and are
  not training-throughput or playing-strength evidence.
* Widths and fitting budgets are unchanged. Default online parameters total
  914,321; dense replay growth is 144 MB including calibration. Details and raw
  measurements: [ENCODER_V5_VALIDATION.json](docs/ENCODER_V5_VALIDATION.json).

## Encoder v4 validation — 2026-11-15

* Offline suite: **44 passed**, with the opt-in Ray/Fell Omen 2 tests skipped.
* Real two-actor collection and exact continuation: **1 passed** separately.
* Board caching preserves every original v3 card descriptor bit-for-bit,
  including sampled equity and transitions, across 80 random card states.
* Tests check shared board-cache hits, immutable arrays, general nut-flush versus
  ace blockers, preserved v3 policy predictions, and cold-cache continuation.
* Shared fixture: 193 cases including all 169 preflop suit classes, identical
  to the fixture tested independently by the no-limit repository.
* Version 0.5.0 wheel built without downloads, installed separately, and passed
  a smoke iteration, policy fit and checkpoint reload outside the checkout.
* Matched 100-hand/800-decision audit: policy-plus-critic encoding changed from
  5.169 to 4.321 ms on the first pass; repeated passes were both about 0.210 ms.
  These are feature-construction measurements with within-hand cache reuse,
  not training throughput or playing strength. See
  [ENCODER_V4_VALIDATION.json](docs/ENCODER_V4_VALIDATION.json).

## Network capacity validation — 2026-11-14

* Standard suite: **40 passed, 8 skipped** in 16.34 seconds. The skipped tests
  were the opt-in Ray and Fell Omen 2 suites.
* Real two-actor collection and exact parallel continuation: **1 passed** in
  17.83 seconds, outside the macOS process-inspection sandbox.
* Verified all six online networks' dimensions and their combined **910,641**
  trainable parameters. Regret sizing no longer resizes the critics or calibration.
  Invalid configured dimensions are rejected before construction.
* The standard suite trains the recommended architecture with reduced budgets,
  fits/reloads its policy, compares cached and uncached learning, and checks exact
  continuation both directly and through the CLI. Older policy format coverage
  also passes. Learning budgets and replay dimensions are unchanged.
* No long training run or playing-strength evaluation was performed for this change.

## Strategic representation validation — 2026-11-14

* V3 feature suite plus existing game/learner/opponent tests: **42 passed**, with
  the opt-in Ray test skipped in that invocation (17.6 seconds).
* Real two-actor collection and exact parallel continuation: **1 passed**
  separately (15.5 seconds), outside the macOS process-inspection sandbox.
* Ranking verified against independent enumeration of every five-card subset
  for 1,500 random five/six/seven-card hands plus pathological cases, and against
  OpenSpiel outcomes on 100 deals. Exact draws, nuts/ties, hidden-information
  boundaries, all suit permutations and public betting branches also pass.
* Three smoke training iterations wrote a new v3 checkpoint. It completed
  20 duplicate pairs / 40 hands against strict Fell Omen 2 in 2.7 seconds,
  without action conversion. This is not evidence of playing strength.
* Built/installed v0.4.0 outside the checkout: its evaluation reproduced every
  hand exactly. The installed loader also loaded the original v2 checkpoint.
* A 128-state timing check (32 deals per street) measured 5.72 ms per cold policy
  encoding and 0.10 ms warm; critic encodings measured 5.48 ms after the policy
  pass and 0.14 ms warm. These are local feature timings, not training throughput.
  Policy parameters increased from 80,771 to 101,251. Full default replay adds
  4,604,000,000 bytes of feature storage.

See [the design review](docs/STRATEGIC_FEATURES.md) for feature definitions,
approximation limits and fresh-training requirements. Tests used the same pinned
runtime dependencies as below, with pytest 9.0.2 available locally.

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
