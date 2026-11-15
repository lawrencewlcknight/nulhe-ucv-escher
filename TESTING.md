# Verification

## Training continuation across process exits — 2026-11-15

* Focused checkpoint, recovery and resume suite: **19 passed, 1 skipped**.
  Skips are the existing opt-in Ray tests, previously verified separately.
* A new regression test launches separate Python CLI processes: one trains for
  three iterations continuously; another trains for two, exits, reloads its
  saved recovery, and trains the third iteration. Both use full production
  network dimensions with reduced smoke replay and update budgets.
* The resulting online/target networks, Adam states, persistent replay, critic
  target history, gate state, RNG states and training counters match exactly.
  Target versions advance after reload, demonstrating further training rather
  than inference-only loading. Latest-only retention also remains effective.
* Existing tests cover recovery-only and policy-bearing checkpoints, save
  failures, schedule restoration and rejection of incompatible continuation.
* Resume usage and required run-directory files are documented in the README.
  No changes to learner or checkpoint implementation were necessary.


## Six-hour wall-clock recovery — 2026-11-15

* Final focused scheduler, retention and cross-variant checks: **16 passed**.
* Timed recovery checks default to six wall-clock hours, on a fixed run-start
  grid. Saves less than one hour old suppress that slot; exactly one hour is
  eligible. Tests cover skipped slots, long downtime and disabled timing.
* A real interrupted smoke run verifies that a skipped six-hour check persists
  in latest-pointer metadata without rewriting replay. Resume at hour seven
  does not repeat the skipped save; the next grid boundary remains hour twelve.
* Actual successful save completion starts the suppression window. Policy and
  final saves remain unconditional. Latest-only retention and failure protection
  continue to pass; no production training or cloud jobs were run.


## Latest-only recovery retention — 2026-11-15

* Focused retention, runner/resume, fitting and cross-variant checks: **66
  passed, 1 skipped** (opt-in Ray tests; collection code is unchanged).
* A real smoke run leaves only its newest recovery, keeps all playable policies,
  and successfully resumes from the retained state.
* Injected write, verification and latest-pointer publication failures preserve
  the previous recovery. Cleanup cannot begin before latest-pointer publication.
* Multiple older saves are removed together. Unrelated files/directories and
  directory symlinks are preserved. Cleanup does not repeat checksum scans of
  the new replay arrays; its time is included in persistence metrics.



## Cross-variant training alignment v0.8.0 — 2026-11-15

* Full offline suite after implementation: **95 passed**, 8 opt-in cases
  skipped. After adding the final regression cases, the affected tests passed
  again: **30 passed**. Real Ray collection and exact continuation:
  **1 passed** separately. Ray required execution outside the macOS sandbox
  because its process inspection is restricted there.
* Cached and uncached regret fitting agree on weights, Adam state and replay
  RNG progression, including partial/full batches and reservoir replacement.
  Large targets exercise actual clipping before optimizer steps. Empty regret
  batches preserve the learner; underfilled control replay preserves targets.
* Collection and critic bootstrap policies agree for signed, tied, tiny positive
  and masked regrets. Explicit cases verify uniform ties only among maxima.
* Stable grouped policy holdout and iteration-mass-weighted validation are tested;
  selecting an earlier checkpoint restores its model and optimizer together.
* Simulated interruption after recovery-only **and** policy-bearing saves resumes
  with exact networks, optimizer states, replay and RNGs. Independent active and
  wall clocks, overdue recovery after downtime, and final saves are covered.
* The sibling-repository guard allows only the documented game-size differences
  and checks identical fitting, replay scheduling and checkpoint scheduling code.
  It skips when the other repository is not checked out alongside this one.

Defaults and CLI semantics: [shared specification](docs/TRAINING_ALIGNMENT.md).
Use a fresh output directory; older continuation fingerprints are rejected.
Saved policy inference remains compatible. No production training, parameter
sweeps, cloud jobs or external opponent calls were run for this alignment.


The current suite validates the fixed-limit game and the learner boundary.
The opt-in opponent suite tests the actual pinned Java core and strategy tables.
These are integration checks, not a playing-strength benchmark.

## Production training specification validation — 2026-11-15

* Offline suite: **68 passed, 8 skipped**. Skips are the opt-in Ray/Fell Omen 2 suites.
* Real two-worker collection and exact resumed continuation: **1 passed** separately.
* Regret-cache tests compare model weights, Adam state and replay RNG against
  uncached fitting with undersized batches, partial cache batches and saturated
  replay. Changing labels and target iteration verifies per-fit rebuilding.
* Scalar collection and batched critic targets agree on zero ties, negative
  ties, unequal negative values, illegal actions and positive regret matching.
* Policy checks cover stable information-set-disjoint validation, preservation
  of iteration-weighted cross-entropy, and restoring the best model together
  with its matching optimizer state.
* Recovery-only checkpoints, independent schedule clocks, serialized schedule
  restoration, final saves, CLI resume and policy-fit RNG isolation pass.
* Production settings and the limit-game bound on regret-buffer occupancy are
  checked. No configuration sweep, long training, production-memory benchmark
  or new playing-strength evaluation was run.

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

## Replay retention and indexed grouping v0.9.0 — 2026-11-15

`tests/test_replay_grouping.py` compares the grouped targets, masks and objective
weights exactly with the former dense NumPy reference across chunk sizes and
reservoir overflow. It exercises forced fingerprint collisions, signed zero,
invalid rows, inconsistent masks, legacy float64 storage, lazy minibatch gradients,
validation losses, replay mutation and untouched RNG state. Production grouping
retains a view of the original float32 feature matrix.

`tests/test_replay_integration.py` checks the new default capacity, all four real
encoder street labels for both players, saturated occupancy metrics, allocated
bytes, and a complete smoke collection/policy fit. Cross-variant tests permit the
intentional policy-capacity difference and require identical grouping code.

Bounded memory checks use `tools/check_policy_replay_memory.py`; add `--reference`
in a separate process to measure the former implementation. Measured 100,000-row
HUNL incremental peaks were 18.5 MB indexed versus 373.0 MB dense; 250,000-row
checks completed too. See [raw measurements](docs/REPLAY_MEMORY_CHECK.json).
The host has 8 GiB RAM, so no full-capacity multi-worker training-memory claim is
made. Synthetic timing and memory results do not establish policy strength.

## Learning-experiment integration

`python -m pytest -q tests/test_learning_experiment.py` tests two real training
processes, recovery, playable P4/P8 exports, deterministic multi-process duplicate
deals, paired statistical planning, incomplete/smoke claim suppression, a small
memory-probe round trip and a source-only bounded cloud plan.

`RUN_FO2_TESTS=1 python -m pytest -q tests/test_learning_experiment.py -m fo2`
additionally runs the pinned local Java opponent against both smoke policies.
It needs the assets prepared with `python -m hulhe_escher.fo2_setup`. No cloud
resources or remote poker games are created by these tests. Full-capacity memory
profiling and an eight-active-hour production run require a suitable VM.
