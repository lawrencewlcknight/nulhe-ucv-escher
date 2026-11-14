# Local Fell Omen 2 benchmark adapter

The interface uses OpenSpiel as the sole dealer/rules engine and a persistent
Java process as the opponent. It supports existing version-2 learned-policy
checkpoints. No training, GPU, network service or Maverick application is needed
for a match.

## Setup and lightweight run

Install this repository's Python dependencies and a JDK (Java 11 or later).
The development machine has Temurin 25; compilation targets Java 11.

```bash
python -m hulhe_escher.fo2_setup
RUN_FO2_TESTS=1 python -m pytest -q -m fo2
python -m hulhe_escher.evaluate --checkpoint outputs/smoke/policy_000003.pkl \
  --pairs 20 --seed 20261114 --output outputs/fo2-evaluation
```

If no checkpoint exists, create a small one using the README's smoke training
command, or use `--baseline call` / `--baseline random` instead of `--checkpoint`.
These baselines test the interface without fitting a network.

Setup downloads approximately 74 MB of strategy archives plus small Java
sources/dependencies, verifies pinned SHA-256 hashes, and compiles the bridge.
The ignored default cache is `.cache/fell_omen2`. Run setup with `--offline` to
rebuild using only cached files. Use `--assets /absolute/path` with both setup
and evaluation to share a cache or run outside the repository. Matches never
fetch assets automatically. A corrupt cache fails explicitly.

Installed commands: `hulhe-fo2-setup` and `hulhe-evaluate`.
Every evaluation requires a new empty output directory. `--pairs 20` means
40 hands: each nine-card deal is played twice with the candidate changing seats.
The physical hole-card/board allocations remain fixed within each pair.

## Fidelity and the known free-fold incompatibility

The opponent is the **Fell Omen 2 implementation and strategy assets in the
Maverick mirror**, pinned to commit
`916315f87e78cec33369ebd404170d2cb219c7b3`. This is not a certification that the
mirror is byte-for-byte the original 2008 tournament executable.

Its classifier, hand evaluator, positive/negative potential calculation, bucket
thresholds, action-node mapping and decision methods are compiled from unchanged
upstream sources. A subclass intercepts the selected strategy table before the
original decision methods reach `Math.random()`. Python then samples the returned
probabilities with independent seeded evaluation RNGs. The Java heap is limited
to 256 MB. Native evaluation libraries are disabled for reproducibility.

The optimized table reader caches ZIP indices and a bounded number of matrices.
Duplicate filenames use the first archive, exactly as the original reader does.
A `reference` request invokes the original ZIP scanner/matrix parser and recomputes
classification without the bridge's caches. Integration tests compare these paths
and committed reference fixtures. The original adapter's terminal-history mapping
method is extracted verbatim at build time and used with the original action-node
method to independently check Python's reconstructed public history.

**Some upstream tables assign a fold when checking is free.** For example,
`p1_strategy828080_5.txt`, bucket 1, returns fold/check/raise probabilities
`[0.308218, 0.426898, 0.264884]` at a river opening decision. The original Java
reader gives the same values. OpenSpiel prohibits that free fold. No original
host correction was established, so strict mode does not invent one.

The default `--fo2-free-fold error` stops and records such a decision. It never
silently drops illegal mass or substitutes an always-call policy. This means a
longer strict evaluation can stop even after a successful smoke test.

An explicitly selected `--fo2-free-fold check` transfers only that free-fold
probability to check, preserving raise mass. It is a **policy adaptation**:
its manifest labels the variant, and each affected decision records both the
raw upstream probabilities and `explicit_free_fold_to_check`. Do not combine
strict and adapted results under the same benchmark label. Other illegal
probabilities, missing tables, timeouts and process errors still fail. Probability
sums are normalized only within a 2e-6 tolerance for decimal table rounding.

## Data flow and information boundary

Python uses `LoadedHULHEPolicy` for the learned algorithm. For FO2 it constructs
a sanitized JSON request containing only:

* The acting player's two hole cards, encoded as rank/suit strings.
* Currently revealed board cards, retaining flop/turn/river order.
* Seat, street, public betting sequence, raise count and prior-round history codes.

Opponent hole cards and unrevealed board cards never cross the bridge. Numeric
card identifiers are not passed between libraries: OpenSpiel and Meerkat use
different suit/index conventions. Player 0 is the button/small blind (FO2 p2),
player 1 is the big blind (FO2 p1). Each request reconstructs the original core's
round counter; caches contain only functions of visible cards or table names.
There is no persistent hand-history state to leak between hands.

Each response includes the raw fold/check-call/bet-raise probabilities, strategy
filename, hand bucket, flop bucket, action node, strength and potential values.
The original Java core also checks the translated history before querying tables.
Messages travel over local stdin/stdout; diagnostic stderr is kept separate.
Timeouts or protocol errors terminate the process so stale replies cannot be
mistaken for another decision. One policy instance serves one sequential match.

## Logs and reproducibility

Each output directory contains:

* `manifest.json`: exact game contract, seed, deal scheme, policy/checkpoint hash,
  opponent source commit, bridge/build fingerprints and runtime versions.
* `hands.jsonl`: deals, seats, legal actions, probabilities, selected actions,
  opponent lookup diagnostics and chip returns. These local audit logs include
  both players' cards; the opponent requests do not.
* `summary.json`: completed pairs/hands, small bets per hand, paired standard
  error and an approximate paired t interval. One small bet/big blind is 100 chips;
  one big bet is 200 chips. The duplicate pair is the uncertainty unit.
* `failure.json`, on failure: completed-hand count and the error; no success
  summary is written for an incomplete run.

Separate local NumPy generators control deals and each seat's actions. Running
with the same seeds, checkpoint, pinned sources and runtime reproduces the hand
log. The runner does not update either policy. Tiny samples validate integration,
not competitive playing strength, regardless of the observed score or interval.

## Lightweight validation performed

The opt-in suite exercises both seats and every within-street betting branch,
blind completion/check-raise/cap histories, ten original-Java reference cases
(including a duplicate filename),
hidden-card isolation, repeated requests across hands, invalid request/timeout
handling, strict detection of the upstream free-fold issue, explicit adaptation,
and repeated matches against simple policies. Core tests verify duplicate deals,
zero-sum accounting, unchanged global NumPy RNG, asset hashes and error logging.

A version-2 checkpoint from three tiny smoke iterations completed **20 pairs / 40
hands in strict mode**, with **94 opponent decisions** and no action adaptations.
The match loop took approximately **1.6 seconds** on the development machine
(excluding setup, imports and asset verification). Runtime depends on how often
hands reach postflop. This is an engineering smoke result, not a strength claim.

## Provenance and attribution

[The asset manifest](../hulhe_escher/data/fell_omen2_assets.json) records every
pinned URL, SHA-256 digest and Git blob id. The primary project description is
[Ian Fellows' download page](https://www.deducer.org/pmwiki.php?n=Main.DownloadINOTAndFellOmen).
The cached sources retain their upstream headers; setup also downloads the
mirror's GPL text, the strategy notice and the University of Alberta evaluator's
redistribution notice. The Meerkat API JAR is taken unchanged from the same
pinned mirror. External sources, binaries and strategy data remain in the local
cache rather than being copied into the Python wheel or committed repository.
The generated reference method remains attributable to the pinned Fell Omen source.
