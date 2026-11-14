# Hold'em strategic representation review

## Current v5: inexpensive blocker and backdoor features

The default `hulhe_strategic_limit_v5` has **417 policy / 605 critic inputs**.
V5 preserves all 91 v4 card channels exactly and appends four features per player:

* Fraction of board-compatible strongest holdings removed by our cards, and the
  mass of that original strongest class among legal opponent holdings. The class
  is defined before own-card removal, so blocking every member yields zero
  remaining mass instead of silently redefining the nuts.
* Exact flop runner-runner straight and flush completion fractions, requiring
  both remaining cards. These are structural draws, not probabilities of winning;
  they are zero before the flop and on turn/river.

The existing board table supplies blocker rankings. Backdoor calculations use
1,081 legal unordered runouts and rank/suit patterns, without additional opponent
or future-hand equity evaluations. The full immutable preflop descriptor vector
is retained in a dedicated 169-entry cache, independent of the general hand
cache. First-use sampling is unchanged; repeated preflop sampling after general
cache eviction is eliminated. All existing equity/category/transition estimates
are bit-for-bit preserved.

These features make exact relationships available to the network without
increasing hidden widths, depth, fitting budgets or introducing a new estimator.
Default dense replay grows by **144 MB** including calibration; online parameters
increase by 2,944 to **914,321**. Improvement in playing strength remains unmeasured.

V5 requires fresh training. Saved v2/v3/v4 policies remain playable with their
original feature contracts; v4 computations are frozen in `v4_card_features.py`.
The shared v2 card fixture adds independent cross-repository checks for all new
blockers/backdoors and the strict-current-nuts feature.


## Historical v4: transfers from the no-limit encoder

The default is `hulhe_strategic_limit_v4`: **413 policy inputs and 597 critic
inputs**. It retains every v3 descriptor, including the same sampled equity and
transition values, and appends one general nut-flush blocker per perspective.
This flags possession of the highest suited rank absent from a threatening
three-or-more-card board suit. For example K-clubs is flagged on A-clubs/7-clubs/
2-clubs. The original ace-blocker feature remains separate and unchanged.

A bounded 64-entry cache now ranks all board-compatible holdings once and reuses
the table across own hands. A board-first cache key improves reuse; the existing
own-first sampling key is deliberately retained. This optimization must not alter
the sampled showdown statistics. Tables are immutable and derived entirely from
public cards; own cards only filter the compatible-opponent holdings afterwards.

The three-branch network architecture and fitting budgets remain unchanged.
Dense replay grows by **36,000,000 bytes** at default capacities including
calibration; online model parameters rise from 910,641 to **911,377**.

Saved v2/v3 policies retain their original encoder metadata and feature semantics
for evaluation. V4 requires fresh training; old replay and optimiser continuation
states are rejected. The original v3 card module is frozen as
`legacy_card_features.py`. The identical shared fixture
`tests/fixtures/holdem_card_contract_v1.json` in both Hold'em repositories locks
common card features and all 169 preflop equity classes to the same values.

## Historical v3 design and measurements

The v2 Hold'em encoder preserved information, but required networks to rediscover
hand rankings, draw structure and pot arithmetic from sparse card/history bits.
FHP Experiment 36's selected AE representation already supplied exact hand/board
features, relative strength, betting economics and strength–betting interactions.
The initial Hold'em transfer retained the learning settings but omitted these
features. Version 3 restores that idea and extends it to four streets.

This is the chosen default implementation, not a demonstrated optimum. Small
correctness and integration tests cannot establish improved playing strength.
No production training or hyperparameter sweep is required to use the change.

## What the networks receive

`hulhe_strategic_limit_v3` supplies **412 policy inputs and 595 critic inputs**.
Every exact v2 channel remains represented: private cards, separate flop/turn/river,
player identity, complete public betting history, street and legal actions.
There is no lossy card bucket, history merging or rule-based action override.

Each player perspective adds 90 strategic card descriptors:

* Starting-hand ranks, pair/suited/connectivity and broadway/ace indicators.
* Exact best-five category and kickers, with correct 5/6/7-card semantics.
* Board ranks, pairs/trips/quads, suit texture, straight connectivity, overcards,
  overpairs/underpairs, rank matches, threatening-suit blockers and board-only hands.
* Exact current win/tie/loss fractions against all compatible opponent holdings,
  current hand strength, unbeatable and strictly unbeatable indicators.
* Exact next-card made-category probabilities, so flush/straight improvements
  and other immediate draws are visible separately from current strength.
* Final showdown win/tie/loss, equity and final-category distribution, plus joint
  loss/tie/win transitions from current board to showdown on flop/turn. These
  distinguish improvement potential from vulnerability to being overtaken.

There are 37 betting/context descriptors per perspective:

* Pot, log-pot, call cost, pot odds, total/current-round contributions, current
  bet size and raises remaining, with fixed-limit normalization.
* Position, next actor, facing a bet, previous action, whether a call/check closes
  the round/hand, and raise counts/last aggressor for each street.
* Strength–pot-odds and strength–position/action interactions. The immediate
  uniform-opponent showdown call value is a feature, **not** a decision rule or
  an estimate of the full value of calling before the river.

The critic receives both player perspectives and four additional values:
availability and player-0 win/tie/loss on the current board. This is privileged
control-variate information only. Policy, regret and calibration inputs never
receive the other player's actual cards. Neither network receives future cards.

All features lie in [0,1]. Signed interactions are shifted/scaled, unavailable
statistics have masks, and exact-vs-sampled showdown strength is explicit.
Metadata records every feature name, normalization, layout and sampling contract.

## Exact strength versus future equity

FHP ends on the flop; full Hold'em does not. Current five-card strength on the
flop is therefore not final showdown equity. These quantities have distinct
channels. On the river, all 990 compatible opposing holdings are enumerated and
strength/equity are exact. Current strength also enumerates all 1,081 holdings on
the flop and 1,035 on the turn. Next-card categories enumerate all 47/46 currently
unseen cards on flop/turn.

Preflop final statistics use 4,096 samples; flop/turn use 512. Each sample deals a
compatible opponent hand and completes the public board without replacement.
The reference prior is **uniform compatible holdings**, not an action-conditioned
opponent range. Exact betting history remains available for learning how observed
actions change decisions relative to that reference.

Sampling uses a local PCG64 generator seeded by a versioned hash of canonical own
and public cards. It never consumes Python/NumPy/Torch training randomness, reads
actual opponent cards or changes after cache eviction/restart. Global suit
permutations and flop order do not change features. The equity sampling standard
error is included (at most about 2.2 percentage points for 512 draws; this is not
a bound on actual error). Sampled zero/one equity is never treated as proof of
being unbeatable: those indicators come only from exact current enumeration.

Full exact preflop/flop equity enumeration is too expensive on the per-state
collection path. The bounded sampling budget and 8,192-entry visible-card cache
make richer inputs practical. Public betting summaries have their own 4,096-entry
cache; exact critic comparisons use an 8,192-entry cache. Cached arrays are
immutable and contain derived data, not learned parameters or RNG state.

The choice to expose distributions and transitions, rather than only mean
strength, is informed by poker abstraction research on strength and potential.
See [Johanson et al., Evaluating State-Space Abstractions in Extensive-Form Games](https://johanson.ca/publications/poker/2013-aamas-kmeans-abstraction/2013-aamas-kmeans-abstraction.html).
Here those descriptors supplement exact inputs instead of merging information
sets. Separate card and betting processing also follows the design direction of
[Deep CFR](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf); the chosen
implementation adds a third branch for the explicit strategic card descriptors.
Neither paper establishes that this particular implementation is optimal.

## Network and learning decisions

Each structured network has three branches: exact cards, strategic card
summaries, and betting context. Their outputs feed a shared hidden trunk.
This prevents the new strategic descriptors from sharing a single narrow input
projection with all betting history. Following the capacity review, regret and
average policy use 96-wide branches and a 256/256 trunk; critics use 96-wide
branches and a 192/192 trunk. Calibration retains 64-wide branches and a 128/128
trunk, with its existing action/control-variate fields in the context branch.
See [the baseline](BASELINE.md#current-hulhe-network-capacity) for parameter
counts, configuration keys and the evidence behind these choices.

The unbiased UCV estimator, cross-fit critics, cached targets, target averaging,
exploration floor, regret updates, grouped average-policy loss and numerical
training budgets remain unchanged. Returns already normalize by the game's
maximum absolute payoff (2,400 chips), so larger chip denominations do not inflate
the learning targets. Changing these mechanisms simultaneously would make it
harder to attribute a result and would discard the FHP evidence without a
replacement. No hard-coded fold/call safeguards or guessed opponent ranges are
introduced.

## Compatibility and operating cost

Fresh training uses v3 automatically. Use a new output directory; v2 replay and
continuation states are rejected because dimensions and semantics changed.
Existing v2 **policy** checkpoints still load with their original two-branch
network and original encoder for evaluation. Policy file format stays at version
2 because its explicit encoder/model metadata distinguishes both representations.

For the default million-entry buffers, added float32 feature storage is about
4.60 GB (4.29 GiB) across regret, average-policy, critic transition and calibration
replay, excluding working batches and cached target predictions. This is a
material cost of retaining exact inputs alongside the derived summaries.

## Verification

Tests cover independent exhaustive best-five ranking, OpenSpiel showdown parity,
royal-board ties/private nuts, exact straight/flush draw counts, all suit
permutations, flop permutation, hidden-card isolation, preserved exact v2 inputs,
all local betting branches including the blind option, legal deterministic
sampling, unchanged global RNGs, cold-cache continuation, checkpoint metadata,
legacy-policy evaluation and real Fell Omen 2 integration. See
[TESTING.md](../TESTING.md) for measured run results.
