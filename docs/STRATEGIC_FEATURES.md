# Hold'em strategic representation review

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

Each structured network has three 64-wide branches: exact cards, strategic card
summaries, and betting context. Their outputs feed the existing hidden trunk.
This prevents the new strategic descriptors from sharing a single narrow input
projection with all betting history. The average-policy trunk remains 192/192;
regret and critic trunks remain 128/128. Calibration uses the same representation
and preserves its existing action/control-variate fields in the context branch.

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
