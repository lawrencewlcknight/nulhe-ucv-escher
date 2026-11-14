# Fell Omen 2 game interface, version 1

The learner receives native OpenSpiel states from `load_hulhe_game()`, then
encodes them through `HULHEFeatureEncoder`. OpenSpiel remains responsible for
chance probabilities, legal actions, betting transitions and hand settlement.
There is no separately implemented poker engine or opponent-specific feature
bucketing in the learner.

## Rules and benchmark evidence

The target is two-player fixed-limit Texas Hold'em, with four bets per street,
including the big blind preflop. There are two private cards per player, a
52-card deck, and three flop cards followed by one turn and one river card.

| Property | Contract |
|---|---|
| OpenSpiel version/game | 1.6.3 / `universal_poker` |
| Contract id | `hulhe_fell_omen2_limit_v1` |
| Player 0 | Small blind/button; first preflop, last postflop |
| Player 1 | Big blind; last preflop, first postflop |
| Blinds | 50 / 100 chips |
| Fixed bet sizes | 100 / 100 / 200 / 200 chips |
| OpenSpiel `firstPlayer` | `1 2 2 2` (one-based engine setting) |
| OpenSpiel `maxRaises` | `3 4 4 4` |
| Action ids | 0 fold; 1 check/call; 2 bet/raise |
| Maximum contribution | 400 + 400 + 800 + 800 = 2,400 chips per player |
| Stacks | Nonbinding bankroll; no stack carryover or short-stack all-in rules |
| Returns | Net chips, zero-sum, no rake |

The small blind may fold, complete or raise preflop. A completion preserves the
big blind's check/raise option. A player cannot fold when checking is free. Once
four bets are reached, the responding player may only fold or call. The cap
continues to apply heads-up; casino variants with uncapped heads-up raising are
not this benchmark game.

Sources checked:

Pinned URLs and inspected-file hashes are recorded in
[game_interface_sources.json](game_interface_sources.json).

* Ian Fellows' [Fell Omen paper](https://www.deducer.org/uploads/Main/fell_omen.pdf),
  section 1, specifies the four-round limit game, four-bet cap and doubled later
  bets. Its dealer/blind wording differs from the adapter's button convention;
  the implementation uses the explicit blind and action-order roles above.
* The [Fell Omen 2 adapter](https://github.com/gioannid/maverick-bot/blob/916315f87e78cec33369ebd404170d2cb219c7b3/maverick-backend/src/poker/FellOmen_2.java)
  declares two-player fixed-limit support. The
  [strategy implementation](https://github.com/gioannid/maverick-bot/blob/916315f87e78cec33369ebd404170d2cb219c7b3/maverick-backend/src/com/ibot/FellOmen_2_Impl.java)
  starts its bet count at one preflop and zero later, adds one per bet/raise,
  and maps button/non-button positions to the strategy tables. Its preflop
  `p1_strategy_2.txt` and `p2_strategy_7.txt` entries (four bets) have zero raise
  probability for every hand. Additional count-five branches in the adapter
  are not evidence that a fifth preflop bet belongs in the game.
* OpenSpiel's [pinned implementation](https://github.com/google-deepmind/open_spiel/blob/v1.6.3/open_spiel/games/universal_poker/universal_poker.cc)
  provides the limit action dispatch, tensor layout and payoff convention.
  Although its dispatch parameter is named `fcpa`, the limit branch has the
  complete three-action set and fixed raises. It is not a pot/all-in abstraction
  in this game. The `fullgame` dispatch is for no-limit chip-sized actions and
  is not appropriate here.

The chip scale is a project convention: 100 chips is one small bet/big blind,
200 chips is one big bet. Scaling stakes does not change the game. No stack
parameter is supplied: OpenSpiel's ACPC limit parser ignores it and uses an
effectively unlimited bankroll. The finite betting caps bound losses to 2,400.

The [opponent bridge](FELL_OMEN2.md) now validates card encoding, history mapping,
table lookup and seat rotation against the pinned Java implementation. It also
documents an upstream free-fold incompatibility: strict mode rejects such a
request, while an explicit adaptation can transfer that probability to check.
This does not change the OpenSpiel game contract.

## State and feature contract

All training actors and the central learner call the same game loader. Native
states supply `chance_outcomes()`, `current_player()`, `legal_actions()`,
`legal_actions_mask()`, `child()`/`apply_action()` and terminal `returns()`.
Action ids are meaningful only at player nodes: chance action ids are card ids.

Encoder id: `hulhe_strategic_limit_v3` (see [design review](STRATEGIC_FEATURES.md)). Inputs are float32. Policy,
regret and calibration networks receive only the acting player's information.
Critics additionally receive both players' private cards.

| Feature block | Policy length | Critic length |
|---|---:|---:|
| Hole cards, flop, turn and river, each with 52 card indicators | 208 | 260 |
| Strategic card descriptors | 90 | 180 |
| Exact current comparison (critic only) | 0 | 4 |
| Player identity | 2 | 2 |
| Native public action history: 34 slots × 2 bits | 68 | 68 |
| Street one-hot | 4 | 4 |
| Betting economics and interactions | 37 | 74 |
| Legal-action mask | 3 | 3 |
| **Total** | **412** | **595** |

Cards retain global suit canonicalisation; flop cards are unordered, while turn
and river have separate channels. Policy canonicalisation never sees the
opponent's cards. The native OpenSpiel tensor has 208 values: 2 player bits,
52 private-card bits, 52 public-card bits, 68 history bits and 34 sizing values.
In this pinned limit game the sizing values are always zero, so they are checked
and omitted. Fixed stakes and the retained history determine contributions.
No no-limit stack-size scaling remains in the encoder. Exact channels are binary; derived features are bounded in [0,1]. History includes deal slots, preserving street boundaries.

The longest legal history has 32 events (nine dealt cards and 23 decisions),
within the engine's 34-slot allocation. Tests exercise this path and the caps
on every street. Exact tensor dimensions are checked to catch library changes.
Foreign rules are rejected even if their tensor dimensions match.

The existing learning code derives action and replay dimensions from the game
and encoder. Its estimator, optimizer, sampling scheme and training budgets
are unchanged. Policy checkpoint version 2 includes the exact game contract;
continuation restores also check the game and encoder metadata. Neither the
old HUNL checkpoints nor the renamed no-limit preparation checkpoints can be
used for this game. V2 fixed-limit policies remain evaluable, but the v3
representation requires a fresh training run and rejects old replay.
