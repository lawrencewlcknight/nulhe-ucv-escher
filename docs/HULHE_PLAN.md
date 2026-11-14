# Heads-up limit Hold'em development plan

Fell Omen 2 is the selected challenge benchmark.

## Completed: game interface

OpenSpiel 1.6.3 supplies the fixed-limit four-street game. The encoder, legal masks,
policy/critic output dimensions, replay and checkpoint contracts now use three
actions. See [GAME_INTERFACE.md](GAME_INTERFACE.md) for the exact rules and sources.
Rules tests cover betting caps, blind option, check-raises, settlement, information
boundaries and incompatible checkpoints. The inherited learning budgets remain
unchanged, including 2,500 updates per critic fit.

## Next iterations

1. **Connect Fell Omen 2.** Build a reproducible opponent adapter from the pinned
   source and strategy tables. Match the existing game contract, player/seat
   conventions, card ids, action events and history indexing. Check strategy
   lookup coverage and legal probabilities before any performance claim. No
   opponent executable or strategy tables are bundled in this repository yet.
2. **Build the match harness.** Log seat-balanced duplicate deals, independent
   evaluation RNG, seeds, raw chip payoffs, small bets per hand and uncertainty.
   Distinguish a big blind (100 chips) from a big bet (200 chips). Validate the
   adapter with scripted trajectories before running learned-policy matches.
3. **Iterate on learning.** Compare algorithm changes against the preserved learner
   and Fell Omen 2 with controlled training/evaluation budgets. Keep benchmark
   evaluation separate from self-play training unless intentionally changing the
   learning objective.
4. **Profile before scaling.** Measure collection, encoding, fitting, persistence,
   memory and total runtime. Test richer features or smaller critic budgets
   against strategy quality per compute. Cloud training remains a later step.
