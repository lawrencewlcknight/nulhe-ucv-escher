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

The pinned Java opponent bridge and duplicate-deal match runner are implemented
and lightweight-tested. See [FELL_OMEN2.md](FELL_OMEN2.md). Resolve the benchmark
policy convention for upstream free folds before interpreting longer results:
strict rejection is the default; an explicit check conversion is labelled as an
adapted opponent.

1. **Establish the evaluation campaign.** Choose fixed evaluation seeds, sample
   size and the declared opponent variant. Retain pair-level uncertainty and
   keep smoke tests separate from strength measurements.
2. **Iterate on learning.** Compare algorithm changes against the preserved learner
   and Fell Omen 2 with controlled training/evaluation budgets. Keep benchmark
   evaluation separate from self-play training unless intentionally changing the
   learning objective.
3. **Profile before scaling.** Measure collection, encoding, fitting, persistence,
   memory and total runtime. Test richer features or smaller critic budgets
   against strategy quality per compute. Cloud training remains a later step.
