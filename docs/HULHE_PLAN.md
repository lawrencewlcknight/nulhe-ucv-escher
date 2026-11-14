# Heads-up limit Hold'em development plan

The repository is a renamed, runnable copy of the HUNL baseline. Its current
engine still uses no-limit FCPA rules. Complete these steps iteratively before
claiming fixed-limit training or benchmark performance.

1. **Define and implement the fixed-limit game contract.** Specify blinds,
   preflop/flop small bets, turn/river big bets, per-street betting caps,
   heads-up cap conventions, starting stacks and payoff units. Match the chosen
   opponent's exact rules. Replace pot-sized raises and all-ins with the legal
   fixed-limit action set. Verify all four streets and zero-sum settlement.
2. **Adapt and version the feature/action contract.** Inspect OpenSpiel's limit
   information tensors and history bounds. Update legal masks, monetary scaling,
   policy/critic output dimensions, replay and checkpoint metadata together.
   Preserve hidden-card isolation, suit symmetry and public street ordering.
3. **Validate the inherited learner.** Keep the initial 2,500-update critic budget,
   estimator and fitting procedure fixed while checking unbiased action estimates,
   sampling support, cached targets and exact sequential/parallel continuation in
   the new game. Then iterate on algorithm improvements with controlled comparisons.
4. **Integrate a fixed-limit opponent.** Evaluate candidate opponents such as
   Fell Omen 2 for availability, licensing, execution and matching rules before
   selecting a benchmark. Log seat-balanced duplicate deals, seeds, payoffs in
   explicitly named units and uncertainty. No opponent is bundled in this copy.
5. **Profile before scaling.** Measure collection, encoding, fitting, persistence,
   memory and total runtime. Test richer features or smaller critic budgets
   against strategy quality per compute. Cloud training remains a later step.

Slumbot and the previously considered GTO Wizard no-limit interfaces are not
our target benchmarks for this fixed-limit project.
