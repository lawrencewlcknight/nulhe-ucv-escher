# Work after baseline extraction

The current repository deliberately establishes a runnable, testable starting
point. It does not claim benchmark-ready play against Slumbot or GTO Wizard.

1. **Choose the training bet menu and off-tree handling.** FCPA is a small initial
   abstraction. Add useful fractions and raise sizes with a fixed, versioned
   action-index contract, legal-size deduplication, min-raise/all-in handling and
   full public-history encoding. Real opponent amounts must affect stacks, pots
   and observations exactly. Silently snapping an opponent's bet to a training
   size is not an adequate production adapter.
2. **Port useful strategic descriptors.** Audit the selected FHP relative-strength
   and betting-interaction inputs for all four streets. Compare feature cost and
   quality; do not transplant flop-only formulas or tables as full-game equity.
3. **Profile on the intended training machine.** Establish memory, mature replay
   cost and iterations/nodes per wall-clock hour. Compare sequential and eight
   actor collection. Select checkpoint frequency and bounded retention. Verify
   installed wheel behaviour and exact continuation on the production runtime.
4. **Add evaluation.** Reusable local opponents, seat-balanced matches, independent
   evaluation RNG, per-hand payoff logs in bb, fixed evaluation sample budgets
   and confidence intervals. Random-opponent success is only an engineering check.
5. **Add Slumbot and GTO Wizard adapters.** Match their 200bb format, betting
   protocol, chip accounting and positional conventions. Separate anonymous
   integration smoke from named leaderboard runs. Add retry/backoff and durable
   hand logging. Obtain GTO Wizard API access before authenticated evaluation.
6. **Run budget comparisons.** Start from 2,500 critic steps. Test 500 against
   2,500 with fresh seeds and both matched experience and matched total compute.
   Keep the action/feature representation fixed for this comparison.
7. **Add cloud orchestration after the local contract is stable.** New job names,
   clean run identities, source pinning, resource caps and restartable storage.
   Do not re-enable old FHP jobs by renaming their labels.

External reference points:

* [OpenSpiel universal poker implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/universal_poker/universal_poker.cc)
* [Slumbot official API example](https://slumbot.com/sample_api.py)
* [GTO Wizard benchmark](https://benchmark.gtowizard.com/)
