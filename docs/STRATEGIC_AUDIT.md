# Limit Hold’em strategic-position evaluation

This suite lives in `hulhe_escher/strategic_audit/`, alongside training and the
Fell Omen 2 match evaluator. It adapts the scientific distinctions in the FHP
strategic-position audit to the existing OpenSpiel Limit Hold’em contract.
It needs playable policies, not a new training run or a replay download.

## What is evaluated

The versioned built-in library contains **270 root positions**:

| Partition | Preflop | Flop | Turn | River | Total |
|---|---:|---:|---:|---:|---:|
| Development | 4 | 4 | 4 | 168 | 180 |
| Locked assessment | 2 | 2 | 2 | 84 | 90 |

River coverage includes 15 distinct public-board suit orbits: ten development,
five assessment. It includes private nuts, board-playing ties, strong and weak
kickers, missed flush/straight draws, four-flush and connected boards, paired
boards, full houses, and quads on the board. Both seats are tested at open,
facing-bet and capped decisions, with smaller/larger pots and different earlier
betting histories. These are selected diagnostic positions, not all Hold’em
information sets. Some related examples intentionally share a board.

Each river position integrates **all 990 compatible opponent hole-card pairs**,
with card removal, exact showdown rankings and every remaining legal river
betting branch. Values are in **big blinds (100 chips) per audited decision**.
They include sunk contributions; differences between values are decision costs.
They are not whole-game winnings in bb/100.

Three measurements remain separate:

1. **Provable dominated folds.** When calling closes the hand and the hero cannot
   lose against any compatible opponent cards, folding is strictly worse than
   calling. Report fold probability and a guaranteed lower bound on the expected
   cost of those folds. This is independent of the stipulated opponent range.
2. **Local decision gap.** Best root action value minus the value of the policy’s
   root action distribution, keeping its own later river decisions fixed.
3. **Remaining-river response gap.** Best information-respecting continuation
   against the frozen response model minus the policy’s deployed continuation.
   Hidden hands are integrated **before** optimizing each hero decision. The
   response cannot choose a different action after secretly observing the
   opponent’s cards.

The root gaps are aggregated separately. Do not sum gaps at overlapping decisions
along a trajectory. The journal includes up to three large continuation errors
per root/panel for diagnosis. A poor result against one response model is not
proof of an equilibrium error or an exploitability estimate.

For terminal calls, the report includes showdown equity (ties count one half),
call cost, pot size and the threshold `call_cost / (pot + call_cost)`. Where
raising is legal, exceeding this threshold only establishes that calling beats
folding; it does not establish that calling beats raising. Calling an always-losing
hand against the stipulated range is labelled **range-conditioned**, separately
from universal unbeatable-hand errors. An empty custom range is unsupported,
never silently replaced with a uniform one.

## Opponent and range assumptions

The default panel has three **synthetic response models** from the audited
position onward, each crossed with three stipulated root ranges:

- `check_call`: always checks or calls.
- `value`: usually folds low-percentile holdings facing a bet, and raises/bets
  stronger holdings more frequently.
- `pressure`: a mixed aggressive response with a nonzero bluff component.

The rank percentile used by the latter two models is computed over all 1,081
board-compatible private hands. It does not condition the opponent’s decisions
on the hero’s hidden cards. Exact action probabilities are versioned in
`engine.py` and hashed in the run manifest.

The root ranges are `uniform`, `strong` and `weak`: weights are proportional to
`1`, `exp(4*(percentile-0.5))` and `exp(-4*(percentile-0.5))`, respectively, after
hero-card removal. They are **stipulated conditional ranges at the root**, not
posteriors fitted from earlier betting. A root history may be counterfactual for
that response model; the model specifies behaviour *from this decision onward*.
Inside the remaining river tree, opponent action likelihoods update the range.
These agents are not Fell Omen 2, equilibrium strategies or trained exploiters.
Keep the existing paired FO2/head-to-head evaluations for playing-strength claims.

Earlier-street positions report legal action probabilities and consistency
checks only. They have **no invented correct action or exact action-value score**.
Preflop premium hands, flop draws and turn made/missed draws are useful for
checkpoint comparisons, but future betting and opponent ranges prevent a simple
universal answer key. Full preflop/flop/turn solving is outside this suite.

## Consistency, partitions and provenance

Every root tests two alternative compatible opponent hands, reversed hero hole
card order, a repeated query, and all 24 suit permutations. Privacy, hole-order,
invalid probabilities and unstable distributions fail the run. Suit differences
are reported, or fail when `--require-suit-invariance` is supplied. These finite
checks are regression tests, not a proof that an arbitrary policy can never leak
hidden information. Future board cards are never dealt into earlier-street roots.

Development and assessment keep entire public-board suit orbits separate; preflop
uses starting-hand orbits. Assessment requires `--partition assessment
--unlock-assessment` and a new output directory. Freeze candidate/model selection
before unlocking. This is a holdout from audit-driven selection, **not a claim
that the board was absent from self-play training**. Repeatedly tuning to its
results consumes the assessment holdout. No confidence intervals are fabricated
from the correlated positions, seats, panel variants or suit permutations.

Each run records the complete library and its hash, position ids, policy hashes,
source hashes, library versions, thread count and budget. Native checkpoints are
trusted pickle/Torch inputs. No credentials or external poker API are needed.
Output directories cannot be overwritten, and models/source checkpoints are not
updated. There is no automatic resume or cloud submission.

## Run without a checkpoint

Use the repository’s existing Python environment with the package installed.
The baseline-only smoke checks execution, not trained playing strength:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
python -m hulhe_escher.strategic_audit run \
  --baseline uniform --baseline check_call --baseline raise \
  --smoke --threads 1 --max-seconds 120 \
  --output outputs/strategic-smoke
```

The smoke uses six river roots and one context root per earlier street, within
the selected partition. Without `--smoke`, all positions in the partition run.

## Compare playable checkpoints

The following P32 policy already exists locally:

```bash
python -m hulhe_escher.strategic_audit run \
  --policy P32=outputs/hulhe-plus24-20261116-063535/training/policy_001132.pkl \
  --baseline check_call --baseline uniform \
  --require-suit-invariance --threads 1 --max-seconds 600 \
  --output outputs/strategic-p32-development
```

Add another `--policy Candidate=/absolute/path/to/candidate.pkl` to evaluate it
on identical positions and panels. Existing structured policies and the larger
residual diagnostic policy use their native checkpoint loader.

An optional `--regret-state Instantaneous=/absolute/path/to/state_001132` adds
the instantaneous regret-matching strategy. Only the verified metadata/manifest
are read; the average-policy replay shards are unnecessary for this adapter.
The current structured regret architecture is supported. This helps distinguish
policy approximation errors from upstream strategy errors. The instantaneous
strategy and historical average policy need not agree, and agreement alone is
not an answer key.

After selection is frozen, use a **new directory** and append:

```bash
  --partition assessment --unlock-assessment
```

To freeze or extend a library, first create its JSON:

```bash
python -m hulhe_escher.strategic_audit library \
  --output outputs/strategic-library-v1.json
```

Then pass `--library outputs/strategic-library-v1.json` to runs. Position entries
contain only hero cards, visible board, seat, partition, tags and per-street
OpenSpiel action histories (`0=fold`, `1=check/call`, `2=bet/raise`). The builder
rejects duplicate cards, illegal histories, wrong actors and partition leakage.
A custom library is a different benchmark: freeze it before comparing candidates.

## Outputs and runtime

- `report.html`, `report.md`, `gaps.svg`: compact comparison and costly provable errors.
- `summary.json`: root metrics by policy, response model/range, seat and scenario family.
- `positions.jsonl`: visible cards/history, actions, exact river values, range-dependent
  responses, pot odds and bounded worst continuation decisions; context-only rows
  contain no fabricated score.
- `checks.json`: privacy, hole-order and suit consistency.
- `manifest.json`, `library.json`: provenance and reproducible input contract.

Read `summary.json` status before interpreting results. Failures are marked
`FAILED`; budget exhaustion is `INCOMPLETE`, with partial journals retained.
The default 600-second guard is cooperative between inference/tree operations,
not a hard process kill; initial checkpoint loading is outside that budget.
Native inference calls are finite, but an arbitrary third-party policy can block
inside a call. No GPU, training VM or full recovery download is required.

The initial full development run of P32 plus uniform and check/call baselines
completed locally in roughly four seconds using one Torch thread. This is a
measurement for these policies and this fixed suite, not a cloud runtime promise.
The assessment partition was not evaluated on P32.

Validation includes all 990 OpenSpiel terminal payoffs for eight hero/board cases,
independent scalar-world evaluation of full river trees, a regression against
clairvoyant maximization, partition/privacy tests, malformed-policy failures,
checkpoint immutability and timeout reporting.
