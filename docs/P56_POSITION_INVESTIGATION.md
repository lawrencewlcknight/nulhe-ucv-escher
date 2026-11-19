# P56 position investigation and paired P32 comparison

Replicate `hulhe-positions-p32-20261118-181957` from P56, iteration 1943, then
compare strategic decisions on the original frozen P32 development library.
The source is `hulhe-p32-plus24-batch-20261118-1105`; policy SHA-256 is
`35b5348443678a2738d8ae52b1b313370739be90b1e0c298d3f73afd72adc30e`.

The replication keeps the [original protocol](POSITION_INVESTIGATION.md): 20,000
self-play hands, 20,000 hands against the pinned Fell Omen 2 bridge, 128 river
and 128 earlier-street selections, retained-replay scan, nine synthetic panels,
learned-opponent river integration, 24 upstream probes with 64 traversals per
fold, and one isolated ordinary iteration using the exact archived P56 trainer.
The learned opponent for this replication is P56. Historical JSON keys
`p32_opponent` and `regret_against_p32` remain for compatibility; the report's
`learned_opponent_label` and policy hash identify the actual opponent.

## Evidence of strategic improvement

A new encounter distribution and opponent can change the replication's gap
scores without demonstrating better decisions on the same situations. The
additional comparison therefore freezes:

- The original P32 library, hash
  `b7b2f15443047b703fcf091fce0676f7691d704d9325082a35e8847a35e8d276`.
  Only its 193 development positions are scored; 95 are river positions.
- The original P32 policy, hash
  `9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b`.
- P32 as the learned opponent, with the same history-conditioned posterior for
  both candidates, plus the same nine synthetic opponent/range panels.

Both exported policies are evaluated on identical positions. Exact remaining-river
response-gap reduction is the main descriptive comparison: the best response is
identical for both candidates, so a reduction equals improved conditional value.
Local-gap changes are also recorded, along with the effect of changing only the
current action probabilities while holding P32's continuation Q values fixed.
Baseline scores must reproduce the archived P32 analysis within numerical
tolerance before the comparison is accepted.

Reports include per-position changes, improved/regressed counts, source, seat
and family breakdowns, inverse-inclusion-weighted development-subset averages,
and unweighted averages. Earlier streets report action changes without claiming
an oracle improvement. The fixed strategic regression suite also runs both
exported policies and P56's instantaneous regret policy.

Assessment remains locked. Results are descriptive, correlated, conditional
bb/decision diagnostics, not bb/100, exploitability or a population significance
test. They concern one training seed and the combined extra training plus policy
sampler correction, without separating the contribution of either.

## Resources and results

One c4-standard-16 with a 150 GB disk, no automatic retries, four-hour emergency
limit. Original per-stage budgets remain; comparison adds a 1,800-second work
budget and 1,860-second process guard. Source recoveries are read-only, and no
resumable production checkpoint is created. The VM and disk are released by Batch.

Main outputs: `report.md`, `summary.json`, `comparison/report.md`,
`comparison/report.json`, `regression/summary.json`, and the original diagnostic
subdirectories. The launch manifest records all identities and the exact source
bundle hash. Incomplete or failed comparisons prevent overall COMPLETE status.

Launch with a fresh run name/output directory:

```bash
RUN="hulhe-positions-p56-$(date -u +%Y%m%d-%H%M%S)"

python3 tools/launch_position_investigation.py \
  --project clever-overview-399515 --region europe-west1 \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --run-name "$RUN" --output "outputs/$RUN-launch" \
  --checkpoint-label P56 --iteration 1943 \
  --source-uri gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-p32-plus24-batch-20261118-1105 \
  --expected-policy-sha256 35b5348443678a2738d8ae52b1b313370739be90b1e0c298d3f73afd72adc30e \
  --comparison-uri gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-positions-p32-20261118-181957 \
  --comparison-policy-uri gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535/training/policy_001132.pkl \
  --comparison-policy-sha256 9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b \
  --comparison-library-sha256 b7b2f15443047b703fcf091fce0676f7691d704d9325082a35e8847a35e8d276 \
  --execute
```
