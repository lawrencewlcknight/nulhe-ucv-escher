# Sequential specialist frozen-target checks

A bounded diagnostic before committing to 36 hours of self-play. This uses the
proposed 1,496,067-parameter network for **both player regret networks and the
average-policy network**. Production training, checkpoint loaders and policies
are unchanged. There are no new traversals, poker matches or playable exports.

## Network and fitting

Three existing 128-wide feature branches feed a 384-wide entrance. Four stages
each contain shared 384→192→384 and specialist 384→224→384 residual transforms.
LayerNorm/ReLU precede linear layers; residual scale is 0.5. The linear 768→3
head receives entrance and final representations. All specialists execute on
every input; there is no street gating. Shared weights continue learning.

For each of two seeds (240701/240702), separately for each player regret dataset
and the policy dataset, train a fresh network for 200 unrestricted warm-up
updates. Copy those exact weights into four arms with fresh identical Adam
optimizers. This initial period makes the policy preservation reference useful;
it is a short fitting warm-up, not six hours of self-play.

| Arm | Primary gradients | Protection |
|---|---|---|
| unrestricted | Every parameter | None |
| selective | Shared weights and the example street's specialist | None |
| protected_0.1 | Same as selective | Auxiliary coefficient 0.1 |
| protected_0.3 | Same as selective | Auxiliary coefficient 0.3 |

Regret protection is supervised rehearsal against **current frozen targets** on
other streets, not preservation of old regret predictions. Policy protection is
KL(reference || current) on other-street training examples. Refresh the reference
every 50 updates; it is frozen between refreshes. Coefficients are predeclared
sensitivity arms, not quantities tuned on poker results. No coefficient winner
is automatically deployed.

All fits use Adam at 0.0003, gradient norm clipping at 10, primary batch 512,
auxiliary batch 128. These bounded diagnostic settings do not change production
learning rates. Choose a street in proportion to its training objective mass,
then sample within it proportionally to mass. The primary mean loss therefore
estimates the same normalized objective as direct mass sampling on this dataset;
there is no second mass weighting. Regret rows have equal mass. All arms receive
the same primary and auxiliary draws. Homogeneous-street batches have different
variance from production mixed-street batches; this is a deliberate diagnostic.

Inactive parameters have grad=None, preventing Adam momentum updates. Gradients
still propagate THROUGH inactive specialists to earlier trainable layers.
The auxiliary loss can update the selected specialist using other-street inputs;
it does not unfreeze the other specialists. Shared updates can affect all streets.

## Immutable saved inputs and scope

- P104 source iteration 3489 from `hulhe-p80-plus24-batch-20261120-171431`.
- Both all-row regret captures (36,699 and 39,792 rows) from iteration 3490 in
  `hulhe-p104-regret-20261122-073047`.
- Only the four average-policy replay shards, metadata and source policy are
  fetched, plus four small captured-target/prediction arrays and their report.
  No critic replay or full resumable state is downloaded.
- Manifest, policy, trainer identity and capture arrays are pinned and verified.
  Preparation verifies metadata before downloading policy replay.

Policy replay is grouped exactly over the full saved reservoir first. Then at
most **4,096 groups per street** are retained using a fixed random seed independent
of targets. All preflop groups are retained if fewer than the cap. Original targets
and relative mass weights within each street are preserved. We rescale retained
weights within each street to match its full-replay total mass: subsampling later
streets must not accidentally make preflop dominate almost every update. The report
records retained/full group counts, mass and calibration factors. This restricted objective is NOT
an unbiased full-replay performance estimate or a production policy refit.

Each dataset uses the same deterministic exact-information-state 60/20/20
training/validation/audit split in every arm and seed. Duplicate states cannot
cross partitions. All partitions must contain all four streets. Related boards
or correlated trajectories can still cross partitions; this is not an independent
game-distribution generalization claim.

Run 300 additional updates per arm, validating at 0/50/150/300. Update zero is
eligible and ties favour earlier updates. Select the fitting budget separately
for each arm/seed using validation target error only. Write all selections before
computing any audit errors. Both seeds and every arm are reported.

## Reports and interpretation

- Policy: objective-weighted KL (target entropy removed), action total variation,
  overall and by street, for training/validation/audit partitions.
- Regret: production-style masked row MSE over three action outputs; additionally
  within-state variation and sampled-mean prediction error by street, with legal
  action denominators explicitly inherited from the existing decomposition.
- Regret-derived action TV is relative to regret matching on sampled targets,
  not an oracle policy. Captured targets are not assumed correct.
- Per-arm five-update, single-street interventions from the same warm start,
  measuring a 4×4 validation error-change matrix. Positive change means damage.
  Five steps matter because KL preservation has zero gradient at its reference.
- Parameter/gradient routing checks, fitting seconds, update throughput,
  source identities, partition counts/hashes and selected budgets.

Lower training error alone is insufficient. Look for validation/audit improvement
in both seeds, acceptable later-street error, reduced off-street damage and
reasonable overhead relative to unrestricted fitting. Short fits can underfit:
an inconclusive result does not establish a low capacity ceiling. These checks
cannot predict self-play convergence or establish stronger poker play.

## Google Batch

One `c4-standard-16`, eight Torch threads, 100 GB disk in `europe-west1`.
No collection workers or automatic task retries. Fitting/data checks have a
90-minute process budget; the coordinator is guarded at 165 minutes and Batch
at three hours including provisioning, transfer and cleanup. Those are safety
limits, not runtime predictions. Failed/incomplete checks cannot report COMPLETE.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-sequential-frozen-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_sequential_frozen.py \
  --run-name "$RUN" --output "outputs/${RUN}-launch" --execute
```

Omit `--execute` to build a local bundle only. After submission the job runs
independently in Google Batch. Source data remains read-only. Final results are
uploaded; Batch releases its VM/disk. A hard kill can prevent final publication.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name~hulhe-sequential-frozen' --format='table(name.basename(),status.state)'
```

Download **four analytical files only**, after setting RUN to the submitted name:

```bash
DEST="/Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher/outputs/${RUN}-results"
URI="gs://clever-overview-399515-fhp-escher-results/hulhe/${RUN}"
mkdir -p "$DEST/diagnostic"
gcloud storage cp "$URI/cloud_status.json" "$DEST/"
gcloud storage cp "$URI/diagnostic/report.json" "$URI/diagnostic/report.md" \
  "$URI/diagnostic/selection.json" "$DEST/diagnostic/"
```

For already-prepared local inputs:

```bash
python -m hulhe_escher.sequential_frozen --state /path/to/p104-policy-subset \
  --policy /path/to/P104.pkl --captures /path/to/saved-capture \
  --output outputs/sequential-frozen-local --threads 8 --max-seconds 5400
```

`--smoke` is only for synthetic fixtures; all report outputs identify smoke mode.
Tests exercise real full-size models, frozen Adam parameters, gradient propagation,
masking, objective sampling, split isolation, source preservation and Batch bounds.
