# Wide-P32 and Deep-P32 versus corrected smaller P32

An evaluation-only Google Batch experiment. It answers two questions directly:

1. Does Wide-P32 beat the corrected smaller P32 benchmark?
2. Does Deep-P32 beat the same benchmark?

The candidates are the final saved policies from
`hulhe-wide32-batch-20261122` (iteration 953) and
`hulhe-deep32-batch-20261122` (iteration 868).
The comparator consists of **both archived corrected P32 refits**, seeds 1701
and 1702, from `hulhe-sampler-p32-20261117-145836`. These are the same fixed
comparators used in the P56-versus-corrected-P32 experiment. Neither is selected
on its match results. This is not the legacy uncorrected P32 export or P104.

The nine input files (four policies and five small provenance reports) have
fixed SHA-256 identities in `hulhe_escher/data/scaling_corrected_sources.json`.
Candidate identities are pinned from the downloaded completed-run reports.
Hashes are checked before checkpoint deserialization. The evaluator also checks
game/encoder/action compatibility, source iterations, configuration agreement
and corrected policy-sampling receipts. Architectures intentionally differ.
There is no training, replay download, model modification or source cleanup.

## Statistical design

- Both candidates use the same fresh deals. For each deal, play each candidate
  against each corrected refit in both seats: **eight hands per deal group**.
- First average the two seats, then the two corrected fit seeds, separately for
  each candidate. Each primary test therefore has one observation per deal group.
  Correlated seats/refits are not treated as independent samples, and the refits
  are not combined into an ensemble policy.
- Exclude a 10,000-group pilot from the final estimates. It estimates variance
  and throughput, targeting **80% estimated power for 2 bb/100**.
- Use the larger sample requirement from the two tests for a common confirmation
  set: **100,000–1,000,000 groups**, or 800,000–8,000,000 confirmation hands total.
  A runtime cap can limit the achievable power and is explicitly reported.
  If the minimum does not fit, stop without a confirmatory claim.
- Two primary tests use **97.5% confidence intervals** (Bonferroni, familywise
  alpha 5%). A positive lower bound supports improvement against this comparator;
  an interval crossing zero is inconclusive, not proof of equivalence.
- Per-refit results use descriptive 95% intervals. They cannot be used to select
  a preferred comparator or replace the declared primary results.
- Production pilot/confirmation seeds are 624701/624702. Matchups have separate
  action streams while sharing cards. Smoke deals use distinct seeds.
- Fix the sample size after the pilot, without significance-based stopping.
  Partial, failed, degenerate and smoke results cannot be labelled confirmatory.

Intervals condition on these four saved policies. There is one training run per
candidate, not replicated training seeds. The smaller model's average policy was
refitted using corrected sampling after its historical training; the new larger
models used corrected sampling throughout fresh training. Replay sizes and other
settings also differ from the smaller run. These are comparisons of saved training
recipes, not an isolated test of depth or width. There is no Fell Omen 2 match in
this experiment.

## Compute

One `c4-standard-16` Batch VM, eight evaluation workers with one Torch thread each,
50 GB disk, standard provisioning and no automatic job retries. Evaluation is
limited to three hours, with a 3-hour-5-minute process guard and four-hour Batch
ceiling including setup and upload. The previous runs suggest roughly **45–90
minutes overall**, subject to the pilot's sample-size and throughput measurements.
No new model training is included in that estimate.

Input policies stay outside the published results directory. Results and raw
deal-group journals are uploaded with the existing transfer retry mechanism;
`cloud_status.json` is published last. Batch releases its VM and disk. All source
experiments and their recovery artifacts remain untouched.

## Launch

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-scaled-vs-p32c-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_scaling_corrected.py \
  --run-name "$RUN" \
  --output "outputs/${RUN}-launch" \
  --execute
```

Without `--execute`, the command only builds a reviewable local bundle. The job
uses a checksummed archive of local source; after submission it runs independently
of your terminal. Use a new run name for each submission.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name~hulhe-scaled-vs-p32c' --format='table(name.basename(),status.state)'
```

## Download analytical results only

Replace the run name below. The explicit list excludes models, logs, source
archives and raw hand/deal journals. The report includes both primary comparisons,
per-refit results, sample planning and completion status.

```bash
RUN="YOUR-SUBMITTED-JOB-NAME"
DEST="/Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher/outputs/${RUN}-results"
URI="gs://clever-overview-399515-fhp-escher-results/hulhe/${RUN}"
mkdir -p "$DEST/evaluation"
gcloud storage cp "$URI/cloud_status.json" "$URI/sources.json" "$DEST/"
gcloud storage cp \
  "$URI/evaluation/report.json" \
  "$URI/evaluation/plan.json" \
  "$URI/evaluation/manifest.json" \
  "$URI/evaluation/comparison_report.md" \
  "$DEST/evaluation/"
```

Failed jobs may lack later-stage files; read their cloud status first.
