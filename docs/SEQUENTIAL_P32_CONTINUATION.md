# Protected sequential Limit Hold’em: P32 plus 24 hours

Continue `hulhe-sequential-p8-plus24-20261125-054522` from its full P32 checkpoint
(iteration 659). Train for 24 additional active hours, preserving the original
archived trainer, model weights, replay, optimizers, random states and configuration.
The protected sequential architecture remains 1,496,067 parameters per regret/policy
network, with protection coefficient 0.3 and corrected objective-mass policy sampling.

| Policy | Additional active training | Nominal cumulative training |
|---|---:|---:|
| P32 | 0 hours | 32 hours |
| P38 | 6 hours | 38 hours |
| P44 | 12 hours | 44 hours |
| P50 | 18 hours | 50 hours |
| P56 | 24 hours | 56 hours |

The saved baseline has 115,582.36501615302 active seconds; targets add 6/12/18/24
hours to that actual time. Stages finish their current iteration. Active training
excludes policy extraction, persistence, restoration, uploads and evaluation.
Each stage exports a playable policy and full recovery, verifies restoration in
a fresh process, and uploads recovery before publishing the policy marker. All
five playable policies and the latest full recovery are retained. P56 can seed
another continuation. The original P8 and P32 runs remain read-only.

## Source verification

`hulhe_escher/data/sequential_p32_continuation_source.json` pins the P32 policy,
original trainer archive/fingerprint, recovery manifest, iteration and active time.
GCS metadata was checked when preparing the experiment. The full replay is downloaded
only on the Batch VM. Incorrect archive or recovery metadata is rejected before
large replay downloads; shard checksums and a full restore are required before training.

The source run's **training_source.tar.gz** contains the unchanged original trainer.
Its `source.tar.gz` contains its coordinator and must not be used as the trainer.
The new run republishes the original trainer for future continuations.

## Evaluation

Use the same protocol as the [P8 continuation](SEQUENTIAL_P8_CONTINUATION.md):

- Primary: **P56 versus P32**, 95% confidence interval.
- Adjacent comparisons: P38/P32, P44/P38, P50/P44 and P56/P50.
- Fell Omen 2: all five policies on shared duplicate deals with seats swapped,
  using the existing free-fold-to-check adaptation. Report absolute scores and
  paired changes from P32.
- The four adjacent comparisons and four FO2 changes form one secondary family,
  with alpha 0.05/8 and confirmatory claims gated on primary success.

Use fresh evaluation seeds (offset 6,000,000). Each match has 10,000 excluded pilot
groups and 100,000–1,000,000 confirmation groups, planned for 80% estimated power
at 2 bb/100. Report achieved precision when sample or time caps limit power.
Absolute FO2 scores are descriptive. This is one training seed and does not test
differences against the smaller, wide, deep or hybrid architectures.

## Google Batch

One **c4-standard-16**, 200 GB disk, eight collection workers/eight fitting threads,
then eight evaluation workers on the same VM. Allow 30 hours for training,
restoration and exports, 12 hours for evaluation, and a 44-hour task ceiling
including setup/transfers. These are limits, not a runtime prediction. Transfer
retries and ordered publication are inherited from the existing continuation.
Automatic paid job retries are disabled. Batch releases its VM and disk on exit;
published artifacts remain in GCS.

Prepare offline, then explicitly submit:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-sequential-p32-plus24-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_sequential_p32_continuation_batch.py \
  --run-name "$RUN" --output "outputs/cloud-$RUN"

python3 tools/launch_sequential_p32_continuation_batch.py \
  --output "outputs/cloud-$RUN" --submit-prepared
```

Alternatively, use `--execute` on the preparation command. Preparation alone
does not create cloud resources. Submission claims a unique output prefix.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name:hulhe-sequential-p32-plus24' \
  --format='table(name.basename(),status.state)'
```

Download analytical results only after completion:

```bash
RUN=hulhe-sequential-p32-plus24-YYYYMMDD-HHMMSS
mkdir -p "outputs/$RUN-results"
gcloud storage rsync --recursive \
  --exclude='(^|/)(training|original-run|archived-trainer|fo2-assets|state_[0-9]+)(/|$)|^(?!.*\.(json|md)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN" \
  "outputs/$RUN-results"
```

Read `continuation.json`, `evaluation/report.json`,
`evaluation/continuation_report.md`, `provenance.json`, checkpoint markers and
`reload-P*.json`. Model files, replay, logs and archives are excluded.

## Validation

The sequential continuation tests cover source identity and configuration guards,
rejection before replay download, Batch settings, fresh evaluation seeds and
multiplicity. The real archived-trainer smoke test exercises four stages from a
small synthetic P32 baseline, checks full restores and source preservation, and
uses the production network with reduced buffers and fitting budgets.
