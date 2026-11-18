# P32 + 24 hours with corrected policy sampling

`tools/launch_continuation_batch.py` prepares a Google Cloud Batch job. Pass
`--execute` to upload its immutable source bundle and submit it. The launcher
uses the same bounded-task, Cloud Logging and GCS output pattern as the FHP
and Leduc jobs. No local process needs to remain running after submission.

The job restores P32 (iteration 1132) from the successful original run
`hulhe-plus24-20261116-063535`, then trains another 24 **active learner hours**.
It exports P38, P44, P50 and P56 at six-hour boundaries. Saving, uploading and
average-policy fitting are outside the active-time budget. The original P32
policy remains the comparator, unchanged.

The average-policy fitter samples grouped replay entries in proportion to
their accumulated objective weight. Each sampled loss receives the training
partition's mean weight, preserving the original expected weighted objective.
The sampler uses a private RNG seeded by `[training_seed, outer_iteration]`;
it does not change collection RNG state. Validation membership, checkpoint
selection, architecture, learning rates, update budgets and buffers are unchanged.

The production learner comes from the **original archived trainer**, not the
current working tree. `corrected_continuation.py` checks the original source
fingerprint and fitting-file checksum, performs a full original-state reload,
then changes only the archived fitter and adds the tested sampler module.
An isolated checkpoint copy changes only `repository_commit` and its checksum
manifest; replay shards, models, optimizers, worker RNGs and the configuration
remain intact. The new source fingerprint is recorded in the training contract.
Ordinary strict resume checks then verify this corrected recovery before use.
The job publishes `sampler-migration.json`, `original-P32-verified.json` and
the corrected `training_source.tar.gz`, so future continuations keep the fix.

Resources: c4-standard-16, 8 collection workers, 8 fitting threads, 200 GB
Hyperdisk Balanced. The container pins Python 3.11.2 and uses the saved library
versions. A pinned Batch Container-Optimized OS image avoids reusing the Debian
kernel that crashed in the failed standalone VM. Task maximum: 44 wall hours;
training coordinator: 30 hours; evaluation: 12 hours. Automatic task retries
are disabled to avoid silently repeating an expensive run. Full six-hour
recoveries are uploaded transactionally; the latest recovery is retained.

Evaluation remains the existing P32+24 protocol: primary P56 versus P32;
four adjacent-policy comparisons; paired Fell Omen 2 improvement comparisons
against P32. The primary uses alpha 0.05 and the eight secondary comparisons
use alpha 0.05/8. Pilot-based sample sizing and evaluation wall limits remain.

Batch releases its VM and boot disk after the task ends. Results and the latest
full recovery remain in the job's separate GCS prefix. Failed standalone-run
cleanup must never remove the original P32 source run or shared infrastructure.

Example preparation (omit `--execute` to inspect only):

```bash
python3 tools/launch_continuation_batch.py \
  --run-name hulhe-p32-plus24-batch-YYYYMMDD-HHMMSS \
  --output outputs/cloud-hulhe-p32-plus24-batch-YYYYMMDD-HHMMSS \
  --execute
```

Status from any terminal:

```bash
gcloud batch jobs list --project=clever-overview-399515 \
  --location=europe-west1 --filter='labels.experiment=hulhe-p32-plus24' \
  --format='table(name.basename(),status.state)'
```
