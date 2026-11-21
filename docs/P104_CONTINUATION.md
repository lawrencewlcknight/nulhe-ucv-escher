# Limit Hold’em P104 + 24 hours

Continue `hulhe-p80-plus24-batch-20261120-171431` from its full P104 recovery,
iteration **3,489**, to P128. Restore networks, optimisers, replay and actor RNGs.
Keep the existing architecture, replay capacities, update schedules, learning
rates and training seed. This tests whether further training improves the
current specification; it does not introduce the proposed offline refitting
or architecture experiments.

The launcher pins the source URI and P104 policy SHA-256:
`7266950fc5fd4adfa37319e41bc69264e938e98c46754874c99e23483e442ad2`.
The completed source run verified `state_003489` and its playable export.
Before training, the coordinator verifies that P104 matches the latest full
recovery, checks every recovery shard, and restores the state in a fresh process.
A mismatch stops the job.

## Corrected policy grouping and sampling

P104 already uses the corrected archived trainer. Grouped average-policy targets
are sampled proportional to accumulated training weight (`objective_mass`),
with loss scaling that preserves the expected objective. The job uses
`--require-corrected-sampler`, verifies the audited trainer fingerprint
`3d66b26f5a914c192effdc0ee591b9c4314513f797422ea8c677efc3b53d6737`,
and checks that every new export reports the required sampler. It preserves
the archived learner rather than reapplying the original P32 migration.

## Training and evaluation

| Item | Specification |
| --- | --- |
| Additional training | 24 active learner hours, in four six-hour stages |
| Playable policies | P104 baseline, P110, P116, P122 and P128 |
| Primary comparison | P128 versus P104; 95% confidence interval; target 1 bb/100 at 80% estimated power; 400,000–1,000,000 confirmation duplicate pairs |
| Adjacent comparisons | P110/P104, P116/P110, P122/P116 and P128/P122 |
| Fell Omen 2 | All five policies on fresh shared paired deals; four changes versus P104 |
| Secondary comparisons | Target 2 bb/100 at 80% estimated power; 100,000–1,000,000 confirmation pairs/groups |
| Secondary intervals | 99.375%, Bonferroni across eight tests; claims gated on primary success |
| Pilots | 10,000 pairs/groups per evaluation, excluded from confirmation statistics |
| Compute | One Google Batch c4-standard-16; eight collection workers, eight training threads, 200 GB boot disk |
| Limits | 30-hour training wall budget, 12-hour evaluation budget, 44-hour task ceiling |

The previous primary comparison used 100,000 pairs. The new minimum of 400,000
would approximately halve its standard error at similar variance. Pilot variance
and measured throughput determine the final count, up to one million pairs.
Power is a planning estimate, not a guaranteed significant result. The registered
minimum is never reduced to fit the available time.

Head-to-head comparisons initially receive 20% of the evaluation budget. The
primary comparison receives four times the scheduling weight of each adjacent
comparison: initially 4,320 seconds for the primary. Unused time flows to the
remaining comparisons and then FO2. If the minimum cannot be completed, reports
identify the shortfall, results are preserved, and the Batch task fails rather
than reporting successful completion of the full experiment.

Six-hour checkpoints occur at completed iteration boundaries, using cumulative
targets relative to P104's actual active time (374,815.1484 seconds). Extraction,
persistence, uploads and evaluation add elapsed time to the 24-hour learner
budget. Comparisons run after all four training stages.

Head-to-head pilot seeds are 124301, 124311, 124321, 124331 and 124341; the FO2
pilot seed is 125501. Each confirmation seed is its pilot seed plus one. These
are fresh seed blocks relative to previous continuations.

The FO2 benchmark retains the pinned free-fold-to-check adaptation. Absolute
FO2 scores are descriptive. A completed secondary evaluation whose primary
test does not pass is labelled `descriptive_primary_gate_not_passed`; genuine
incomplete or zero-variance results retain `incomplete_or_degenerate`.
This is one training seed: intervals measure evaluation uncertainty, not
variation across independent training runs.

Every stage uploads its playable policy and full recovery transactionally.
All five policies remain available. Only the newest full recovery in the new
run is retained after successful publication of its successor; the source P104
run remains untouched. The final P128 recovery undergoes a full restore and
playable-policy equivalence check. Its `training_source.tar.gz` is retained for
future continuations. Interrupted runs can resume from their last successfully
published full state; there are no automatic whole-job retries.

## Launch

Run from the repository. Omit `--execute` to prepare a local bundle without
uploading or submitting a job. Use a fresh run name and output directory.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p104-plus24-batch-$(date -u +%Y%m%d-%H%M%S)"

python3 tools/launch_continuation_batch.py \
  --baseline-hours 104 \
  --project clever-overview-399515 \
  --region europe-west1 \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --run-name "$RUN" \
  --output "outputs/$RUN-launch" \
  --execute
```

After submission, the job runs independently in Google Batch. Storage transfers
retain the existing retry policy. Batch releases the VM and boot disk when the
job ends; uploaded results and recovery data remain in Cloud Storage.

Headline status from any terminal:

```bash
gcloud batch jobs list \
  --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment=hulhe-p104-plus24' \
  --format='table(name.basename(),status.state)'
```

## Analytical download only

Set `RUN` to the actual completed job name. This explicit list downloads summary
reports and provenance only: no models, replay shards, logs or per-hand journals.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="REPLACE_WITH_ACTUAL_JOB_NAME"
DEST="outputs/$RUN-results"
URI="gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN"
mkdir -p "$DEST/evaluation/fo2"

for FILE in launch.json continuation.json provenance.json sampler-verification.json \
  reload-P128.json P104.json P110.json P116.json P122.json P128.json \
  evaluation/report.json evaluation/design.json evaluation/continuation_report.md \
  evaluation/fo2/report.json evaluation/fo2/plan.json evaluation/fo2/manifest.json
do
  gcloud storage cp "$URI/$FILE" "$DEST/$FILE" || break
done
```

Start with `evaluation/continuation_report.md` and `evaluation/report.json`.
These analytical files cannot resume training; use the full cloud recovery and
its archived trainer, as recorded in the cloud `resume.json`.
