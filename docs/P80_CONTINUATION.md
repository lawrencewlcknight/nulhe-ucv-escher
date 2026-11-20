# Limit Hold’em P80 + 24 hours

Continue `hulhe-p56-plus24-batch-20261119-144002` from its full P80 recovery,
iteration **2,711**, to P104. Restore networks, optimisers, replay and actor RNGs.
Keep the existing architecture, replay capacities, update schedules, learning
rates and training seed. This is a continuation, not a policy-only warm start.

The launcher pins the source URI and P80 policy SHA-256:
`822e1d499ce2688f362e46595220b341dbff6adf8546e9ca65a2e459f4020e8c`.
The completed source run verified `state_002711` and its playable export. At
launch the coordinator checks that P80 still matches the latest recovery,
verifies every recovery shard, and restores the complete state in a fresh
process before any training. A mismatch stops the job.

## Corrected policy grouping and sampling

P80 already uses the corrected archived trainer. Grouped average-policy targets
are sampled proportional to accumulated training weight (`objective_mass`),
with the corresponding loss scaling preserving the expected objective. The job
uses `--require-corrected-sampler`, verifies the audited trainer fingerprint
`3d66b26f5a914c192effdc0ee591b9c4314513f797422ea8c677efc3b53d6737`,
and checks that every new export reports the required sampler. It does not
reapply the original P32 migration or substitute the current repository's
learner for the archived trainer.

## Training and evaluation

| Item | Specification |
| --- | --- |
| Additional training | 24 active learner hours, in four six-hour stages |
| Playable policies | P80 baseline, P86, P92, P98 and P104 |
| Primary comparison | P104 versus P80, 95% confidence interval |
| Adjacent comparisons | P86/P80, P92/P86, P98/P92 and P104/P98 |
| Fell Omen 2 | All five policies on fresh shared paired deals; four changes versus P80 |
| Secondary intervals | 99.375%, Bonferroni across eight tests, gated on primary success |
| Sample planning | Excluded 10,000-pair/group pilot; target 2 bb/100 at 80% estimated power; 100,000–1,000,000 confirmation pairs/groups |
| Compute | One Google Batch c4-standard-16; eight collection workers, eight training threads, 200 GB boot disk |
| Limits | 30-hour training wall budget, 12-hour evaluation budget, 44-hour task ceiling |

Six-hour checkpoints occur at completed iteration boundaries, using cumulative
targets relative to P80's actual active time. Policy extraction, persistence,
uploads and evaluation add elapsed time to the 24-hour learner budget.
Comparisons run after all four training stages. Head-to-head pilot seeds are
100301, 100311, 100321, 100331 and 100341; the FO2 pilot seed is 101501.
Each confirmation seed is its pilot seed plus one. These differ from previous
continuations. Pilot deals are excluded from confirmation statistics.

The FO2 benchmark retains the pinned free-fold-to-check adaptation. Absolute
FO2 scores are descriptive. Reports identify capped or incomplete evaluations.
This is one training seed; confidence intervals measure evaluation uncertainty,
not variation across independent training runs.

Every stage uploads its playable policy and full recovery transactionally.
All five policies remain available. Only the newest full recovery in the new
run is retained after successful publication of its successor; the source P80
run remains untouched. The final P104 recovery undergoes a full restore and
playable-policy equivalence check. Its `training_source.tar.gz` is retained for
future continuations. Interrupted runs can be recovered from their last
successfully published full state; there are no automatic whole-job retries.

## Launch

Run from the repository. Omit `--execute` for a local bundle only, with no cloud
submission or upload. Always use a fresh run name and output directory.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p80-plus24-batch-$(date -u +%Y%m%d-%H%M%S)"

python3 tools/launch_continuation_batch.py \
  --baseline-hours 80 \
  --project clever-overview-399515 \
  --region europe-west1 \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --run-name "$RUN" \
  --output "outputs/$RUN-launch" \
  --execute
```

After submission the job runs independently in Google Batch. Storage transfers
retain the existing retry policy. Batch releases the VM and boot disk when the
job ends; uploaded results and recovery data remain in Cloud Storage.

Headline status from any terminal:

```bash
gcloud batch jobs list \
  --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment=hulhe-p80-plus24' \
  --format='table(name.basename(),status.state)'
```

After completion, set `RUN` to the actual job name, then download analytical
results without policy files, replay shards or per-hand journals:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
mkdir -p "outputs/$RUN-results"
gcloud storage rsync --recursive \
  --exclude='^(?!.*\.(json|md|html|csv|svg|png)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN" \
  "outputs/$RUN-results"
```

Read `evaluation/continuation_report.md`, `evaluation/report.json`,
`continuation.json`, `sampler-verification.json` and `reload-P104.json`.
Analytical downloads cannot resume training; the full cloud recovery and its
archived trainer are required, as recorded in `resume.json`.
