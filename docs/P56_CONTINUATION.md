# Limit Hold’em P56 + 24 hours

Continue the completed `hulhe-p32-plus24-batch-20261118-1105` experiment from
its full P56 recovery, iteration **1,943**, to P80. This is one training seed
with the same networks, replay capacities, update schedules and learning rates.
No replay refitting or new architecture is introduced.

The archived trainer already implements **objective-mass sampling of grouped
average-policy targets**, with the corresponding loss scaling. The coordinator
verifies its complete source fingerprint before training and checks that every
new policy export reports `group_sampling: objective_mass`. It does not apply
the original P32 migration again or change the source identity of the recovery.

The baseline policy is pinned to SHA-256
`35b5348443678a2738d8ae52b1b313370739be90b1e0c298d3f73afd72adc30e`.
Every recovery shard is checksum-verified, and a fresh process restores the
solver, optimizers, replay and actor RNGs and checks the playable export before
training. The same full restore check runs after P80 is saved.

## Training and evaluation

| Item | Specification |
| --- | --- |
| Additional training | 24 active learner hours, in four six-hour stages |
| Exports | P62, P68, P74 and P80; P56 retained as the baseline |
| Primary evaluation | P80 versus P56, 95% confidence interval |
| Adjacent evaluations | P62/P56, P68/P62, P74/P68 and P80/P74 |
| Fell Omen 2 | All five policies on shared paired deals; four improvements versus P56 |
| Secondary intervals | 99.375%, Bonferroni across eight tests, gated on primary success |
| Sample planning | Excluded 10,000-pair/group pilot; target 2 bb/100 at 80% estimated power; 100,000–1,000,000 confirmation pairs/groups |
| Compute | One Google Batch c4-standard-16, eight collection workers, eight training threads, 200 GB boot disk |
| Limits | 30-hour training wall budget, 12-hour evaluation budget, 44-hour task ceiling |

Checkpoints are taken at completed iteration boundaries, using cumulative
targets relative to P56’s actual active time. Policy extraction, persistence,
uploads and evaluation do not consume the 24-hour learner budget. Evaluation
runs after all four training stages. Fresh deterministic seeds are distinct
from the P32→P56 evaluation. The FO2 benchmark retains the existing pinned
free-fold-to-check adaptation. Absolute FO2 scores are descriptive; capped or
incomplete evaluations are labelled explicitly.

Each boundary uploads its playable policy and full recovery transactionally.
All five playable policies are retained. As in the previous experiment, only
the latest full recovery is retained in the new run after successful publication
of its successor. The source P56 run remains untouched. P80 includes everything
needed for another continuation, plus its exact `training_source.tar.gz`.

## Launch

From the repository, run the following. Omit `--execute` to create and inspect
the local bundle without uploading or submitting a job. Use a fresh run name
and output directory for each invocation; an existing bundle is never overwritten.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-p56-plus24-batch-$(date -u +%Y%m%d-%H%M%S)"

python3 tools/launch_continuation_batch.py \
  --baseline-hours 56 \
  --project clever-overview-399515 \
  --region europe-west1 \
  --bucket gs://clever-overview-399515-fhp-escher-results/hulhe \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --run-name "$RUN" \
  --output "outputs/$RUN-launch" \
  --execute
```

The launcher pins the known P56 source URI and policy hash. The job executes
independently in Google Batch after submission. It has no automatic whole-job
retries; storage transfers retain their existing retry policy. Batch releases
the VM and disk on completion or failure; cloud results remain available.

Headline status, from any terminal:

```bash
gcloud batch jobs list \
  --project=clever-overview-399515 --location=europe-west1 \
  --filter='labels.experiment=hulhe-p56-plus24' \
  --format='table(name.basename(),status.state)'
```

After completion, set `RUN` to the submitted job name to download analytical
results without model/replay payloads or per-hand journals:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
mkdir -p "outputs/$RUN-results"
gcloud storage rsync --recursive \
  --exclude='^(?!.*\.(json|md|html|csv|svg|png)$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/$RUN" \
  "outputs/$RUN-results"
```

Read `evaluation/continuation_report.md`, `evaluation/report.json`,
`continuation.json`, `sampler-verification.json`, and `reload-P80.json`.
Analytical downloads alone cannot resume training: use the full cloud recovery
and archived trainer described by `resume.json`.
