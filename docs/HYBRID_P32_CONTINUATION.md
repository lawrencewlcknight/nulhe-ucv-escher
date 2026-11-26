# Hybrid P32 +48-hour continuation

One Google Batch job restores the completed Hybrid-P32 full recovery and adds
**48 active training hours**, reaching **Hybrid-P80**. Architecture, learning
rates, update counts, replay capacities and corrected objective-mass policy-group
sampling are preserved. The original archived trainer performs all learning.

## Source and checkpoints

Source: `hulhe-hybrid32-batch-20261124-105219`, full recovery `state_000777`.
Exact policy, trainer archive, source fingerprint and recovery-manifest SHA-256
pins are in `hulhe_escher/data/hybrid_continuation_sources.json`. Read-only cloud
checks confirmed the latest pointer and downloaded only its manifest and small
trainer archive. Full replay is downloaded by the cloud worker, not the laptop.

Before the large replay download, the worker checks policy/archive hashes,
iteration and recovery-manifest identity. It then checks the production config,
sampler provenance and trainer fingerprint, and restores the complete solver,
optimizers, replay and RNG state. Source files remain unchanged.

Retained playable policies: **P32, P38, P44, P50, P56, P62, P68, P74, P80**.
The eight targets are the actual P32 active time plus 6/12/18/24/30/36/42/48 hours.
Iteration boundaries may overshoot slightly; overshoots do not accumulate into
subsequent targets. Export, verification, upload and evaluation add wall time.

Each stage saves and reload-verifies a full recovery before ordered publication.
Existing upload retries apply. All playable policies and the latest complete
recovery remain in GCS; older full recovery directories may be pruned by existing
retention. P80 includes the model, replay, optimizer and RNG state needed for a
further continuation, alongside the preserved `training_source.tar.gz`, training
manifest and latest pointer. This is a continuation of the same training seed,
not an independent replication.

## Evaluation after training

- Primary: **Hybrid-P80 versus its own P32**, with a two-sided 95% confidence interval.
- Eight adjacent checkpoint comparisons, each policy versus its predecessor.
- Fell Omen 2: all nine policies on shared duplicate deals, with absolute scores
  and eight paired improvements relative to P32.
- The 16 secondary tests use alpha 0.05/16 (99.6875% intervals). Confirmatory
  secondary claims are gated by a successful primary test.
- An excluded 10,000-group pilot plans 100,000–1,000,000 confirmation groups,
  targeting 80% estimated power for 2 bb/100. Binding time/sample caps are reported;
  statistical significance is not guaranteed.
- Seat-swapped hands form one observation. FO2 comparisons pair policies within
  the same deal group (18 hands across nine policies).
- Fresh evaluation seed offset: 5,000,000. Training seed remains 20261115.

This job evaluates the hybrid's learning progress. Cross-architecture comparisons
remain separate experiments.

## Resources

Google Batch in `europe-west1`, project `clever-overview-399515`, using
**c4-standard-16, eight collection workers, eight training threads and 200 GB disk**.
The network remains 1,245,699 parameters per regret/average-policy network;
policy replay capacity remains 12 million rows.

Allow roughly **52–60 hours**, subject to export, storage and evaluation throughput.
Training has a 60-hour wall guard, evaluation a 12-hour budget and the whole task
a 74-hour ceiling. These are limits, not expected runtime. No automatic job retry
is enabled; verified completed stages support explicit recovery. Batch releases
the VM/disk when finished, retaining GCS artifacts.

## Prepare and launch

Prepare without submitting:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-hybrid-p32-plus48-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_hybrid_continuation_batch.py \
  --run-name "$RUN" --output "outputs/${RUN}-launch"
```

Review `launch.json` and `job.json`, then submit the prepared bundle:

```bash
python3 tools/launch_hybrid_continuation_batch.py \
  --output "outputs/${RUN}-launch" --submit-prepared
```

Alternatively add `--execute` to the preparation command using a fresh name and
output directory. The launcher snapshots local source with the existing archive
allowlist and uploads using create-only preconditions. After submission, the job
runs independently in the cloud; the terminal can close.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name~"hulhe-hybrid-p32-plus48"' \
  --format='table(name.basename(),status.state)'
```

## Download analytical results only

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="REPLACE_WITH_ACTUAL_JOB_NAME"
mkdir -p "outputs/${RUN}-results"
gcloud storage rsync --recursive \
  --exclude='^(?!(?:(?:launch|continuation|provenance|sampler-verification|P[0-9]+|reload-P[0-9]+)\.json|evaluation/(?:.*/)?(?:report\.json|plan\.json|manifest\.json|design\.json|continuation_report\.md))$).*' \
  "gs://clever-overview-399515-fhp-escher-results/hulhe/${RUN}" \
  "outputs/${RUN}-results"
```

This excludes model/replay files, logs, per-hand journals and per-iteration
metrics. Start with `evaluation/continuation_report.md` and `evaluation/report.json`.

## Local verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests/test_hybrid_continuation.py tests/test_scaled_continuation.py \
  tests/test_scaled_p56_continuation.py tests/test_continuation.py tests/test_continuation_batch.py
```

Tests check pinned identity/configuration, stage routing, output separation, Batch
bounds, fresh evaluation seeds and statistical design. A shortened eight-stage
smoke test uses full-sized hybrid networks, restores each checkpoint, checks
sampler provenance and verifies that source files remain unchanged. It uses small
replay and update counts; it establishes operability, not poker performance.
