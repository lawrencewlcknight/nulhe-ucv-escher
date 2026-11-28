# Wide Limit Hold’em: P104 plus 48 hours

Resume the full Wide-P104 checkpoint from
`hulhe-wide-p56-plus48-20261125-192126`, iteration **3028**, and add **48 active
training hours**. This continues the existing learner with the same archived
trainer, weights, replay, optimizers, random states, architecture and configuration.
Corrected objective-mass policy-group sampling remains enabled.

## Checkpoints and recovery

| Policy | Additional active training |
|---|---:|
| P104 | 0 hours (saved baseline) |
| P110 | 6 hours |
| P116 | 12 hours |
| P122 | 18 hours |
| P128 | 24 hours |
| P134 | 30 hours |
| P140 | 36 hours |
| P146 | 42 hours |
| P152 | 48 hours |

Targets add these durations to the actual baseline time, **374601.65730819444
seconds**; each stage finishes its current iteration. Active time excludes policy
extraction, persistence, restoration, uploads and evaluation.

Each endpoint exports a playable average policy and full recovery, verifies a fresh
process can restore it, and uploads recovery before publishing the policy marker.
All nine playable policies and the latest full recovery are retained. P152 can be
used for another continuation. Earlier full recoveries in the new run may be pruned;
the source experiments remain unchanged.

`hulhe_escher/data/scaled_p104_continuation_sources.json` pins the policy, iteration,
actual active time, original trainer archive/fingerprint and recovery manifest.
The GCS recovery metadata, original archived trainer and production configuration
were checked when creating the experiment. The cloud job checks metadata before
large replay transfers, checks recovery shards, and restores the full state before
new training. Full replay is not downloaded locally by the launcher.

The original trainer is `training_source.tar.gz`; `source.tar.gz` contains the new
coordinator. Preserve the former with the latest recovery for future continuation.

## Evaluation after training

Match the [previous 48-hour continuation](SCALED_P56_CONTINUATIONS.md):

- **Primary:** P152 versus P104, with a two-sided 95% confidence interval.
- **Eight adjacent comparisons:** each new policy against its predecessor.
- **Fell Omen 2:** all nine policies on shared seat-swapped deals, with descriptive
  absolute scores and eight paired improvements relative to P104. Use the same
  free-fold-to-check adaptation as previous runs.
- The 16 secondary comparisons use alpha **0.05/16** and **99.6875% intervals**.
  Confirmatory secondary claims require primary improvement.
- Exclude the 10,000-group pilot. Plan **100,000–1,000,000 confirmation groups**,
  targeting 80% estimated power for **2 bb/100**. Report achieved precision and any
  binding sample/time cap; inconclusive results do not imply equivalence.
- Use fresh evaluation seed offset **7,000,000**. Do not pool these hands with
  previous evaluations or select policies based on interim results.

This is one continued training seed. It does not directly compare Wide P152 with
Deep, Sequential or the historical smaller model, or measure exploitability.

## Google Batch

One **c4-standard-16**, eight collection workers, eight training threads, then eight
evaluation workers on the same VM, with a **200 GB disk**. The architecture remains
777,731 parameters per regret/policy network, with a 12-million-row policy reservoir.
Keep the existing 60-hour training/restoration/export guard, 12-hour evaluation
budget and 74-hour task ceiling including setup/transfers. These are limits.

Bounded transfer retries and ordered checkpoint publication use the established
continuation code. Automatic paid task retries are disabled. Batch releases its
VM/disk on exit; GCS retains the published policies, latest full recovery and reports.

Prepare offline, then explicitly submit:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hulhe-ucv-escher
RUN="hulhe-wide-p104-plus48-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_wide_p104_continuation_batch.py \
  --run-name "$RUN" --output "outputs/${RUN}-launch"

python3 tools/launch_wide_p104_continuation_batch.py \
  --output "outputs/${RUN}-launch" --submit-prepared
```

Alternatively add `--execute` to the preparation command. After submission, the
job runs independently of the local terminal.

```bash
gcloud batch jobs list --project=clever-overview-399515 --location=europe-west1 \
  --filter='name:hulhe-wide-p104-plus48' --format='table(name.basename(),status.state)'
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

The allowlist excludes models, replay, logs, source archives and raw hand journals.
Start with `evaluation/continuation_report.md` and `evaluation/report.json`.

## Validation

Tests cover pinned P104 identity, exact baseline time, rejection before replay
download, original-trainer routing, eight-stage scheduling, fresh evaluation seeds,
the 16-test secondary family, source protection and offline Batch packaging.
An eight-stage smoke test uses the original archived Wide trainer, production-size
network and reduced replay/fitting budgets to verify actual restores and exports
from a synthetic P104 baseline through P152 without modifying the source.
