# Verification

This preparation copy still executes the inherited no-limit game. These checks
validate the copied and renamed baseline; they do not validate fixed-limit rules.

Use the pinned dependencies and Python 3.10–3.12.

```bash
python -m pytest -q
RUN_RAY_TESTS=1 python -m pytest -q -m ray
python -m hulhe_escher.train --smoke --iterations 2 --output outputs/smoke
python -m hulhe_escher.train --smoke --iterations 3 --output outputs/smoke --resume
```

The suite covers the four-street no-limit game contract, pot/all-in/fold rules,
zero-sum settlements, suit symmetry, hidden-card isolation, retention of public
turn/river order, bounded monetary features, four-action unbiased estimation,
sampling support, checkpoint policy probabilities, cached/uncached learner
agreement and exact continuation through replay, optimisers and critic history.
The opt-in Ray test checks real worker collection and exact parallel continuation.

Smoke fits actual networks and writes actual checkpoints using reduced budgets.
It does not establish strategy quality, full-budget throughput, cloud deployment
readiness or integration with a fixed-limit benchmark opponent.

## Preparation-copy validation — 2026-11-14

* Sequential suite: 10 passed; the opt-in Ray test was skipped in this run.
* Separate real Ray collection/resume test: 1 passed (two local actors).
  Ray required execution outside the macOS process-inspection sandbox.
* Built the `hulhe_ucv_escher` wheel and installed it in a separate temporary
  environment. Ran the installed `hulhe-train` from outside the repository for
  two smoke iterations, then resumed to iteration three successfully.
* Checked wheel package names and bundled baseline configuration, source-file
  provenance hashes, and the manifest's explicit inherited-no-limit status.
* Compared every Python source against the source repository after namespace
  substitution: only package/CLI documentation and game-status metadata differ.
  All learner code and numerical baseline settings are preserved.

Validation used Python 3.12, NumPy 1.26.4, PyTorch 2.7.0, OpenSpiel 1.6.3 and
Ray 2.51.2. The available test environment had pytest 9.0.2; the declared dev
extra remains pytest 8.x as inherited. No full-budget training was run.
