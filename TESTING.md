# Verification

Use the pinned dependencies and Python 3.10–3.12.

```bash
python -m pytest -q
RUN_RAY_TESTS=1 python -m pytest -q -m ray
python -m hunl_escher.train --smoke --iterations 2 --output outputs/smoke
python -m hunl_escher.train --smoke --iterations 3 --output outputs/smoke --resume
```

The suite covers the four-street no-limit game contract, pot/all-in/fold rules,
zero-sum settlements, suit symmetry, hidden-card isolation, retention of public
turn/river order, bounded monetary features, four-action unbiased estimation,
sampling support, checkpoint policy probabilities, cached/uncached learner
agreement and exact continuation through replay, optimisers and critic history.
The opt-in Ray test checks real worker collection and exact parallel continuation.

Smoke fits actual networks and writes actual checkpoints using reduced budgets.
It does not establish strategy quality, full-budget throughput, cloud deployment
readiness or integration with either external benchmark API.
