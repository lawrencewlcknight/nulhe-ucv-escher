"""Readable snapshot of the primary test, diagnostics and optional benchmarks."""
import argparse
import json
from pathlib import Path

from .train import write_json


def render(root):
    root = Path(root)
    experiment = json.loads((root / "experiment.json").read_text())
    primary = experiment.get("evaluations", {}).get("primary", {})
    result = primary.get("result") or {}
    smoke = experiment.get("smoke", False)
    primary_pass = not smoke and primary.get("status") == "COMPLETE" and result.get("confirmatory", False) and result.get("conclusion") == "improved"
    rows = [("P8 vs P4", primary.get("status", "PENDING"), result)]
    external = {}
    for provider in ("fo2",):
        path = root / "benchmarks" / provider / "report.json"
        data = json.loads(path.read_text()) if path.exists() else {"status": "PENDING_OR_NOT_REQUESTED"}
        improvement = data.get("improvement") or {}
        data["supports_overall_improvement_claim"] = (primary_pass and data["status"] == "COMPLETE" and not data.get("smoke", False) and
            improvement.get("confirmatory", False) and improvement.get("conclusion") == "improved")
        external[provider] = {k: v for k, v in data.items() if k not in ("scores", "raw")}
        rows.append((provider+" P8 minus P4", data["status"], improvement))
    evidence = "P8 beats P4" if primary_pass else "Improvement not established"
    if smoke:
        evidence = "SMOKE ONLY: engineering validation, no playing-strength claim"
    elif result.get("confirmatory") and result.get("conclusion") == "regressed":
        evidence = "P8 loses to P4: statistically significant head-to-head regression"
    summary = dict(training_status=experiment["status"], primary=primary, external=external,
                   smoke=smoke,
                   training_diagnostics=experiment.get("training_diagnostics"),
                   strength_evidence=evidence)
    lines = ["# HULHE learning check", "", summary["strength_evidence"], "",
             "One training seed; results do not establish exploitability or general improvement across seeds.", "",
             "| Comparison | Status | bb/100 | Interval | Conclusion |",
             "|---|---|---:|---|---|"]
    for name, status, estimate in rows:
        score = f"{estimate['bb_per_100']:.2f}" if estimate else "—"
        ci = (f"{100*estimate['confidence']:.1f}% [{estimate['interval'][0]:.2f}, {estimate['interval'][1]:.2f}]"
              if estimate else "—")
        conclusion = estimate.get("conclusion", "—")
        if smoke or (name.startswith("fo2") and not primary_pass):
            conclusion = "descriptive only"
        lines.append(f"| {name} | {status} | {score} | {ci} | {conclusion} |")
    lines += ["", "Fell Omen 2 uses paired 95% intervals and the explicit free-fold-to-check adaptation.",
              "Overall improvement claims are gated on the primary 95% head-to-head test passing.",
              "Incomplete/time-capped samples are descriptive only; pilots are excluded from final estimates.", ""]
    for provider, data in external.items():
        for label, estimate in data.get("estimates", {}).items():
            if estimate:
                lines.append(f"- {label} against {provider}: {estimate['bb_per_100']:.2f} bb/100; "
                             f"95% interval [{estimate['interval'][0]:.2f}, {estimate['interval'][1]:.2f}] (descriptive).")
    for label, checkpoint in experiment.get("checkpoints", {}).items():
        lines.append(f"- {label}: {checkpoint['active_seconds']/3600:.3f} active hours, "
                     f"iteration {checkpoint['outer_iteration']}, SHA-256 {checkpoint['sha256']}.")
    diagnostics = experiment.get("training_diagnostics") or {}
    for label, resource in diagnostics.get("resource_measurements", {}).items():
        lines.append(f"- {label} process: {resource['seconds']/3600:.2f} wall hours; "
                     f"peak summed process RSS {resource['peak_process_tree_rss_bytes']/1024**3:.2f} GiB "
                     "(shared pages can be counted twice).")
    for name, data in [("Primary", primary), *external.items()]:
        plan = data.get("plan")
        if plan:
            lines.append(f"- {name} sample plan: {plan['selected']:,} selected, {plan['required']:,} required; "
                         f"estimated detectable effect {plan['estimated_detectable_effect_bb100']:.2f} bb/100.")
    write_json(root / "learning_report.json", summary)
    (root / "learning_report.md").write_text("\n".join(lines)+"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args(argv)
    render(args.experiment)


if __name__ == "__main__":
    main()
