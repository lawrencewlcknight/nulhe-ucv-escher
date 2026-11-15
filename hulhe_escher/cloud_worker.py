"""VM entry point: training publisher or independent external evaluation worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .experiment import run_experiment, upload_recovery
from .train import write_json


def sync_output(path, destination, *, exclude=None):
    return subprocess.run(["gcloud", "storage", "rsync", "--recursive",
                          *(["--exclude", exclude] if exclude else []), str(path), destination],
                          timeout=1200, check=False).returncode


def fetch_file(uri, destination):
    temporary = destination.with_suffix(destination.suffix+".download")
    result = subprocess.run(["gcloud", "storage", "cp", uri, str(temporary)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    if result.returncode == 0:
        temporary.replace(destination)
        return True
    return False


def external_worker(root, uri, *, max_seconds=86400):
    import signal
    from .experiment_runtime import run_child
    root.mkdir(parents=True, exist_ok=False)
    inputs, outputs = root / "input", root / "benchmarks"
    inputs.mkdir()
    outputs.mkdir()
    deadline = time.monotonic()+max_seconds
    job = None
    try:
        setup = run_child([sys.executable, "-m", "hulhe_escher.fo2_setup"],
                          outputs / "setup.log", timeout=min(600, max_seconds))
        if setup["returncode"] != 0:
            raise RuntimeError("Fell Omen 2 asset setup failed")
        with (outputs / "fo2.log").open("x") as handle:
            job = subprocess.Popen([sys.executable, "-m", "hulhe_escher.benchmark",
                    "--experiment", str(inputs), "--output", str(outputs / "fo2"),
                    "--max-seconds", str(max(.01, deadline-time.monotonic()))],
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            while job.poll() is None and time.monotonic() < deadline:
                for label in ("P4", "P8"):
                    for suffix in ("pkl", "json"):
                        path = inputs / f"{label}.{suffix}"
                        if not path.exists():
                            try:
                                fetch_file(uri+f"/{label}.{suffix}", path)
                            except subprocess.TimeoutExpired:
                                pass
                try:
                    fetch_file(uri+"/training_failed.json", inputs / "training_failed.json")
                    sync_output(outputs, uri+"/benchmarks")
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(min(30, max(0, deadline-time.monotonic())))
    except Exception as exc:
        write_json(outputs / "worker_failure.json", {"error_type": type(exc).__name__, "reason": str(exc)})
    finally:
        if job is not None and job.poll() is None:
            os.killpg(job.pid, signal.SIGTERM)
            try:
                job.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(job.pid, signal.SIGKILL)
                job.wait()
        path = outputs / "fo2" / "report.json"
        report = json.loads(path.read_text()) if path.exists() else {"status": "INCOMPLETE_WORKER_EXIT"}
        if job is None or job.returncode != 0:
            report["status"] = "INCOMPLETE_WORKER_EXIT"
            if report.get("improvement"):
                report["improvement"].update(confirmatory=False, conclusion="incomplete_or_degenerate")
        path.parent.mkdir(exist_ok=True)
        write_json(path, report)
        sync_output(outputs, uri+"/benchmarks")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("train", "external"), required=True)
    p.add_argument("--uri", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project", required=True)
    args = p.parse_args(argv)
    if args.role == "train":
        try:
            run_experiment(args.output, upload_uri=args.uri)
        finally:
            if args.output.exists():
                # Retain latest full state, all policy exports, metrics and journals.
                try:
                    if (args.output / "training" / "latest.json").exists():
                        upload_recovery(args.output, args.uri)
                finally:
                    # Recovery pointers and endpoint markers have a transactional
                    # publication order. An unordered rsync must never bypass it.
                    sync_output(args.output, args.uri, exclude=(
                        r"^training/(?:state_[0-9]+(?:/.*)?|latest\.json|manifest\.json)$"
                        r"|^P[48]\.(?:json|pkl)$|^cloud_checkpoint\.json$|^benchmarks(?:/.*)?$"))
    else:
        external_worker(args.output, args.uri)


if __name__ == "__main__":
    main()
