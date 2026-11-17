"""Run the frozen-replay sampler experiment and publish results on one CPU VM."""
import argparse
from pathlib import Path
import threading

from .checkpointing import sha256_file
from .fo2_setup import prepare as prepare_fo2
from .policy_replay_artifacts import prepare
from .policy_sampling_experiment import run_fit
from .policy_sampling_evaluate import run_evaluation
from .storage_transfer import run_transfer
from .train import write_json


def run_cloud(workspace, source_uri, output_uri, expected_policy_sha256, *, iteration=1132):
    source_uri, output_uri = source_uri.rstrip("/"), output_uri.rstrip("/")
    if (source_uri == output_uri or output_uri.startswith(source_uri + "/")
            or source_uri.startswith(output_uri + "/")):
        raise ValueError("Use a separate cloud output prefix")
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    output = root / "results"
    output.mkdir()
    state = root / "policy-replay"
    state.mkdir()
    policy = state / f"policy_{iteration:06d}.pkl"
    status = dict(status="STARTING", source_uri=source_uri, output_uri=output_uri,
                  expected_policy_sha256=expected_policy_sha256, iteration=iteration)
    stop = threading.Event()

    def stage(name):
        status["status"] = name
        write_json(output / "cloud_status.json", status)

    def sync(final=False):
        exclude = r"^cloud_status\.json$" + (r"|\.jsonl$" if not final else "")
        run_transfer(["gcloud", "storage", "rsync", "--recursive", "--exclude", exclude,
                      str(output), output_uri], timeout=1200)
        # Publish the status after the results it describes.
        run_transfer(["gcloud", "storage", "cp", str(output / "cloud_status.json"),
                      output_uri + "/cloud_status.json"], timeout=300)

    def publish_loop():
        while not stop.wait(120):
            try:
                sync()
            except Exception as exc:
                print(f"Periodic upload failed; final upload remains mandatory: {exc}", flush=True)

    stage("PREPARING")
    publisher = threading.Thread(target=publish_loop, daemon=True)
    publisher.start()
    try:
        run_transfer(["gcloud", "storage", "cp",
            f"{source_uri}/training/policy_{iteration:06d}.pkl", str(policy)], timeout=300)
        if sha256_file(policy) != expected_policy_sha256:
            raise ValueError("Source policy SHA-256 differs from the requested baseline")
        # Check opponent availability before spending time fitting policies.
        assets = root / "fo2-assets"
        build = prepare_fo2(assets)
        status["fo2_fingerprint"] = build["fingerprint"]
        prepare(f"{source_uri}/training/state_{iteration:06d}", state)
        stage("FITTING")
        run_fit(state, policy, output / "fit", seeds=(1701, 1702), threads=8)
        stage("EVALUATING")
        report = run_evaluation(output / "fit", output / "evaluation", assets=assets,
                                workers=8, max_seconds=7200)
        stage("COMPLETE" if report["status"] == "COMPLETE" else "EVALUATION_" + report["status"])
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        stage("FAILED")
        raise
    finally:
        stop.set()
        publisher.join()  # Finish any bounded in-flight transfer before final publication.
        sync(final=True)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--source-uri", required=True)
    p.add_argument("--output-uri", required=True)
    p.add_argument("--expected-policy-sha256", required=True)
    p.add_argument("--iteration", type=int, default=1132)
    result = run_cloud(**vars(p.parse_args()))
    if result["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
