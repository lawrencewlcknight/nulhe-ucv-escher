"""Metadata-only, independent Google Batch supervision audit."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

from .checkpointing import sha256_file
from .experiment_runtime import run_child
from .p104_cloud import storage
from .p104_protocol import SOURCE_URI, POLICY_SHA256, ITERATION, TRAINER_FINGERPRINT
from .policy_replay_artifacts import prepare, read_metadata
from .train import write_json


def prepare_inputs(root):
    source = root/'source'
    source.mkdir()
    state = source/f'state_{ITERATION:06d}'
    prepare(SOURCE_URI+f'/training/state_{ITERATION:06d}',state,metadata_only=True)
    policy = source/'P104.pkl'
    storage('cp',SOURCE_URI+'/P104.pkl',policy)
    _,metadata = read_metadata(state)
    if (sha256_file(policy) != POLICY_SHA256 or metadata['solver']['num_iteration'] != ITERATION
            or metadata['repository_commit'] != TRAINER_FINGERPRINT):
        raise ValueError('Pinned P104 identity mismatch')
    return state,policy,dict(source_uri=SOURCE_URI,source_iteration=ITERATION,
        policy_sha256=POLICY_SHA256,manifest_sha256=sha256_file(state/'manifest.json'),
        trainer_fingerprint=TRAINER_FINGERPRINT,metadata_only=True,source_read_only=True)


def run_cloud(workspace,output_uri):
    output_uri = output_uri.rstrip('/')
    if output_uri == SOURCE_URI or output_uri.startswith(SOURCE_URI+'/') or SOURCE_URI.startswith(output_uri+'/'):
        raise ValueError('Output must not overlap source')
    root = Path(workspace)
    root.mkdir(parents=True,exist_ok=False)
    output = root/'results'
    output.mkdir()
    started = time.monotonic()
    deadline = started+7*3600
    status = dict(status='PREPARING',output_uri=output_uri,source_uri=SOURCE_URI)
    stop = threading.Event()

    def stage(name):
        status.update(status=name,elapsed_seconds=time.monotonic()-started)
        write_json(output/'cloud_status.json',status)

    def sync(final=False):
        exclude = r'^cloud_status\.json$'+('' if final else r'|\.npz$|\.log$')
        storage('rsync','--recursive','--exclude',exclude,output,output_uri)
        storage('cp',output/'cloud_status.json',output_uri+'/cloud_status.json')

    def publish():
        while not stop.wait(120):
            try:
                sync()
            except Exception as exc:
                print(f'Periodic upload failed: {exc}; mandatory final upload remains.',flush=True)

    stage('PREPARING')
    publisher = threading.Thread(target=publish,daemon=True)
    publisher.start()
    try:
        state,policy,provenance = prepare_inputs(root)
        write_json(output/'provenance.json',provenance)
        budget = min(6.5*3600,deadline-time.monotonic()-120)
        if budget < 300:
            raise TimeoutError('Setup exhausted diagnostic budget')
        stage('AUDITING')
        result = run_child([sys.executable,'-m','hulhe_escher.p104_supervision','--state',str(state),
            '--policy',str(policy),'--output',str(output/'audit'),'--workers','8',
            '--max-seconds',str(budget)],output/'audit.log',timeout=budget+30)
        if result['timed_out'] or result['returncode'] != 0:
            raise RuntimeError('Audit incomplete; inspect preserved partial results')
        summary = json.loads((output/'audit/summary.json').read_text())
        if summary['status'] != 'COMPLETE' or not summary['source_preserved']:
            raise RuntimeError('Audit failed completeness/source checks')
        stage('COMPLETE')
    except BaseException as exc:
        status['reason'] = f'{type(exc).__name__}: {exc}'
        stage('FAILED')
        raise
    finally:
        stop.set()
        publisher.join(timeout=1900)
        sync(final=True)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output-uri',required=True)
    run_cloud(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
