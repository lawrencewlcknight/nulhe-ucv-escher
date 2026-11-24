"""Read-only input preparation and bounded Batch execution for sequential checks."""
import argparse
import json
from pathlib import Path
import sys
import time

from .checkpointing import sha256_file
from .experiment_runtime import run_child
from .p104_cloud import storage
from .policy_replay_artifacts import prepare, read_metadata
from . import sequential_protocol as protocol
from .train import write_json


def reject_overlap(uri):
    uri = uri.rstrip('/')
    for source in (protocol.SOURCE_URI, protocol.CAPTURE_URI.rsplit('/', 1)[0]):
        if uri == source or uri.startswith(source+'/') or source.startswith(uri+'/'):
            raise ValueError('Output overlaps a read-only source')


def prepare_inputs(root):
    source = root/'source'
    source.mkdir()
    state = source/f'state_{protocol.ITERATION:06d}'
    uri = protocol.SOURCE_URI+f'/training/state_{protocol.ITERATION:06d}'
    prepare(uri, state, metadata_only=True)
    _, metadata = read_metadata(state)
    if (sha256_file(state/'manifest.json') != protocol.MANIFEST_SHA256
            or metadata['solver']['num_iteration'] != protocol.ITERATION
            or metadata['repository_commit'] != protocol.TRAINER_FINGERPRINT):
        raise ValueError('Pinned P104 metadata mismatch before replay download')
    policy = source/'P104.pkl'
    storage('cp', protocol.SOURCE_URI+'/P104.pkl', policy)
    if sha256_file(policy) != protocol.POLICY_SHA256:
        raise ValueError('Pinned policy checksum mismatch')
    captures = source/'captures'
    captures.mkdir()
    storage('cp', protocol.CAPTURE_URI+'/report.json', captures/'report.json')
    for name, digest in protocol.CAPTURE_HASHES.items():
        storage('cp', protocol.CAPTURE_URI+'/'+name, captures/name)
        if sha256_file(captures/name) != digest:
            raise ValueError('Pinned regret capture checksum mismatch')
    prepare(uri, state)  # Only the four policy replay shards, not full recovery.
    return state, policy, captures


def run_cloud(workspace, output_uri):
    reject_overlap(output_uri)
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    output = root/'results'
    output.mkdir()
    started = time.monotonic()
    status = dict(status='PREPARING', protocol=protocol.PROTOCOL, source_read_only=True,
                  collection_iterations=0, policy_uri=protocol.SOURCE_URI, capture_uri=protocol.CAPTURE_URI)
    try:
        write_json(output/'cloud_status.json', status)
        storage('cp', output/'cloud_status.json', output_uri+'/cloud_status.json')
        state, policy, captures = prepare_inputs(root)
        status['status'] = 'FITTING'
        write_json(output/'cloud_status.json', status)
        storage('cp', output/'cloud_status.json', output_uri+'/cloud_status.json')
        outcome = run_child([sys.executable, '-m', 'hulhe_escher.sequential_frozen',
            '--state', str(state), '--policy', str(policy), '--captures', str(captures),
            '--output', str(output/'diagnostic'), '--threads', '8', '--max-seconds', '5400'],
            output/'diagnostic.log', timeout=5460)
        if outcome['timed_out'] or outcome['returncode'] != 0:
            raise RuntimeError('Frozen checks failed or exceeded the 90-minute diagnostic guard')
        report = json.loads((output/'diagnostic/report.json').read_text())
        if report['status'] != 'COMPLETE':
            raise RuntimeError('Incomplete frozen checks')
        status['status'] = 'COMPLETE'
    except BaseException as exc:
        status.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic()-started
        write_json(output/'cloud_status.json', status)
        # Upload analytical reports and this small diagnostic log only, never source data.
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'cloud_status.json':
                storage('cp', path, output_uri+'/'+str(path.relative_to(output)))
        storage('cp', output/'cloud_status.json', output_uri+'/cloud_status.json')
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output-uri', required=True)
    run_cloud(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
