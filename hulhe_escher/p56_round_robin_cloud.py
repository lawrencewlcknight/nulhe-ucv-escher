"""Download only pinned playable policies and metadata; run a bounded comparison."""
import argparse
import json
from pathlib import Path
import sys

from .p56_round_robin import SOURCES, verify_inputs
from .checkpointing import sha256_file
from .experiment_runtime import run_child
from .storage_transfer import run_transfer
from .train import write_json


def storage(*args):
    return run_transfer(['gcloud', 'storage', *map(str, args)], timeout=600)


def validate_output(uri):
    uri = uri.rstrip('/')
    sources = json.loads(SOURCES.read_text())
    parents = {r['uri'].split('/fit/')[0] if '/fit/' in r['uri'] else r['uri'].rsplit('/', 1)[0]
               for r in sources['files'].values()}
    if any(uri == p or uri.startswith(p+'/') or p.startswith(uri+'/') for p in parents):
        raise ValueError('Output overlaps a source experiment')
    return uri


def run_cloud(workspace, output_uri):
    output_uri = validate_output(output_uri)
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    inputs, output = root/'inputs', root/'results'
    inputs.mkdir(); output.mkdir()
    status = dict(status='PREPARING', collection=False, retraining=False, replay_download=False)
    try:
        sources = json.loads(SOURCES.read_text())
        write_json(output/'sources.json', sources)
        for name, record in sources['files'].items():
            storage('cp', record['uri'], inputs/name)
            if sha256_file(inputs/name) != record['sha256']:
                raise ValueError(f'Input checksum mismatch: {name}')
        verify_inputs(inputs)
        status['status'] = 'EVALUATING'
        write_json(output/'cloud_status.json', status)
        result = run_child([sys.executable, '-m', 'hulhe_escher.p56_round_robin',
            '--inputs', str(inputs), '--output', str(output/'evaluation'), '--workers', '8',
            '--max-seconds', '10800'], output/'evaluation.log', timeout=11100)
        if result['timed_out']:
            raise TimeoutError('Evaluation process guard reached')
        report = json.loads((output/'evaluation/report.json').read_text())
        if result['returncode'] != 0 or report['status'] != 'COMPLETE':
            raise RuntimeError(f"Evaluation did not complete: {report['status']}")
        status['status'] = 'COMPLETE'
    except BaseException as exc:
        status.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Publish status last so COMPLETE is not visible before its results arrive.
        final_status = dict(status)
        write_json(output/'cloud_status.json', dict(status, status='UPLOADING'))
        storage('rsync', '--recursive', output, output_uri)
        write_json(output/'cloud_status.json', final_status)
        storage('cp', output/'cloud_status.json', output_uri+'/cloud_status.json')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output-uri', required=True)
    run_cloud(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
