"""Download pinned saved policies and metadata for the Sequential P32 comparison."""
import argparse
import json
from pathlib import Path
import re
import sys

from .sequential_p32_comparison import SOURCES, verify_inputs
from .checkpointing import sha256_file
from .experiment_runtime import run_child
from .storage_transfer import run_transfer
from .train import write_json


def storage(*args):
    return run_transfer(['gcloud', 'storage', *map(str, args)], timeout=600)


def validate_output(uri):
    uri = uri.rstrip('/')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?', uri):
        raise ValueError('Invalid output bucket/prefix')
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
    inputs.mkdir()
    output.mkdir()
    status = dict(status='PREPARING', collection=False, retraining=False,
                  replay_download=False, fo2=False)
    try:
        sources = json.loads(SOURCES.read_text())
        write_json(output/'sources.json', sources)
        for name, record in sources['files'].items():
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError(f'Invalid pinned input filename: {name}')
            storage('cp', record['uri'], inputs/name)
            if sha256_file(inputs/name) != record['sha256']:
                raise ValueError(f'Input checksum mismatch: {name}')
        verify_inputs(inputs)
        status['status'] = 'EVALUATING'
        write_json(output/'cloud_status.json', status)
        result = run_child([sys.executable, '-m', 'hulhe_escher.sequential_p32_comparison',
            '--inputs', str(inputs), '--output', str(output/'evaluation'), '--workers', '8',
            '--max-seconds', '21600'], output/'evaluation.log', timeout=21900)
        status['evaluation_process'] = result
        if result['timed_out']:
            raise TimeoutError('Evaluation process guard reached')
        report_path = output/'evaluation/report.json'
        if not report_path.is_file():
            raise RuntimeError('Evaluation did not complete: report.json is missing')
        report = json.loads(report_path.read_text())
        if result['returncode'] != 0 or report['status'] != 'COMPLETE':
            raise RuntimeError(f"Evaluation did not complete: {report['status']}")
        verify_inputs(inputs)
        status['status'] = 'COMPLETE'
    except BaseException as exc:
        status.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Inputs are a sibling of results and never enter the upload tree.
        # Publish final status only after every result has arrived successfully.
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
