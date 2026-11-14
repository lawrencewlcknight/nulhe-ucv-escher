"""Fetch checksum-pinned FO2 assets and compile the local Java bridge."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

PACKAGE = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE / 'data/fell_omen2_assets.json'
DEFAULT_ASSETS = Path('.cache/fell_omen2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def bridge_fingerprint():
    h = hashlib.sha256(MANIFEST_PATH.read_bytes())
    h.update(Path(__file__).read_bytes())
    for p in sorted((PACKAGE / 'java').glob('*.java')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def verify_assets(root):
    root = Path(root)
    for entry in asset_manifest()['files']:
        path = root / entry['path']
        if not path.is_file() or digest(path) != entry['sha256']:
            raise ValueError(f'Missing or corrupt FO2 asset: {path}; run python -m hulhe_escher.fo2_setup')


def verify_build(root):
    root = Path(root)
    verify_assets(root)
    marker = root / 'build.json'
    if not marker.exists():
        raise ValueError('FO2 bridge is not built; run python -m hulhe_escher.fo2_setup')
    build = json.loads(marker.read_text())
    if build['fingerprint'] != bridge_fingerprint():
        raise ValueError('FO2 bridge sources changed; rebuild with hulhe_escher.fo2_setup')
    classes = {str(p.relative_to(root)): digest(p) for p in (root / 'classes').rglob('*.class')}
    if not classes or classes != build['classes']:
        raise ValueError('FO2 compiled classes differ from their build manifest; rebuild')
    return build


def prepare(root=DEFAULT_ASSETS, *, offline=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for entry in asset_manifest()['files']:
        path = root / entry['path']
        if path.exists():
            if digest(path) != entry['sha256']:
                raise ValueError(f'Corrupt existing asset: {path}; remove it explicitly before fetching again')
            continue
        if offline:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.download')
        try:
            with urllib.request.urlopen(entry['url'], timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out)
            if digest(temporary) != entry['sha256']:
                raise ValueError(f'Hash mismatch downloading {entry["url"]}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    verify_assets(root)
    if not shutil.which('javac') or not shutil.which('java'):
        raise RuntimeError('Install a JDK (Java 11 or later) to build and run the FO2 bridge')
    classes = root / 'classes'
    classes.mkdir(exist_ok=True)
    (root / 'no-native-libraries').mkdir(exist_ok=True)
    # Preserve the original adapter's terminal-history mapping verbatim for an
    # independent Java replay check of the Python observation translator.
    adapter = (root / 'reference/FellOmen_2.java').read_text()
    start = adapter.index('public int nodeToTerminalNodeIndex(int node)')
    end = adapter.index('{', start) + 1
    depth = 1
    while depth:
        depth += (adapter[end] == '{') - (adapter[end] == '}')
        end += 1
    generated = root / 'generated/ReferenceHistory.java'
    generated.parent.mkdir(exist_ok=True)
    generated.write_text('package com.ibot;\npublic class ReferenceHistory extends FellOmen_2_Impl {\n'
                         + adapter[start:end] + '\n@Override public void debug(String s) {}\n}\n')
    sources = sorted((root / 'src').rglob('*.java')) + sorted((PACKAGE / 'java').glob('*.java'))
    sources.append(generated)
    subprocess.run(['javac', '--release', '11', '-encoding', 'UTF-8', '-cp', str(root / 'meerkat.jar'),
                    '-d', str(classes), *map(str, sources)], check=True)
    build = {'fingerprint': bridge_fingerprint(), 'upstream_commit': asset_manifest()['commit'],
             'javac': subprocess.check_output(['javac', '-version'], stderr=subprocess.STDOUT, text=True).strip(),
             'classes': {str(p.relative_to(root)): digest(p) for p in classes.rglob('*.class')}}
    (root / 'build.json').write_text(json.dumps(build, indent=2) + '\n')
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--offline', action='store_true', help='Only use already cached, verified downloads')
    args = parser.parse_args()
    build = prepare(args.assets, offline=args.offline)
    print(json.dumps({'assets': str(args.assets.resolve()), 'fingerprint': build['fingerprint']}))


if __name__ == '__main__':
    main()
