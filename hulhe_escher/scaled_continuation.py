"""Immutable wide/deep P32/P56 continuation contracts; never migrate either trainer."""
import json
from pathlib import Path

from .checkpointing import sha256_file, load_checkpoint_payload
from .continuation import checked_pointer
from .corrected_continuation import fingerprint
from .scaling_config import scaling_config, deep_scaling_config

SOURCES = Path(__file__).with_name('data')/'scaled_continuation_sources.json'


def source_spec(variant, baseline_hours=32):
    if baseline_hours not in (32, 56):
        raise ValueError('Scaled source must be P32 or P56')
    path = SOURCES if baseline_hours == 32 else SOURCES.with_name('scaled_p56_continuation_sources.json')
    specs = json.loads(path.read_text())
    if variant not in specs:
        raise ValueError('Scaled continuation requires wide or deep')
    return specs[variant]


def verify_scaled_source(source, trainer, variant, baseline_hours=32):
    source, trainer = Path(source), Path(trainer)
    spec = source_spec(variant) if baseline_hours == 32 else source_spec(variant, baseline_hours)
    baseline = f'P{baseline_hours}'
    config = deep_scaling_config() if variant == 'deep' else scaling_config()
    contract = json.loads((source/'training/manifest.json').read_text())
    pointer = checked_pointer(source/'training')
    marker = json.loads((source/f'{baseline}.json').read_text())
    if (sha256_file(source/'training_source.tar.gz') != spec['source_archive_sha256']
            or fingerprint(trainer) != spec['source_fingerprint']
            or contract['source_fingerprint'] != spec['source_fingerprint']):
        raise ValueError('Scaled archived trainer identity differs from the pinned source')
    if (contract['config'] != config or contract['smoke']
            or contract['workers'] != 8 or contract['threads'] != 8
            or contract['seed'] != 20261115 or contract['policy_every'] != 0
            or contract['checkpoint_every'] != 0 or contract['recovery_seconds'] != 21600
            or contract['policy_seconds'] != 1000000000):
        raise ValueError('Scaled continuation requires the original production contract')
    if (pointer['iteration'] != spec['iteration'] or marker['outer_iteration'] != spec['iteration']
            or marker['recovery'] != pointer['state'] or marker['label'] != baseline
            or marker.get('smoke') or marker.get('experimental_smoke')
            or marker['active_seconds'] < baseline_hours*3600
            or marker['sha256'] != spec['policy_sha256']
            or sha256_file(source/f'{baseline}.pkl') != spec['policy_sha256']
            or sha256_file(source/'training'/pointer['state']/'manifest.json') != spec['state_manifest_sha256']):
        raise ValueError(f'Scaled {baseline} recovery or playable identity mismatch')
    fit = marker['metrics']['policy_validation']
    if (fit['group_sampling'] != 'objective_mass'
            or fit['policy_sampling_seed'] != [20261115, spec['iteration']]):
        raise ValueError(f'Scaled {baseline} lacks corrected policy-group sampling provenance')
    payload = load_checkpoint_payload(source/f'{baseline}.pkl')
    if (payload['training_config'] != config or payload['seed'] != 20261115
            or payload['outer_iteration'] != spec['iteration']
            or payload['training_elapsed_seconds'] != marker['active_seconds']):
        raise ValueError('Scaled policy payload differs from its production contract')
    return dict(status='VERIFIED', variant=variant, policy_group_sampling='objective_mass',
        training_source_fingerprint=spec['source_fingerprint'], source_modified=False,
        config_changed=False, baseline_iteration=spec['iteration'], pins=spec)
