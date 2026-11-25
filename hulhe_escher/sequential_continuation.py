"""Pinned protected sequential P8 continuation; never migrate the saved learner."""
import json
from pathlib import Path

from .checkpointing import sha256_file, load_checkpoint_payload
from .continuation import checked_pointer
from .corrected_continuation import fingerprint
from .sequential_training import sequential_config

SOURCE = Path(__file__).with_name('data')/'sequential_continuation_source.json'


def source_spec():
    return json.loads(SOURCE.read_text())


def verify_sequential_source(source, trainer):
    source, trainer = Path(source), Path(trainer)
    spec = source_spec()
    contract = json.loads((source/'training/manifest.json').read_text())
    pointer = checked_pointer(source/'training')
    marker = json.loads((source/'P8.json').read_text())
    if (sha256_file(source/'source.tar.gz') != spec['source_archive_sha256']
            or fingerprint(trainer) != spec['source_fingerprint']
            or contract['source_fingerprint'] != spec['source_fingerprint']):
        raise ValueError('Sequential archived trainer identity differs from the pinned source')
    if (contract['config'] != sequential_config() or contract['smoke']
            or contract['workers'] != 8 or contract['threads'] != 8
            or contract['seed'] != 20261115 or contract['policy_every'] != 0
            or contract['checkpoint_every'] != 0 or contract['recovery_seconds'] != 21600
            or contract['policy_seconds'] != 1000000000):
        raise ValueError('Sequential continuation requires the original production contract')
    if (pointer['iteration'] != spec['iteration'] or marker['outer_iteration'] != spec['iteration']
            or marker['recovery'] != pointer['state'] or marker['label'] != 'P8'
            or marker.get('smoke') or marker.get('experimental_smoke')
            or marker['active_seconds'] != spec['active_seconds']
            or marker['sha256'] != spec['policy_sha256']
            or sha256_file(source/'P8.pkl') != spec['policy_sha256']
            or sha256_file(source/'training'/pointer['state']/'manifest.json') != spec['state_manifest_sha256']):
        raise ValueError('Sequential P8 recovery or playable identity mismatch')
    fit = marker['metrics']['policy_validation']
    if (fit['group_sampling'] != 'objective_mass'
            or fit['policy_sampling_seed'] != [20261115, spec['iteration']]
            or fit.get('protected_training', {}).get('protocol') != 'protected_selective_v1'
            or fit['protected_training']['coefficient'] != .3):
        raise ValueError('Sequential P8 lacks corrected protected policy fitting provenance')
    payload = load_checkpoint_payload(source/'P8.pkl')
    if (payload['training_config'] != sequential_config() or payload['seed'] != 20261115
            or payload['outer_iteration'] != spec['iteration']
            or payload['training_elapsed_seconds'] != marker['active_seconds']
            or payload.get('policy_model', {}).get('type') != 'structured_hulhe_protected_sequential_v1'):
        raise ValueError('Sequential policy payload differs from its production contract')
    return dict(status='VERIFIED', architecture='protected_sequential_v1',
        policy_group_sampling='objective_mass', protection_coefficient=.3,
        training_source_fingerprint=spec['source_fingerprint'], source_modified=False,
        config_changed=False, baseline_iteration=spec['iteration'], pins=spec)
