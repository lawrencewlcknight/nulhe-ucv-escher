"""Pinned protected sequential continuations; never migrate the saved learner."""
import json
from pathlib import Path

from .checkpointing import sha256_file, load_checkpoint_payload
from .continuation import checked_pointer
from .corrected_continuation import fingerprint
from .sequential_training import sequential_config

SOURCE = Path(__file__).with_name('data')/'sequential_continuation_source.json'


def source_spec(baseline_hours=8):
    if baseline_hours not in (8, 32):
        raise ValueError('Sequential continuation supports pinned P8 or P32 sources')
    path = SOURCE if baseline_hours == 8 else SOURCE.with_name('sequential_p32_continuation_source.json')
    return json.loads(path.read_text())


def verify_sequential_source(source, trainer, baseline_hours=8):
    source, trainer = Path(source), Path(trainer)
    spec = source_spec(baseline_hours)
    label = f'P{baseline_hours}'
    archive = 'source.tar.gz' if baseline_hours == 8 else 'training_source.tar.gz'
    contract = json.loads((source/'training/manifest.json').read_text())
    pointer = checked_pointer(source/'training')
    marker = json.loads((source/f'{label}.json').read_text())
    if (sha256_file(source/archive) != spec['source_archive_sha256']
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
            or marker['recovery'] != pointer['state'] or marker['label'] != label
            or marker.get('smoke') or marker.get('experimental_smoke')
            or marker['active_seconds'] != spec['active_seconds']
            or marker['sha256'] != spec['policy_sha256']
            or sha256_file(source/f'{label}.pkl') != spec['policy_sha256']
            or sha256_file(source/'training'/pointer['state']/'manifest.json') != spec['state_manifest_sha256']):
        raise ValueError(f'Sequential {label} recovery or playable identity mismatch')
    fit = marker['metrics']['policy_validation']
    if (fit['group_sampling'] != 'objective_mass'
            or fit['policy_sampling_seed'] != [20261115, spec['iteration']]
            or fit.get('protected_training', {}).get('protocol') != 'protected_selective_v1'
            or fit['protected_training']['coefficient'] != .3):
        raise ValueError(f'Sequential {label} lacks corrected protected policy fitting provenance')
    payload = load_checkpoint_payload(source/f'{label}.pkl')
    if (payload['training_config'] != sequential_config() or payload['seed'] != 20261115
            or payload['outer_iteration'] != spec['iteration']
            or payload['training_elapsed_seconds'] != marker['active_seconds']
            or payload.get('policy_model', {}).get('type') != 'structured_hulhe_protected_sequential_v1'):
        raise ValueError('Sequential policy payload differs from its production contract')
    return dict(status='VERIFIED', architecture='protected_sequential_v1',
        policy_group_sampling='objective_mass', protection_coefficient=.3,
        training_source_fingerprint=spec['source_fingerprint'], source_modified=False,
        config_changed=False, baseline_iteration=spec['iteration'], pins=spec)
