"""Audited P32 migration: change only average-policy group sampling.

The archived learner remains authoritative. Never bypass ordinary resume
checks: verify the old checkpoint first, then rebind its source identity to
an explicitly patched archive, retaining every model/replay/RNG value.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

import torch

from .checkpointing import sha256_file
from .continuation import checked_pointer
from .experiment_runtime import run_child
from .train import write_json

P32_SOURCE = "fa4f727456f265d52a75f7f06c9e53fdd3d65a3da6640fa5c141e9cad851dca1"
CORRECTED_SOURCE = "3d66b26f5a914c192effdc0ee591b9c4314513f797422ea8c677efc3b53d6737"
LEGACY_FITTING_SHA256 = "37b716e89eeeac823223e9d430945767ebbe9df0541611d983e96293114985c3"
PACKAGES = ("hulhe_escher", "unbiased_escher", "adaptive_escher", "vr_deep_cfr")


def fingerprint(root):
    digest = hashlib.sha256()
    for package in PACKAGES:
        for path in sorted((Path(root) / package).rglob("*")):
            if path.suffix in (".py", ".json"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_corrected_source(source, trainer):
    """Require the audited production sampler without patching a second time."""
    source = Path(source)
    contract = json.loads((source / "training/manifest.json").read_text())
    if contract["source_fingerprint"] != CORRECTED_SOURCE or fingerprint(trainer) != CORRECTED_SOURCE:
        raise ValueError("Archived trainer is not the audited objective_mass trainer")
    if contract["smoke"] or contract["workers"] != 8 or contract["threads"] != 8 or contract["seed"] != 20261115:
        raise ValueError("Corrected continuation requires the original production contract")
    return dict(status="VERIFIED", policy_group_sampling="objective_mass",
                training_source_fingerprint=CORRECTED_SOURCE, source_modified=False)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Archived fitting implementation differs from migration recipe")
    return text.replace(before, after, 1)


def patch_trainer(trainer, seed):
    """Two-file, checksum-guarded patch; no unrelated current learner changes."""
    trainer = Path(trainer)
    fitting = trainer / "unbiased_escher/holdem_fitting.py"
    if sha256_file(fitting) != LEGACY_FITTING_SHA256:
        raise ValueError("Unexpected legacy policy fitter checksum")
    sampler = trainer / "unbiased_escher/policy_sampling.py"
    if sampler.exists():
        raise ValueError("Archived trainer already contains a policy sampler")
    text = fitting.read_text()
    # Appending the import is legal: module execution finishes before any fit.
    text += "\nfrom .policy_sampling import PolicyGroupSampler\n"
    text = replace_once(text,
        '    history, best, best_score = [], None, float("inf")\n    trainer.model.train()',
        '    history, best, best_score = [], None, float("inf")\n'
        '    sampler = PolicyGroupSampler(train_ids, weights, batch_size,\n'
        f'        mode="objective_mass", seed=[{int(seed)}, int(iteration)])\n'
        '    trainer.model.train()')
    text = replace_once(text,
        '        selected = (train_ids if batch_size == len(train_ids) else\n'
        '                    train_ids[random.sample(range(len(train_ids)), batch_size)])\n'
        '        idx = torch.as_tensor(selected, device=features.device)',
        '        idx, loss_weights = sampler.sample()')
    text = replace_once(text, '        loss = (losses * weights[idx]).mean()',
                        '        loss = (losses * loss_weights).mean()')
    text = replace_once(text, '        "validation_history": history,',
        '        "validation_history": history,\n'
        '        "group_sampling": sampler.mode,\n'
        f'        "policy_sampling_seed": [{int(seed)}, int(iteration)],')
    fitting.write_text(text)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "unbiased_escher/policy_sampling.py", sampler)
    return {str(p.relative_to(trainer)): sha256_file(p) for p in (fitting, sampler)}


def rebind_metadata(source, destination, new_fingerprint):
    """Copy a verified checkpoint, changing only its source identity metadata.

Replay shards are hard-linked locally and never edited; metadata is replaced
atomically, so the input remains byte-for-byte intact. Full verification by
the patched learner follows before collection starts.
"""
    source, destination = Path(source), Path(destination)
    shutil.copytree(source, destination, copy_function=os.link)
    contract = json.loads((source / "training/manifest.json").read_text())
    pointer = checked_pointer(source / "training")
    state = destination / "training" / pointer["state"]
    manifest = json.loads((state / "manifest.json").read_text())
    metadata = state / manifest["metadata_path"]
    payload = torch.load(metadata, map_location="cpu", weights_only=False)
    if (payload["repository_commit"] != contract["source_fingerprint"] or
            payload["config"] != contract["config"] or payload["seed"] != contract["seed"]):
        raise ValueError("Saved state identity differs from verified contract")
    payload["repository_commit"] = new_fingerprint
    temporary = metadata.with_suffix(".migration.tmp")
    torch.save(payload, temporary)
    temporary.replace(metadata)
    old_size = manifest["metadata_size_bytes"]
    manifest.update(metadata_sha256=sha256_file(metadata), metadata_size_bytes=metadata.stat().st_size)
    manifest["total_size_bytes"] += manifest["metadata_size_bytes"] - old_size
    write_json(state / "manifest.json", manifest)
    contract["source_fingerprint"] = new_fingerprint
    write_json(destination / "training/manifest.json", contract)


def prepare_corrected_source(source, trainer, root):
    source, trainer, root = map(Path, (source, trainer, root))
    contract = json.loads((source / "training/manifest.json").read_text())
    if contract["source_fingerprint"] != P32_SOURCE or fingerprint(trainer) != P32_SOURCE:
        raise ValueError("This migration is restricted to the audited original P32 trainer")
    if (contract["smoke"] or contract["workers"] != 8 or contract["threads"] != 8 or
            checked_pointer(source / "training")["iteration"] != 1132):
        raise ValueError("This migration requires the production P32 recovery")
    verifier = Path(__file__).with_name("verify_recovery.py")
    result = run_child([sys.executable, str(verifier), "--source", str(trainer),
        "--training", str(source / "training"), "--policy", str(source / "P32.pkl"),
        "--report", str(root / "original-P32-verified.json")],
        root / "original-P32-verify.log", timeout=1800, cwd=trainer,
        env={**os.environ, "PYTHONPATH": str(trainer), "PYTHONNOUSERSITE": "1"})
    if result["returncode"]:
        raise RuntimeError("Original P32 restore failed; migration refused")
    patched = root / "corrected-trainer"
    shutil.copytree(trainer, patched)
    files = patch_trainer(patched, contract["seed"])
    new_fingerprint = fingerprint(patched)
    migrated = root / "corrected-source"
    rebind_metadata(source, migrated, new_fingerprint)
    # The old archive is hard-linked: replace, never overwrite, that file.
    archive = migrated / "training_source.tar.gz"
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as handle:
        for path in sorted(patched.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".java", ".toml", ".md"):
                handle.add(path, arcname=str(path.relative_to(patched)), recursive=False)
    temporary.replace(archive)
    receipt = dict(migration="p32_objective_mass_v1", original_contract=contract,
        original_source_fingerprint=P32_SOURCE, training_source_fingerprint=new_fingerprint,
        changed_source_files=files, policy_group_sampling="objective_mass",
        sampling_seed="[training_seed, outer_iteration]", original_policy_sha256=sha256_file(source / "P32.pkl"),
        metadata_changes=["repository_commit"], replay_shards_changed=False,
        config_changed=False, original_restore_verified=True,
        training_source_archive_sha256=sha256_file(archive))
    write_json(root / "sampler-migration.json", receipt)
    return migrated, patched
