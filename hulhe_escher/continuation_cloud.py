"""Download an existing full recovery, continue, then evaluate; never retrain the baseline."""
import argparse
import json
from pathlib import Path
import tarfile
import threading

from .checkpointing import sha256_file
from .continuation import checked_pointer, checkpoint_labels, run_continuation
from .continuation_evaluate import run_suite
from .learning_evaluate import validate_design
from .fo2_setup import prepare
from .train import write_json
from .storage_transfer import run_transfer


def storage(*args):
    run_transfer(['gcloud','storage',*map(str,args)],timeout=1800)


def extract_source(archive, destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as handle:
        for member in handle.getmembers():
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile():
                raise ValueError('Source archive must contain only safe regular files')
            target=destination/path
            target.parent.mkdir(parents=True,exist_ok=True)
            with handle.extractfile(member) as source, target.open('xb') as out:
                import shutil
                shutil.copyfileobj(source,out)


def download_source(uri, workspace, expected_p8_sha256, baseline_hours=8, *,
                    expected_archive_sha256=None, expected_iteration=None,
                    expected_state_manifest_sha256=None):
    baseline = checkpoint_labels(baseline_hours)[0]
    archive = "source.tar.gz" if baseline_hours == 8 else "training_source.tar.gz"
    root=Path(workspace)
    root.mkdir(parents=True,exist_ok=False)
    source=root/'original-run'
    (source/'training').mkdir(parents=True)
    for name in (f'{baseline}.json',f'{baseline}.pkl','training/manifest.json','training/latest.json',archive):
        storage('cp',uri.rstrip('/')+'/'+name,source/name)
    if sha256_file(source/f'{baseline}.pkl') != expected_p8_sha256:
        raise ValueError('Downloaded policy does not match requested baseline')
    if expected_archive_sha256 and sha256_file(source/archive) != expected_archive_sha256:
        raise ValueError('Archived trainer checksum mismatch')
    pointer=checked_pointer(source/'training')
    marker=json.loads((source/f'{baseline}.json').read_text())
    if marker['label'] != baseline or marker['recovery'] != pointer['state'] or marker['outer_iteration'] != pointer['iteration']:
        raise ValueError('Requested baseline is not the latest full recovery')
    if expected_iteration is not None and pointer['iteration'] != expected_iteration:
        raise ValueError('Requested baseline recovery iteration differs from pin')
    state=source/'training'/pointer['state']
    state.mkdir()
    if expected_state_manifest_sha256:
        storage('cp', uri.rstrip('/')+'/training/'+pointer['state']+'/manifest.json', state/'manifest.json')
        if sha256_file(state/'manifest.json') != expected_state_manifest_sha256:
            raise ValueError('Pinned recovery manifest mismatch before replay download')
    storage('rsync','--recursive',uri.rstrip('/')+'/training/'+pointer['state'],state)
    if expected_state_manifest_sha256 and sha256_file(state/'manifest.json') != expected_state_manifest_sha256:
        raise ValueError('Pinned recovery manifest changed during download')
    trainer=root/'archived-trainer'
    extract_source(source/archive,trainer)
    return source,trainer


def run_cloud(workspace, source_uri, output_uri, expected_p8_sha256, *,
              resume_checkpoint=None, expected_resume_sha256=None,
              training_wall_seconds=108000, baseline_hours=8, corrected_sampler=False,
              require_corrected_sampler=False, primary_minimum=100000, primary_effect_bb100=2.,
              scaled_variant=None, sequential_source=False, stage_count=4):
    validate_design(0, 8, 10000, primary_minimum, 1000000, 43200, primary_effect_bb100)
    baseline = checkpoint_labels(baseline_hours, stage_count)[0]
    archive = "source.tar.gz" if baseline_hours == 8 else "training_source.tar.gz"
    # Avoid overwriting or pruning any part of the original experiment.
    source_uri,output_uri=source_uri.rstrip('/'),output_uri.rstrip('/')
    if source_uri==output_uri or output_uri.startswith(source_uri+'/') or source_uri.startswith(output_uri+'/'):
        raise ValueError('Continuation must use a separate sibling cloud prefix')
    root=Path(workspace)
    if corrected_sampler and (baseline_hours != 32 or resume_checkpoint):
        raise ValueError('Sampler migration requires a fresh P32 continuation')
    if corrected_sampler and require_corrected_sampler:
        raise ValueError('Select migration or verification of an already corrected trainer')
    pinned = None
    if scaled_variant is not None:
        from .scaled_continuation import source_spec
        pinned = source_spec(scaled_variant) if baseline_hours == 32 else source_spec(scaled_variant, baseline_hours)
        if ((baseline_hours, stage_count) not in ((32, 4), (56, 8)) or corrected_sampler or not require_corrected_sampler
                or source_uri != pinned['source_uri'] or expected_p8_sha256 != pinned['policy_sha256']):
            raise ValueError(f'Scaled continuation must use its pinned corrected P{baseline_hours} source and stage count')
        originals = [source_spec(v)['source_uri'] for v in ('wide', 'deep')]
        if baseline_hours == 56:
            originals += [source_spec(v, 56)['source_uri'] for v in ('wide', 'deep')]
        if any(output_uri == uri or output_uri.startswith(uri+'/') or uri.startswith(output_uri+'/')
               for uri in originals):
            raise ValueError('Scaled continuation output must preserve both original runs')
    if sequential_source:
        from .sequential_continuation import source_spec as sequential_spec
        if scaled_variant is not None:
            raise ValueError('Choose sequential or scaled continuation, not both')
        pinned = sequential_spec()
        if (baseline_hours != 8 or corrected_sampler or not require_corrected_sampler
                or source_uri != pinned['source_uri'] or expected_p8_sha256 != pinned['policy_sha256']):
            raise ValueError('Sequential continuation must use its pinned corrected P8 source')
    if resume_checkpoint:
        source, trainer = root/'original-run', root/'archived-trainer'
        if sha256_file(source/f'{baseline}.pkl') != expected_p8_sha256:
            raise ValueError('Retained policy does not match the requested original baseline')
        if not trainer.is_dir():
            raise ValueError('Original archived trainer is missing')
    else:
        options = (dict(expected_archive_sha256=pinned['source_archive_sha256'],
                        expected_iteration=pinned['iteration'],
                        expected_state_manifest_sha256=pinned['state_manifest_sha256']) if pinned else {})
        source,trainer=download_source(source_uri,root,expected_p8_sha256,baseline_hours, **options)
    original_archive_sha256 = sha256_file(source/archive)
    if corrected_sampler:
        from .corrected_continuation import prepare_corrected_source
        source,trainer=prepare_corrected_source(source,trainer,root)
        for name in ('sampler-migration.json', 'original-P32-verified.json', 'original-P32-verify.log'):
            storage('cp','--if-generation-match=0',root/name,output_uri+'/'+name)
    if require_corrected_sampler:
        if sequential_source:
            from .sequential_continuation import verify_sequential_source
            receipt = verify_sequential_source(source, trainer)
        elif pinned:
            from .scaled_continuation import verify_scaled_source
            receipt = (verify_scaled_source(source, trainer, scaled_variant) if baseline_hours == 32 else
                       verify_scaled_source(source, trainer, scaled_variant, baseline_hours))
        else:
            from .corrected_continuation import verify_corrected_source
            receipt = verify_corrected_source(source, trainer)
        write_json(root/'sampler-verification.json', receipt)
        storage('cp', root/'sampler-verification.json', output_uri+'/sampler-verification.json')
    # The resumable state belongs to this archived trainer, not to the new
    # coordinator source.tar.gz. Publish it explicitly for future continuations.
    if not resume_checkpoint:
        storage('cp','--if-generation-match=0',source/archive,output_uri+'/training_source.tar.gz')
    output=root/'run'
    stop=threading.Event()
    def sync(final=False):
        if not output.exists():
            return
        # Publication of recovery pointers and playable policy markers is ordered
        # by upload_checkpoint. Bulk sync must never bypass that transaction.
        exclude=r'^training/(?:state_[0-9]+(?:/.*)?|manifest\.json|latest\.json)$|^P[0-9]+\.(?:pkl|json)$'
        if not final:
            exclude+=r'|\.jsonl$'
        run_transfer(['gcloud', 'storage', 'rsync', '--recursive', '--exclude', exclude,
                      str(output), output_uri], timeout=1200)
    def publish_loop():
        while not stop.wait(120):
            try:
                sync()
            except Exception:
                pass  # Final sync is mandatory and reports failure.
    publisher=threading.Thread(target=publish_loop,daemon=True)
    publisher.start()
    try:
        # Fail before paid training if the evaluator assets cannot be prepared.
        assets=prepare(root/'fo2-assets')
        report=run_continuation(source,trainer,output,upload_uri=output_uri,
            resume_checkpoint=resume_checkpoint, expected_resume_sha256=expected_resume_sha256,
            wall_seconds=training_wall_seconds, baseline_hours=baseline_hours,
            require_objective_mass=corrected_sampler or require_corrected_sampler,
            verify_each_stage=bool(pinned),
            require_protected_sequential=sequential_source, stage_count=stage_count)
        write_json(output/'provenance.json',dict(source_uri=source_uri,output_uri=output_uri,
            baseline=baseline,expected_baseline_sha256=expected_p8_sha256,original_source_sha256=original_archive_sha256,
            scaled_variant=scaled_variant, sequential_source=sequential_source,
            stage_count=stage_count, additional_active_hours=6*stage_count,
            policy_group_sampling='objective_mass' if corrected_sampler or require_corrected_sampler else 'archived_trainer',
            training_source_fingerprint=report['training_source_fingerprint'],
            fo2_build_fingerprint=assets['fingerprint']))
        write_json(output/'resume.json',dict(training_source_archive=output_uri+'/training_source.tar.gz',
            training_manifest=output_uri+'/training/manifest.json',
            latest_pointer=output_uri+'/training/latest.json',
            source_fingerprint=report['training_source_fingerprint'],
            instructions='Restore the latest full state and manifest, use the archived training source and exact runtime; preserve all contract arguments when invoking hulhe_escher.train --resume. Playable policies alone cannot resume training.'))
        sync()
        evaluation=run_suite(output,output/'evaluation',workers=8,max_seconds=43200,assets=root/'fo2-assets',
            primary_minimum=primary_minimum,primary_effect_bb100=primary_effect_bb100,
            extra_seed_offset=pinned['evaluation_seed_offset'] if pinned else 0)
        if evaluation['status'] != 'COMPLETE':
            raise RuntimeError('Continuation evaluation incomplete; inspect evaluation/report.json')
    except Exception as exc:
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/'cloud_failure.json',dict(error_type=type(exc).__name__,reason=str(exc)))
        raise
    finally:
        stop.set()
        publisher.join(timeout=1250)
        sync(final=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--source-uri',required=True)
    p.add_argument('--output-uri',required=True)
    p.add_argument('--expected-baseline-sha256','--expected-p8-sha256',dest='expected_p8_sha256',required=True)
    p.add_argument('--baseline-hours',type=int,default=8)
    p.add_argument('--stage-count',type=int,choices=(4,8),default=4)
    p.add_argument('--resume-checkpoint')
    p.add_argument('--expected-resume-sha256')
    p.add_argument('--training-wall-seconds', type=float, default=108000)
    p.add_argument('--primary-minimum',type=int,default=100000)
    p.add_argument('--primary-effect-bb100',type=float,default=2.)
    p.add_argument('--corrected-sampler', action='store_true')
    p.add_argument('--require-corrected-sampler', action='store_true',
                   help='Verify the already corrected archived trainer; do not migrate it again')
    p.add_argument('--sequential-source', action='store_true',
                   help='Continue the pinned protected sequential P8 with its unchanged archived trainer')
    p.add_argument('--scaled-variant', choices=('wide', 'deep'),
                   help='Continue the pinned architecture-specific P32 or P56 with its original archived trainer')
    run_cloud(**vars(p.parse_args()))


if __name__=='__main__':
    main()
