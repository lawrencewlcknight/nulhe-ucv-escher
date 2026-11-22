"""Frozen captured-target diagnostic; no collection or production checkpoint writes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .checkpointing import LoadedHULHEPolicy, sha256_file
from .diagnostic_model import ResidualPolicy
from .diagnostic_probes import regret_models
from .game import load_hulhe_game, serialisable_game_definition
from .policy_replay_artifacts import read_metadata
from .position_investigation.common import deadline_after, STREETS
from .position_investigation.oracle import LearnedOpponent
from .strategic_audit.engine import RiverSpace, river_audit, consistency_checks, probabilities
from .strategic_audit.library import build_state, digest, validate_library
from .strategic_audit.workflow import RegretPolicy, write_json

ARMS = ('warm_current', 'expanded_current')
SEEDS = (3901, 3902)


class ExpandedRegret(torch.nn.Module):
    """Extra capacity with exactly the source prediction at initialization."""
    def __init__(self, source, layout):
        super().__init__()
        self.base = deepcopy(source)
        self.correction = ResidualPolicy(layout)  # Zero-initialized output head.

    def forward(self, x):
        return self.base(x)+self.correction(x)

    def checkpoint_metadata(self):
        return dict(type='hulhe_diagnostic_regret_expansion_v1',
                    base=self.base.checkpoint_metadata(),correction=self.correction.checkpoint_metadata())


def groups(features, mask):
    x = np.asarray(features, dtype=np.float32).copy()
    x[x == 0] = 0  # Canonicalise signed zero before byte hashing.
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    _, first, inverse, counts = np.unique(x, axis=0, return_index=True, return_inverse=True, return_counts=True)
    if not np.array_equal(mask, mask[first][inverse]):
        raise ValueError('Identical information states have different legal masks')
    return first, inverse, counts


def decomposition(features, target, prediction, mask):
    """Empirical ANOVA identity, weighted by retained rows and legal actions.

    Singleton groups provide no estimate of within-state variation. Sample means
    are noisy; neither component estimates true expected-target error by itself.
    """
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    mask = np.asarray(mask)
    if (target.shape != (len(features), 3) or prediction.shape != target.shape or mask.shape != target.shape
            or not np.isfinite(target).all() or not np.isfinite(prediction).all()
            or not np.isin(mask, [0, 1]).all() or not np.all(mask.sum(1) > 0)):
        raise ValueError('Invalid targets, predictions or masks')
    mask = mask.astype(bool)
    first, inv, count = groups(features, mask)
    means = np.zeros((len(first), 3)); np.add.at(means, inv, target)
    means /= count[:, None]
    pred = np.zeros_like(means); np.add.at(pred, inv, prediction); pred /= count[:, None]
    # The decomposition requires a deterministic prediction for the same input.
    if not np.allclose(prediction[mask], pred[inv][mask], atol=2e-5, rtol=2e-5):
        raise ValueError('Repeated-state predictions differ; cannot decompose model error')
    sse = np.zeros_like(means); np.add.at(sse, inv, (target-means[inv])**2)
    legal = mask[first]
    def summarize(keep):
        denom = int((count[keep, None]*legal[keep]).sum())
        if not denom:
            return dict(groups=0, rows=0, legal_values=0, total_mse=None,
                        within_state_variance=None, mean_target_mse=None, mean_error_fraction=None)
        within = float(sse[keep][legal[keep]].sum()/denom)
        between = float((count[keep, None]*(pred[keep]-means[keep])**2)[legal[keep]].sum()/denom)
        return dict(groups=int(keep.sum()), rows=int(count[keep].sum()), legal_values=denom,
            total_mse=within+between, within_state_variance=within, mean_target_mse=between,
            mean_error_fraction=between/(within+between) if within+between else 0.)
    result = dict(all=summarize(np.ones(len(first), bool)), repeated=summarize(count > 1),
        singletons=summarize(count == 1), max_multiplicity=int(count.max()))
    # Report unbiased sample variance / n as an uncertainty scale, not as an
    # independent standard error: traversal targets may be correlated.
    repeated = count > 1
    if repeated.any():
        variance = sse[repeated]/(count[repeated, None]-1)
        result['repeated']['iid_mean_noise_scale'] = float(
            variance[legal[repeated]].sum()/result['repeated']['legal_values'])
        result['repeated']['iid_noise_adjusted_mean_mse'] = (
            result['repeated']['mean_target_mse']-result['repeated']['iid_mean_noise_scale'])
    else:
        result['repeated']['iid_mean_noise_scale'] = None
        result['repeated']['iid_noise_adjusted_mean_mse'] = None
    result['by_multiplicity'] = {name:summarize(keep) for name,keep in (
        ('1',count==1),('2-4',(count>=2)&(count<5)),('5-9',(count>=5)&(count<10)),('10+',count>=10))}
    empirical = float(((prediction-target)**2)[mask].mean())
    result['identity_residual'] = empirical-result['all']['total_mse']
    if not np.isclose(empirical, result['all']['total_mse'], rtol=1e-5, atol=1e-8):
        raise ValueError('Variance decomposition identity failed')
    return result


def partition(features, mask):
    first, inv, _ = groups(features, mask)
    # All duplicates stay in one partition, independently of targets and errors.
    values = np.asarray([int.from_bytes(hashlib.blake2b(np.where(row == 0, 0., row).astype('float32').tobytes(),
        digest_size=8, person=b'regret-split-v1').digest(), 'big') % 100 for row in features[first]])
    split = np.where(values < 60, 0, np.where(values < 80, 1, 2))[inv]
    return [np.flatnonzero(split == i) for i in range(3)]


def load_captures(captures, encoder):
    root = Path(captures)
    report = json.loads((root/'report.json').read_text())
    if report['status'] != 'COMPLETE' or report['iteration'] != report['source_iteration']+1:
        raise ValueError('Requires a complete single-iteration capture')
    result = []
    for player in range(2):
        name = f'regret_{player}'
        record = next(x for x in report['captures'] if x['name'] == name)
        before, after = root/f'{name}_before.npz', root/f'{name}_after.npz'
        if record['artifact'] != before.name or sha256_file(before) != record['sha256']:
            raise ValueError('Captured target checksum mismatch')
        if record.get('after_sha256') and sha256_file(after)!=record['after_sha256']:
            raise ValueError('Captured post-fit prediction checksum mismatch')
        with np.load(before, allow_pickle=False) as f:
            data = {k:f[k].copy() for k in ('features', 'target', 'mask', 'before', 'row_indices')}
        with np.load(after, allow_pickle=False) as f:
            data['after'] = f['prediction'].copy()
        if len(np.unique(data['row_indices'])) != len(data['features']) or len(data['features']) != record['sampled_rows']:
            raise ValueError('Duplicate replay row indices or wrong sample size')
        player_cols, street_cols = encoder.replay_context_columns()
        if data['features'].shape[1] != encoder.policy_layout.total_size:
            raise ValueError('Captured encoder shape mismatch')
        if not np.all(data['features'][:, player_cols].argmax(1) == player):
            raise ValueError('Capture contains wrong player')
        data['street'] = data['features'][:, street_cols].argmax(1)
        data['partitions'] = partition(data['features'], data['mask'])
        for phase in ('before', 'after'):
            mse = float(((data[phase].astype(float)-data['target'])**2)[data['mask'].astype(bool)].mean())
            if not np.isclose(mse, record[phase+'_mse'], rtol=1e-5, atol=1e-8):
                raise ValueError('Capture predictions disagree with recorded MSE')
        data['hashes'] = dict(before=sha256_file(before), after=sha256_file(after))
        result.append(data)
    return report, result


def by_street(data, prediction, ids=None):
    ids = np.arange(len(data['features'])) if ids is None else ids
    out = {}
    for i, name in enumerate(STREETS):
        rows = ids[data['street'][ids] == i]
        out[name] = decomposition(data['features'][rows], data['target'][rows], prediction[rows], data['mask'][rows]) if len(rows) else None
    return out


def fit_gate(decompositions):
    repeated = [d['after']['preflop']['repeated'] if d['after']['preflop'] else {} for d in decompositions]
    enough = all(r.get('groups', 0) >= 30 and r.get('rows', 0) >= 100 for r in repeated)
    material = any((r.get('mean_error_fraction') or 0) >= .10 for r in repeated)
    return dict(run_fits=enough and material, sufficient_repetition=enough, material_empirical_mean_error=material,
        rule='Both players: >=30 repeated preflop groups and >=100 repeated rows; at least one has empirical mean-target error >=10% of repeated-row MSE after fitting.',
        interpretation='Exploratory budget gate, not a significance test or proof of insufficient capacity. Compare the empirical residual with the reported IID mean-noise sensitivity before interpreting it.')


def predict(model, data):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(data['features'][s:s+1024])).numpy()
            for s in range(0, len(data['features']), 1024)])


def scores(models, data, split):
    cells = [by_street(d, predict(m, d), d['partitions'][split]) for m,d in zip(models,data)]
    # Selection uses only preflop mean-target errors; all streets are reported.
    rows = [c['preflop']['all'] for c in cells if c['preflop']]
    denom = sum(r['legal_values'] for r in rows)
    return dict(players=cells, preflop_mean_target_mse=sum(r['mean_target_mse']*r['legal_values'] for r in rows)/denom if denom else None)


def fit_candidate(source, data, encoder, arm, seed, check, *, updates=(100,250,500), lr=.0003):
    torch.manual_seed(seed)
    models = [deepcopy(m) for m in source]
    if arm == 'expanded_current':
        models = [ExpandedRegret(m,encoder.policy_layout) for m in source]
    elif arm != 'warm_current':
        raise ValueError('Unknown fitting arm')
    for m,s,d in zip(models,source,data):
        if not np.array_equal(predict(m,d),predict(s,d)):
            raise ValueError('Capacity candidate must preserve initial source predictions exactly')
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    rng = np.random.default_rng(seed)
    initial_validation = scores(models,data,1)
    best_score = initial_validation['preflop_mean_target_mse']
    if best_score is None or not np.isfinite(best_score):raise ValueError('No finite preflop validation score')
    best_models, best_step = deepcopy(models), 0
    history = [dict(update=0,validation_preflop_mean_mse=best_score)]; started = time.monotonic()
    for step in range(1, max(updates)+1):
        check()
        for m,opt,d in zip(models,optimizers,data):
            train = d['partitions'][0]
            idx = rng.choice(train, min(1024,len(train)), replace=True)
            x,y,mask = [torch.from_numpy(d[k][idx]).float() for k in ('features','target','mask')]
            m.train(); opt.zero_grad(set_to_none=True)
            # Uniform rows, mean over three actions, matching production masked MSE.
            loss = ((m(x)-y)*mask).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fit loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 10.); opt.step()
        if step in updates:
            for m in models: m.eval()
            value = scores(models,data,1)['preflop_mean_target_mse']
            if value is None or not np.isfinite(value): raise ValueError('No finite preflop validation score')
            history.append(dict(update=step, validation_preflop_mean_mse=value))
            if value < best_score:
                best_score,best_step,best_models = value,step,deepcopy(models)
    return best_models, dict(arm=arm,seed=seed,selected_update=best_step,validation_history=history,
        validation_preflop_mean_mse=best_score,seconds=time.monotonic()-started,learning_rate=lr,
        optimizer='Fresh Adam; both arms copy source pre-iteration weights, not captured post-fit weights or optimizer state',
        initialization='Source predictions preserved exactly; expanded_current adds a zero-output trainable residual correction',
        parameters_per_player=sum(p.numel() for p in models[0].parameters()),
        initial_validation=initial_validation, training=scores(best_models,data,0),validation=scores(best_models,data,1))


def river_positions(library):
    # Every encountered common river in development, no error-based filtering.
    return [p for p in library['positions'] if p['partition']=='development' and len(p['board'])==5]


def river_coverage(library, data, encoder):
    result={}
    for pos in river_positions(library):
        state,_=build_state(pos); d=data[pos['hero']]
        x=encoder.information_state(state,pos['hero'])
        matches=np.all(d['features']==x,axis=1)
        result[pos['id']]=dict(captured_rows=int(matches.sum()),
            **{name:int(matches[ids].sum()) for name,ids in zip(('train','validation','audit'),d['partitions'])})
    return result


def same_regret_models(a,b):
    if not isinstance(a,RegretPolicy) or not isinstance(b,RegretPolicy):return False
    if a.encoder.metadata()!=b.encoder.metadata() or vars(a.settings)!=vars(b.settings):return False
    if len(a.models)!=len(b.models):return False
    for x,y in zip(a.models,b.models):
        if type(x) is not type(y) or x.checkpoint_metadata()!=y.checkpoint_metadata():return False
        xs,ys=x.state_dict(),y.state_dict()
        if xs.keys()!=ys.keys() or any(not torch.equal(xs[k],ys[k]) for k in xs):return False
    return True


def audit_rivers(library, baseline, candidate, opponent, check, progress=lambda rows:None):
    rows = []
    if same_regret_models(baseline,candidate):
        # Selecting update zero changes no strategy. Verify all common roots and
        # invariants, but do not repeat expensive hidden-hand tree integration.
        for pos in river_positions(library):
            check();state,_=build_state(pos)
            a,b=probabilities(baseline,state),probabilities(candidate,state)
            if not np.array_equal(a,b):raise ValueError('Identical model action check failed')
            rows.append(dict(id=pos['id'],family=pos['family'],source=pos['source'],weight=pos['encounter_weight'],
                baseline=dict(status='COMPLETE',probabilities=a.tolist()),
                candidate=dict(status='COMPLETE',probabilities=b.tolist()),
                checks=consistency_checks(candidate,pos,check)))
        return dict(positions=rows,requested_positions=len(rows),method='Exact model and strategy identity; common-root actions checked',
            by_encounter_source={s:dict(positions=sum(r['source']==s for r in rows),
                encounter_weight=sum(r['weight'] for r in rows if r['source']==s),
                value_change_bb=0.,local_gap_change_bb=0.) for s in sorted({r['source'] for r in rows})},
            note='Unchanged architecture, parameters, encoder and regret-matching settings imply identical strategy throughout each tree. Zero paired changes follow from identity; absolute conditional values were not recomputed. Assessment remains locked.')
    for pos in river_positions(library):
        check(); space=RiverSpace(pos); response=LearnedOpponent(space,opponent,check)
        weights=response.posterior()
        a=river_audit(space,baseline,response,weights,check)
        b=river_audit(space,candidate,response,weights,check)
        rows.append(dict(id=pos['id'],family=pos['family'],source=pos['source'],weight=pos['encounter_weight'],
            baseline=a,candidate=b,checks=consistency_checks(candidate,pos,check)))
        if len(rows)%8==0:progress(rows)
    summary={}
    for source in sorted({r['source'] for r in rows}):
        subset=[r for r in rows if r['source']==source and r['baseline']['status']==r['candidate']['status']=='COMPLETE']
        mass=sum(r['weight'] for r in subset)
        summary[source]=dict(positions=len(subset),encounter_weight=mass,
            value_change_bb=sum(r['weight']*(r['candidate']['candidate_value_bb']-r['baseline']['candidate_value_bb']) for r in subset)/mass if mass else None,
            local_gap_change_bb=sum(r['weight']*(r['candidate']['root']['local_gap_bb']-r['baseline']['root']['local_gap_bb']) for r in subset)/mass if mass else None)
    return dict(positions=rows,by_encounter_source=summary,requested_positions=len(river_positions(library)),
        note='Fixed source average-policy opponent and history-conditioned posterior for both candidates, including FO2-encountered roots. Conditional bb, not bb/100 or exploitability. Correlated development panel, no significance claim; assessment remains locked.')


def render(output, report):
    lines=['# Captured regret-target investigation','',f"Status: {report['status']}",'',
        'Empirical squared-error decomposition; all numbers use normalized regret-target units.', '',
        '| Player | Phase | Repeated preflop groups | Target variance | Mean-target MSE | IID mean-noise scale |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for player,d in enumerate(report.get('decomposition',[])):
        for phase in ('before','after'):
            r=d[phase]['preflop']['repeated'] if d[phase]['preflop'] else None
            if r and r['rows']:
                lines.append(f"| {player} | {phase} | {r['groups']} | {r['within_state_variance']:.6g} | {r['mean_target_mse']:.6g} | {r['iid_mean_noise_scale']:.6g} |")
    if report.get('results'):
        lines += ['', '| Fit | Seed | Update | Training mean MSE | Validation mean MSE | Audit mean MSE | Seconds |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in report['results']:
            audit=r.get('audit',{}).get('preflop_mean_target_mse')
            lines.append(f"| {r['arm']} | {r['seed']} | {r['selected_update']} | {r['training']['preflop_mean_target_mse']:.6g} | {r['validation_preflop_mean_mse']:.6g} | {format(audit,'.6g') if audit is not None else 'pending'} | {r['seconds']:.1f} |")
    if 'selection' in report:
        s=report['selection']
        lines += ['',f"Selected: {s['arm']}, fixed seed {s['seed']}. Mean validation score beats source: {s['beats_source_validation']}."]
    if 'river' in report:
        lines += ['', '| Encounter source | Positions | Conditional value change (bb) | Local-gap change (bb) |',
                  '| --- | ---: | ---: | ---: |']
        for source,r in report['river']['by_encounter_source'].items():
            lines.append(f"| {source} | {r['positions']} | {r['value_change_bb']} | {r['local_gap_change_bb']} |")
        lines += ['',report['river']['note']]
    lines += ['',report.get('fit_status',''),'',
        'Within-state variation is an empirical component, not a proven irreducible noise floor. Sample mean targets remain uncertain; singleton groups cannot estimate variation.',
        'Fitting partitions are held out from this refit only, not necessarily from historical training. One iteration and two fit seeds do not establish playing strength.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def run(captures, policy, encounters, output, *, state=None, fits='auto', threads=4, max_seconds=3600):
    if fits not in ('auto','off','force') or threads<1: raise ValueError('Invalid diagnostic settings')
    check=deadline_after(max_seconds); torch.set_num_threads(threads)
    output=Path(output); output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    report=dict(status='ANALYSING',protocol='hulhe_regret_target_diagnostic_v1',fit_mode=fits,results=[],
        limitations=['Captured rows are a bounded sample from one iteration, not full replay.',
        'Within-state variation need not be independent or irreducible; empirical mean-target error includes finite-sample uncertainty.',
        'Fresh group holdouts apply to diagnostic refits; the source model may have historical exposure.',
        'Candidates are diagnostic instantaneous regret strategies, not average policies or resumable checkpoints.'])
    try:
        loaded=LoadedHULHEPolicy(load_hulhe_game(),policy)
        capture_report,data=load_captures(captures,loaded.feature_encoder)
        encounter_report=json.loads((Path(encounters)/'report.json').read_text())
        library=validate_library(json.loads((Path(encounters)/'library.json').read_text()))
        if (encounter_report['status']!='COMPLETE' or digest(library)!=encounter_report['library_sha256']
                or sha256_file(policy)!=encounter_report['source_policy_sha256']
                or sha256_file(policy)!=capture_report['source_policy_sha256']
                or loaded.checkpoint['outer_iteration']!=capture_report['source_iteration']):
            raise ValueError('Captures, positions and source policy identities differ')
        report.update(source_policy_sha256=sha256_file(policy),source_manifest_sha256=capture_report['source_manifest_sha256'],
            source_iteration=capture_report['source_iteration'],capture_iteration=capture_report['iteration'],
            library_sha256=digest(library),capture_hashes=[d['hashes'] for d in data],
            decomposition=[{phase:by_street(d,d[phase]) for phase in ('before','after')} for d in data],
            splits=[{name:dict(rows=len(ids),sha256=hashlib.sha256(ids.tobytes()).hexdigest())
                     for name,ids in zip(('train','validation','audit'),d['partitions'])} for d in data])
        report['fit_gate']=fit_gate(report['decomposition'])
        write_json(output/'report.json',report)
        should_fit=fits=='force' or (fits=='auto' and report['fit_gate']['run_fits'])
        if fits=='off' or not should_fit:
            report['fit_status']='SKIPPED: fitting disabled or empirical residual/repetition gate not met.'
        elif state is None:
            report['status']='NEEDS_METADATA';report['fit_status']='Gate passed; supply --state with checksum-verified source metadata for frozen fits.'
            return report
        else:
            _,metadata=read_metadata(state)
            if (sha256_file(Path(state)/'manifest.json')!=capture_report['source_manifest_sha256']
                    or metadata['solver']['num_iteration']!=capture_report['source_iteration']
                    or metadata['game']!=serialisable_game_definition()
                    or metadata['encoder']!=loaded.checkpoint['feature_encoder']
                    or metadata['config']!=loaded.checkpoint['training_config']):
                raise ValueError('Recovery metadata differs from captured source')
            source=regret_models(metadata,loaded.feature_encoder)
            for m,d in zip(source,data):
                if not np.allclose(predict(m,d)*d['mask'],d['before'],rtol=3e-5,atol=3e-5):
                    raise ValueError('Source regret model does not reproduce captured pre-fit predictions')
                if any(not len(ids) or not np.any(d['street'][ids]==0) for ids in d['partitions']):
                    raise ValueError('Insufficient preflop coverage in fitting partitions')
            report['source_validation']=scores(source,data,1)
            report['status']='FITTING'
            candidates={}
            for seed in SEEDS:
                for arm in ARMS:
                    check();models,row=fit_candidate(source,data,loaded.feature_encoder,arm,seed,check)
                    candidates[(arm,seed)]=models;report['results'].append(row)
                    write_json(output/'report.json',report)
            means={arm:float(np.mean([r['validation_preflop_mean_mse'] for r in report['results'] if r['arm']==arm])) for arm in ARMS}
            winner=min(ARMS,key=lambda a:(means[a],ARMS.index(a)))
            selected=candidates[(winner,SEEDS[0])]
            report['selection']=dict(arm=winner,seed=SEEDS[0],mean_validation_scores=means,
                rule='Lowest two-seed mean preflop validation mean-target MSE; first predeclared seed exported; river/audit not used for selection.',
                beats_source_validation=means[winner]<report['source_validation']['preflop_mean_target_mse'])
            write_json(output/'selection.json',report['selection'])
            # Selection is now frozen; inspect audit partitions and river decisions.
            for row in report['results']:
                row['audit']=scores(candidates[(row['arm'],row['seed'])],data,2)
            report['source_audit']=scores(source,data,2)
            torch.save(dict(kind='diagnostic_regret_models_only',source_policy_sha256=report['source_policy_sha256'],
                selection=report['selection'],models=[m.state_dict() for m in selected],
                architectures=[m.checkpoint_metadata() for m in selected]),output/'selected_regret.pt')
            baseline=RegretPolicy(state);candidate=deepcopy(baseline);candidate.models=selected
            report['river_capture_coverage']=river_coverage(library,data,loaded.feature_encoder)
            report['status']='RIVER_AUDIT';write_json(output/'report.json',report)
            def river_progress(rows):
                report['river_progress']=dict(completed_positions=len(rows),requested_positions=len(river_positions(library)))
                write_json(output/'river_partial.json',dict(status='INCOMPLETE',positions=rows))
                write_json(output/'report.json',report)
            report['river']=audit_rivers(library,baseline,candidate,loaded,check,river_progress)
            write_json(output/'river_partial.json',dict(status='COMPLETE',positions=report['river']['positions']))
            report['fit_status']='COMPLETE: frozen-target fits and common-river decision checks; no production configuration changed.'
        report['status']='COMPLETE'
    except TimeoutError as exc:
        report.update(status='INCOMPLETE',reason=str(exc))
    except BaseException as exc:
        report.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        write_json(output/'report.json',report);render(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('captures','policy','encounters','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--state',type=Path);p.add_argument('--fits',choices=('auto','off','force'),default='auto')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--max-seconds',type=float,default=3600)
    if run(**vars(p.parse_args()))['status']!='COMPLETE':raise SystemExit(2)

if __name__=='__main__':main()
