"""Locked P56 development river panels for fixed P104 diagnostic candidates."""
import argparse
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from .checkpointing import LoadedHULHEPolicy, sha256_file
from .game import load_hulhe_game
from .p104_policy_evaluate import verified_exports
from .p104_protocol import P56_SHA256, P56_LIBRARY_SHA256, POLICY_SEEDS
from .policy_replay_artifacts import safe_path
from .position_investigation.common import deadline_after
from .position_investigation.compare import paired_values, summarize
from .position_investigation.oracle import LearnedOpponent
from .regret_diagnostic import ExpandedRegret, same_regret_models
from .strategic_audit.engine import RiverSpace, probabilities, river_audit, consistency_checks
from .strategic_audit.library import validate_library, digest, build_state
from .strategic_audit.workflow import RegretPolicy
from .train import write_json

_MODELS={}
_OPPONENT=None
_DEADLINE=None
_INITIALISATION_ERROR=None


def load_models(specs, opponent, deadline):
    global _MODELS,_OPPONENT,_DEADLINE
    torch.set_num_threads(1);_DEADLINE=deadline
    _MODELS={}
    for name,spec in specs.items():
        if spec['kind']=='checkpoint':model=LoadedHULHEPolicy(load_hulhe_game(),spec['checkpoint'])
        else:
            model=RegretPolicy(spec['state'])
            if spec.get('artifact'):
                saved=torch.load(spec['artifact'],map_location='cpu',weights_only=False)
                if saved['kind']!='p104_diagnostic_regret_only':raise ValueError('Invalid regret export')
                if saved['arm']=='expanded_group_mean':
                    model.models=[ExpandedRegret(m,model.encoder.policy_layout) for m in model.models]
                for m,weights,architecture in zip(model.models,saved['models'],saved['architectures'],strict=True):
                    if m.checkpoint_metadata()!=architecture:raise ValueError('Regret architecture mismatch')
                    m.load_state_dict(weights);m.eval()
        _MODELS[name]=model
    _OPPONENT=LoadedHULHEPolicy(load_hulhe_game(),opponent)


def initialise(specs, opponent, deadline):
    global _INITIALISATION_ERROR
    _INITIALISATION_ERROR=None
    try:load_models(specs,opponent,deadline)
    except Exception as exc:
        # Pool must report one failure, not repeatedly respawn failing initializers.
        _INITIALISATION_ERROR=f'{type(exc).__name__}: {exc}'


def position_result(pos):
    if _INITIALISATION_ERROR:raise RuntimeError(_INITIALISATION_ERROR)
    def check():
        if time.monotonic()>_DEADLINE:raise TimeoutError('River stage deadline')
    check();state,_=build_state(pos);space=RiverSpace(pos)
    opponent=LearnedOpponent(space,_OPPONENT,check)
    panels=[('frozen_P56',opponent,opponent.posterior())]+[
        (name+'/'+prior,name,space.weights(prior)) for name in ('check_call','value','pressure')
        for prior in ('uniform','strong','weak')]
    row=dict(id=pos['id'],source=pos['source'],family=pos['family'],position=pos,
        encounter_weight=pos['encounter_weight'],policies={})
    for name,model in _MODELS.items():
        values={}
        identical=(name!='P104' and same_regret_models(_MODELS['P104'],model))
        for panel,response,weights in panels:
            values[panel]=(row['policies']['P104']['panels'][panel] if identical else
                           river_audit(space,model,response,weights,check))
        row['policies'][name]=dict(probabilities=probabilities(model,state).tolist(),panels=values,
            checks=consistency_checks(model,pos,check))
    return row


def comparison_summaries(rows,pairs):
    result={}
    for candidate,baseline in pairs:
        converted=[]
        for r in rows:
            a,b=r['policies'][baseline],r['policies'][candidate]
            converted.append(dict(id=r['id'],source=r['source'],family=r['family'],position=r['position'],
                encounter_weight=r['encounter_weight'],panels={k:dict(change=paired_values(a['panels'][k],b['panels'][k])) for k in a['panels']}))
        groups={}
        for r in rows:
            for group in ('all','source:'+r['source'],'family:'+r['family']):
                groups.setdefault(group,[]).append(r)
        actions={g:dict(zip(('fold','call_check','raise_bet'),np.average([
            np.asarray(r['policies'][candidate]['probabilities'])-np.asarray(r['policies'][baseline]['probabilities'])
            for r in subset],axis=0,weights=[r['encounter_weight'] for r in subset]).tolist())) for g,subset in groups.items()}
        result[candidate+'-vs-'+baseline]=dict(summaries=summarize(converted),action_probability_changes=actions)
    return result


def run(experiment,policy,frozen_policy,encounters,output,*,mode,state=None,workers=8,max_seconds=2400,
        expected_frozen_sha256=P56_SHA256,expected_library_sha256=P56_LIBRARY_SHA256,smoke=False):
    if mode not in ('policy','regret') or workers<1:raise ValueError('Invalid river settings')
    check=deadline_after(max_seconds);deadline=time.monotonic()+max_seconds
    encounters=Path(encounters)
    library=validate_library(json.loads((encounters/'library.json').read_text()))
    original=json.loads((encounters/'report.json').read_text())
    if (sha256_file(frozen_policy)!=expected_frozen_sha256 or digest(library)!=expected_library_sha256
        or original['status']!='COMPLETE' or original['source_policy_sha256']!=expected_frozen_sha256
        or original['library_sha256']!=expected_library_sha256):raise ValueError('Frozen P56 input identity mismatch')
    positions=[p for p in library['positions'] if p['partition']=='development' and len(p['board'])==5]
    if not positions or (not smoke and len(positions)!=90):raise ValueError('Expected 90 locked development rivers')
    specs={};pairs=[];experiment=Path(experiment)
    if mode=='policy':
        exports,fit=verified_exports(experiment,policy)
        if bool(fit['smoke'])!=bool(smoke):raise ValueError('Smoke mismatch')
        specs['P104']=dict(kind='checkpoint',checkpoint=str(Path(policy).resolve()))
        specs.update(exports)
        for seed in POLICY_SEEDS:
            control,refit=f'control_{seed}',f'all_replay_{seed}'
            pairs += [(control,'P104'),(refit,'P104'),(refit,control)]
    else:
        report=json.loads((experiment/'report.json').read_text())
        selection=json.loads((experiment/'selection.json').read_text())
        if (report['status']!='COMPLETE' or report['selection']!=selection or bool(report['smoke'])!=bool(smoke)
                or report['source_policy_sha256']!=sha256_file(policy)
                or report['source_manifest_sha256']!=sha256_file(Path(state)/'manifest.json')):
            raise ValueError('Regret selection not frozen or source mismatch')
        specs['P104']=dict(kind='regret',state=str(Path(state).resolve()))
        for row in report['exports']:
            path=safe_path(experiment,row['path'])
            if sha256_file(path)!=row['sha256']:raise ValueError('Regret export checksum mismatch')
            name=f'regret_{row["seed"]}'
            specs[name]=dict(kind='regret',state=str(Path(state).resolve()),artifact=str(path))
            pairs.append((name,'P104'))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',mode=mode,smoke=smoke,positions=[],comparisons={},requested_positions=len(positions),
        library_sha256=expected_library_sha256,frozen_opponent_sha256=expected_frozen_sha256,
        candidate_specs=specs,assessment_unlocked=False,
        limitations=['Frozen P56 opponent and conditional ranges, including strong/uniform/weak synthetic panels.',
            'Development positions already inspected: descriptive diagnostics, never model-selection or independent playing-strength evidence.',
            'Conditional bb per decision, not bb/100 or exploitability. No confidence intervals on correlated position averages.'])
    write_json(output/'manifest.json',{k:v for k,v in report.items() if k not in ('positions','comparisons')})
    try:
        if workers==1:
            initialise(specs,str(frozen_policy),deadline)
            for pos in positions:
                check();report['positions'].append(position_result(pos));write_json(output/'report.json',report)
        else:
            with mp.get_context('spawn').Pool(workers,initializer=initialise,initargs=(specs,str(frozen_policy),deadline)) as pool:
                iterator=pool.imap(position_result,positions,chunksize=1)
                for _ in positions:
                    check();report['positions'].append(iterator.next(timeout=max(.1,deadline-time.monotonic())))
                    write_json(output/'report.json',report)
        report['status']='COMPLETE'
    except (TimeoutError,mp.TimeoutError) as exc:report.update(status='INCOMPLETE',reason=str(exc))
    except BaseException as exc:
        report.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['comparisons']=comparison_summaries(report['positions'],pairs)
        report['completed_positions']=len(report['positions']);write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('experiment','policy','frozen-policy','encounters','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('policy','regret'),required=True);p.add_argument('--state',type=Path)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--max-seconds',type=float,default=2400)
    p.add_argument('--smoke',action='store_true')
    if run(**vars(p.parse_args()))['status']!='COMPLETE':raise SystemExit(2)

if __name__=='__main__':main()
