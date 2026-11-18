"""Match frozen positions to replay and score development positions only by default."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from unbiased_escher.replay_grouping import grouped_policy_data, exact_group_indices
from unbiased_escher.holdem_fitting import policy_validation_mask
from ..checkpointing import LoadedHULHEPolicy, sha256_file
from ..game import load_hulhe_game, serialisable_game_definition
from ..policy_replay_artifacts import load_policy_replay
from ..strategic_audit.engine import RiverSpace, river_audit, probabilities, consistency_checks
from ..strategic_audit.library import build_state, validate_library, digest
from ..strategic_audit.workflow import RegretPolicy, write_json
from .common import families, deadline_after
from .oracle import learned_audit


def fingerprint(x):
    x=np.asarray(x,dtype=np.float32).copy(); x[x==0]=0
    return x.tobytes()


def replay_scan(buffer,metadata,policy,positions,deadline=lambda:None):
    encoder=policy.feature_encoder
    encoded=np.stack([encoder.information_state(build_state(p)[0],p['hero']) for p in positions])
    wanted=defaultdict(list)
    for i,x in enumerate(encoded): wanted[fingerprint(x)].append(i)
    data,coverage=grouped_policy_data(buffer,metadata['solver']['num_iteration'],metadata['config']['gamma'],'cpu',
        context_columns=encoder.replay_context_columns())
    first,inverse=exact_group_indices(buffer.infostate_buf)
    if not np.array_equal(first,data[0].rows): raise ValueError('Replay grouping identity mismatch')
    counts=np.bincount(inverse,minlength=len(first)); del inverse
    matches=[None]*len(positions); bins=defaultdict(lambda:np.zeros(5))
    heldout=policy_validation_mask(data[0],metadata['config']['policy_validation_fraction'])
    for start in range(0,len(first),2048):
        deadline(); stop=min(start+2048,len(first)); x=data[0].numpy_batch(start,stop)
        keys=families(x,encoder)
        y,mask,w=[v[start:stop].numpy() for v in data[1:]]
        with torch.no_grad():
            logp=torch.log_softmax(policy.model(torch.from_numpy(x)).masked_fill(torch.from_numpy(mask)!=1,-1e20),-1).numpy()
        kl=np.sum(y*(np.log(np.maximum(y,1e-30))-logp),axis=1)
        for j,(feature,key) in enumerate(zip(x,keys)):
            idx=start+j; split='historical_validation' if heldout[idx] else 'historical_training'
            # Store count, raw rows, objective mass, weighted KL, weighted L1.
            bins[key+'|'+split]+=np.asarray([1,counts[idx],w[j],w[j]*max(0,float(kl[j])),
                w[j]*np.abs(y[j]-np.exp(logp[j])).sum()])
            for target in wanted.get(fingerprint(feature),[]):
                matches[target]=dict(retained_rows=int(counts[idx]),objective_weight=float(w[j]),
                    target=y[j].tolist(),weighted_target_kl=max(0,float(kl[j])),split=split,
                    mask=mask[j].tolist())
    summary={k:dict(groups=int(v[0]),retained_rows=int(v[1]),objective_weight=float(v[2]),
        weighted_kl=float(v[3]/v[2]) if v[2]>0 else None,
        weighted_action_l1=float(v[4]/v[2]) if v[2]>0 else None) for k,v in bins.items()}
    return matches,summary,coverage


def select_upstream(rows, count=24):
    eligible=[r for r in rows if r['partition']=='development' and 'p32_opponent' in r
              and r['p32_opponent']['status']=='COMPLETE']
    # Equal high/low diagnostic gaps; choose controls before estimator inspection.
    ordered=sorted(eligible,key=lambda r:(r['p32_opponent']['root']['local_gap_bb'],r['id']))
    high=ordered[-min(count//2,len(ordered)):] if count//2 else []
    used={r['id'] for r in high}
    controls=[r for r in ordered if r['id'] not in used][:count-len(high)]
    return [dict(position=r['position'],selection_group='high_gap' if r in high else 'low_gap_control',
                 policy_gap_bb=r['p32_opponent']['root']['local_gap_bb']) for r in high+controls]


def summarize(rows,encounters):
    groups=defaultdict(list)
    for r in rows: groups[(r['source'],r['family'])].append(r)
    result=[]
    for (source,family),items in groups.items():
        scored=[r for r in items if r.get('p32_opponent',{}).get('status')=='COMPLETE']
        n=encounters.get('family_counts',{}).get(source+'|'+family,0)
        total=sum(v for k,v in encounters.get('family_counts',{}).items() if k.startswith(source+'|'))
        weights=np.asarray([r['encounter_weight'] for r in scored])
        gap=float(np.average([r['p32_opponent']['root']['local_gap_bb'] for r in scored],weights=weights)) if scored else None
        matches=[r['replay'] for r in items if r['replay'] is not None]
        result.append(dict(source=source,family=family,encounters=n,
            decision_frequency=n/total if total else None, audited_positions=len(items),
            exact_replay_matches=len(matches), mean_local_gap_bb=gap,
            prioritization_score=(n/total)*gap if gap is not None and total else None,
            note='Frequency times conditional gap is a triage score, not additive whole-game loss or bb/100.'))
    return sorted(result,key=lambda r:r['prioritization_score'] if r['prioritization_score'] is not None else -1,reverse=True)


def run(state,policy,encounters,output,*,max_seconds=3000,threads=1,unlock_assessment=False,upstream_positions=24):
    torch.set_num_threads(threads)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    check=deadline_after(max_seconds); started=time.monotonic()
    report=dict(status='ANALYSING',positions=[],
        partition='assessment' if unlock_assessment else 'development',
        limitations=['Conditional fixed-opponent response gaps are not equilibrium errors or exploitability.',
            'Missing retained replay is not proof a state was never visited; family matches are not substitute targets.',
            'Historical validation was used in source policy selection; it is not a fresh independent test.',
            'Repeated board orbits/decisions are correlated; no confidence intervals on audit gap averages.'])
    write_json(output/'report.json',report)
    try:
        encounter_report=json.loads((Path(encounters)/'report.json').read_text())
        library=validate_library(json.loads((Path(encounters)/'library.json').read_text()))
        if encounter_report['status']!='COMPLETE' or digest(library)!=encounter_report['library_sha256']:
            raise ValueError('Encounter library is incomplete or changed')
        if sha256_file(policy)!=encounter_report['source_policy_sha256']: raise ValueError('Policy differs from encounter policy')
        chosen=[p for p in library['positions'] if p['partition']==report['partition']]
        if not chosen: raise ValueError('Empty selected partition')
        loaded=LoadedHULHEPolicy(load_hulhe_game(),policy)
        buffer,metadata,manifest=load_policy_replay(state)
        if (metadata['game']!=serialisable_game_definition() or metadata['encoder']!=loaded.checkpoint['feature_encoder']
                or metadata['config']!=loaded.checkpoint['training_config']
                or metadata['solver']['num_iteration']!=loaded.checkpoint['outer_iteration']
                or any(not torch.equal(v,loaded.model.state_dict()[k]) for k,v in metadata['average_policy_trainer']['model'].items())):
            raise ValueError('Recovery and exported policy differ')
        regret=RegretPolicy(state)
        report.update(source_policy_sha256=sha256_file(policy),library_sha256=digest(library),
            source_manifest_sha256=sha256_file(Path(state)/'manifest.json'),requested_positions=len(chosen))
        write_json(output/'manifest.json',dict(**{k:report[k] for k in ('source_policy_sha256','library_sha256','source_manifest_sha256')},
            code_hashes={str(p.relative_to(Path(__file__).parents[1])):sha256_file(p)
                for folder in (Path(__file__).parent,Path(__file__).parents[1]/'strategic_audit') for p in folder.glob('*.py')},
            protocol='position_investigation_v1',threads=threads,max_seconds=max_seconds,partition=report['partition']))
        matches,coverage,global_coverage=replay_scan(buffer,metadata,loaded,chosen,check)
        write_json(output/'replay_coverage.json',dict(families=coverage,global_coverage=global_coverage))
        del buffer
        learned_times=[]
        for i,p in enumerate(chosen):
            check(); s,_=build_state(p)
            row=dict(id=p['id'],source=p['source'],family=p['family'],partition=p['partition'],
                encounter_weight=p['encounter_weight'],position=p,replay=matches[i],
                policy_probabilities=probabilities(loaded,s).tolist(),regret_probabilities=probabilities(regret,s).tolist(),
                checks=consistency_checks(loaded,p,check),regret_checks=consistency_checks(regret,p,check))
            with torch.no_grad():
                row['regret_values']=regret.models[p['hero']](torch.from_numpy(regret.encoder.information_state(s,p['hero']))).numpy().tolist()
            if len(p['board'])==5:
                space=RiverSpace(p)
                row['synthetic_panels']={name+'/'+prior:river_audit(space,loaded,name,space.weights(prior),check)
                    for name in ('check_call','value','pressure') for prior in ('uniform','strong','weak')}
                tick=time.monotonic()
                row['p32_opponent']=learned_audit(p,loaded,loaded,check)
                row['regret_against_p32']=learned_audit(p,regret,loaded,check)
                learned_times.append(time.monotonic()-tick)
            report['positions'].append(row)
            report['completed_positions']=len(report['positions'])
            write_json(output/'report.json',report)
            # Descriptive timing pilot, never change the benchmark based on errors.
            if len(learned_times) in (1,4):
                write_json(output/'timing.json',dict(river_pilot_positions=len(learned_times),
                    seconds_per_river=float(np.mean(learned_times)),
                    projected_remaining_river_seconds=sum(len(q['board'])==5 for q in chosen[i+1:])*float(np.mean(learned_times)),
                    note='Timing projection only; fixed benchmark is not shortened silently. Stage guard marks unfinished work incomplete.'))
        report['status']='COMPLETE'
        if not unlock_assessment:
            selected=select_upstream(report['positions'],upstream_positions)
            write_json(output/'upstream_positions.json',dict(positions=selected,library_sha256=digest(library),
                source_policy_sha256=report['source_policy_sha256'],source_manifest_sha256=report['source_manifest_sha256'],
                selection='Half largest P32 conditional gaps, half smallest controls; development only, frozen before traversal checks'))
    except TimeoutError as exc:
        report.update(status='INCOMPLETE',reason=str(exc))
    except BaseException as exc:
        report.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        if 'encounter_report' in locals(): report['ranked_families']=summarize(report['positions'],encounter_report)
        write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('state','policy','encounters','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--threads',type=int,default=1); p.add_argument('--max-seconds',type=int,default=3000)
    p.add_argument('--upstream-positions',type=int,default=24); p.add_argument('--unlock-assessment',action='store_true')
    if run(**vars(p.parse_args()))['status']!='COMPLETE': raise SystemExit(2)

if __name__=='__main__': main()
