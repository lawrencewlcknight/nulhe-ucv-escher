"""Paired checkpoint diagnostics on an immutable development position library."""
import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
import time

import numpy as np
import torch

from ..checkpointing import LoadedHULHEPolicy, sha256_file
from ..game import load_hulhe_game
from ..strategic_audit.engine import RiverSpace, probabilities, river_audit, consistency_checks
from ..strategic_audit.library import build_state, validate_library, digest
from ..strategic_audit.workflow import write_json
from .common import deadline_after
from .oracle import LearnedOpponent


def paired_values(baseline, candidate):
    """Positive changes mean improvement; the full-response reference is fixed."""
    if baseline['status']!='COMPLETE' or candidate['status']!='COMPLETE':
        return dict(status='UNSUPPORTED')
    a,b=baseline['root'],candidate['root']
    if not np.isclose(a['best_response_value_bb'],b['best_response_value_bb'],atol=1e-8,rtol=1e-9):
        raise ValueError('Paired policies do not share the same best-response reference')
    # Local-gap changes alone can be misleading when later hero actions change.
    # Also hold baseline continuation Q fixed to isolate the current action mixture.
    root_change=sum((b['probabilities'][int(k)]-a['probabilities'][int(k)])*q for k,q in a['q_bb'].items())
    return dict(status='COMPLETE',baseline_local_gap_bb=a['local_gap_bb'],candidate_local_gap_bb=b['local_gap_bb'],
        local_gap_reduction_bb=a['local_gap_bb']-b['local_gap_bb'],
        baseline_remaining_gap_bb=a['remaining_response_gap_bb'],candidate_remaining_gap_bb=b['remaining_response_gap_bb'],
        remaining_gap_reduction_bb=a['remaining_response_gap_bb']-b['remaining_response_gap_bb'],
        value_improvement_bb=b['candidate_value_bb']-a['candidate_value_bb'],
        root_action_improvement_fixed_baseline_continuation_bb=float(root_change))


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        for panel,result in row.get('panels',{}).items():
            if result['change']['status']!='COMPLETE':continue
            for scope in ('all', 'source:'+row['source'], 'seat:'+str(row['position']['hero']),
                          'family:'+row['family'], 'source_family:'+row['source']+'|'+row['family']):
                groups[(panel,scope)].append((row,result['change']))
    summaries=[]
    for (panel,scope),items in sorted(groups.items()):
        weights=np.asarray([r['encounter_weight'] for r,c in items],dtype=float)
        metrics={k:float(np.average([c[k] for r,c in items],weights=weights))
                 for k in items[0][1] if k!='status'}
        delta=np.asarray([c['remaining_gap_reduction_bb'] for r,c in items])
        summaries.append(dict(panel=panel,scope=scope,positions=len(items),
            improved=int((delta>1e-8).sum()),regressed=int((delta < -1e-8).sum()),
            unchanged=int((np.abs(delta)<=1e-8).sum()),weighted=metrics,
            unweighted_remaining_gap_reduction_bb=float(delta.mean())))
    return summaries


def render(output,report):
    baseline_label=report.get('baseline_label','P32')
    lines=['# Paired strategic-position comparison','',f"Status: **{report['status']}**",'',
        f"{report['candidate_label']} versus {baseline_label} on the frozen {baseline_label} development positions.",
        'Positive reductions mean improvement. All values are bb per decision, not bb/100.',
        'Both policies face the same frozen opponent and root range. Assessment remains locked.', '',
        f'| Panel | Positions | {baseline_label} remaining gap | Candidate remaining gap | Reduction | Improved / regressed |',
        '|---|---:|---:|---:|---:|---|']
    for row in report['summaries']:
        if row['scope']!='all':continue
        m=row['weighted']
        lines.append(f"| {row['panel']} | {row['positions']} | {m['baseline_remaining_gap_bb']:.6f} | "
            f"{m['candidate_remaining_gap_bb']:.6f} | {m['remaining_gap_reduction_bb']:+.6f} | {row['improved']} / {row['regressed']} |")
    lines += ['', f'## Families against frozen {baseline_label}', '',
        f'| Family | Positions | {baseline_label} remaining gap | Candidate remaining gap | Reduction |',
        '|---|---:|---:|---:|---:|']
    for row in report['summaries']:
        if row['panel']!=f'frozen_{baseline_label}' or not row['scope'].startswith('family:'):continue
        m=row['weighted']
        lines.append(f"| {row['scope'][7:]} | {row['positions']} | {m['baseline_remaining_gap_bb']:.6f} | "
                     f"{m['candidate_remaining_gap_bb']:.6f} | {m['remaining_gap_reduction_bb']:+.6f} |")
    lines += ['', *['- '+x for x in report['limitations']], '']
    (output/'report.md').write_text('\n'.join(lines))


def run(baseline_policy,candidate_policy,encounters,output,*,expected_baseline_sha256,
        expected_candidate_sha256,expected_library_sha256,candidate_label='P56',max_seconds=1800,baseline_analysis=None,baseline_label='P32'):
    if not all(re.fullmatch(r'P[0-9]+',label) for label in (baseline_label,candidate_label)):
        raise ValueError('Checkpoint labels must be P followed by active hours')
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();check=deadline_after(max_seconds)
    report=dict(status='RUNNING',candidate_label=candidate_label,baseline_label=baseline_label,positions=[],summaries=[],
        limitations=[
            'Paired descriptive changes on frozen development positions, not a confirmatory population test; one training seed.',
            'Encounter inverse-inclusion weights are retained on the development subset; overlapping decisions and card orbits are correlated.',
            'No confidence intervals or whole-game loss estimates are inferred from these position averages.',
            'Remaining-river response gap uses the same best response for both candidates; local gap also depends on candidate continuation.',
            f'Fixed-baseline-continuation root action improvement isolates the current action mixture using {baseline_label} continuation values.',
            'Earlier streets report action changes only; no oracle or improvement sign is assigned.',
            'This compares fixed exported policies from one training run; it does not isolate individual causes of any change.'])
    try:
        for path,expected in ((baseline_policy,expected_baseline_sha256),(candidate_policy,expected_candidate_sha256)):
            if sha256_file(path)!=expected:raise ValueError('Policy checksum mismatch')
        encounters=Path(encounters)
        original=json.loads((encounters/'report.json').read_text())
        library=validate_library(json.loads((encounters/'library.json').read_text()))
        if (original['status']!='COMPLETE' or original['source_policy_sha256']!=expected_baseline_sha256
                or original['library_sha256']!=digest(library) or digest(library)!=expected_library_sha256):
            raise ValueError('Frozen baseline library provenance mismatch')
        positions=[p for p in library['positions'] if p['partition']=='development']
        if not positions:raise ValueError('No development positions')
        reference={}
        if baseline_analysis:
            saved=json.loads(Path(baseline_analysis).read_text())
            if (saved['status']!='COMPLETE' or saved['source_policy_sha256']!=expected_baseline_sha256
                    or saved['library_sha256']!=digest(library)
                    or saved.get('learned_opponent_label','P32')!=baseline_label):
                raise ValueError('Original analysis provenance mismatch')
            reference={row['id']:row for row in saved['positions']}
            if set(reference)!={p['id'] for p in positions}:raise ValueError('Original analysis position set mismatch')
            report.update(baseline_analysis_sha256=sha256_file(baseline_analysis),baseline_reproduction_max_abs_error=0.)
        baseline=LoadedHULHEPolicy(load_hulhe_game(),baseline_policy)
        candidate=LoadedHULHEPolicy(load_hulhe_game(),candidate_policy)
        report.update(baseline_policy_sha256=expected_baseline_sha256,candidate_policy_sha256=expected_candidate_sha256,
            library_sha256=digest(library),requested_positions=len(positions),partition='development')
        write_json(output/'manifest.json',dict(protocol='paired_position_comparison_v1',
            baseline_policy_sha256=expected_baseline_sha256,candidate_policy_sha256=expected_candidate_sha256,
            library_sha256=digest(library),position_ids=[p['id'] for p in positions],max_seconds=max_seconds,
            baseline_label=baseline_label,candidate_label=candidate_label,
            learned_opponent=baseline_label,assessment_unlocked=False,
            code_sha256={str(p.relative_to(Path(__file__).parents[1])):sha256_file(p)
                for folder in (Path(__file__).parent,Path(__file__).parents[1]/'strategic_audit') for p in folder.glob('*.py')}))
        for p in positions:
            check();state,_=build_state(p)
            pa,pb=probabilities(baseline,state),probabilities(candidate,state)
            row=dict(id=p['id'],source=p['source'],family=p['family'],position=p,
                encounter_weight=p['encounter_weight'],baseline_probabilities=pa.tolist(),candidate_probabilities=pb.tolist(),
                action_l1=float(np.abs(pa-pb).sum()),candidate_checks=consistency_checks(candidate,p,check),panels={})
            if len(p['board'])==5:
                space=RiverSpace(p);opponent=LearnedOpponent(space,baseline,check);weights=opponent.posterior()
                references=[(f'frozen_{baseline_label}',opponent,weights)]
                references += [(name+'/'+prior,name,space.weights(prior))
                    for name in ('check_call','value','pressure') for prior in ('uniform','strong','weak')]
                for name,response,weights in references:
                    check()
                    a=river_audit(space,baseline,response,weights,check)
                    b=river_audit(space,candidate,response,weights,check)
                    if reference:
                        old=(reference[p['id']]['p32_opponent'] if name==f'frozen_{baseline_label}'
                             else reference[p['id']]['synthetic_panels'][name])
                        if a['status']!=old['status']:raise ValueError('Original baseline support did not reproduce')
                        if a['status']=='COMPLETE':
                            for key in ('candidate_value_bb','remaining_response_gap_bb','local_gap_bb'):
                                error=abs(a['root'][key]-old['root'][key])
                                report['baseline_reproduction_max_abs_error']=max(error,report['baseline_reproduction_max_abs_error'])
                                if not np.isclose(a['root'][key],old['root'][key],atol=2e-5,rtol=1e-6):
                                    raise ValueError(f'Original baseline did not reproduce: {p["id"]}, {name}, {key}')
                    row['panels'][name]=dict(baseline=a,candidate=b,change=paired_values(a,b))
            report['positions'].append(row)
            report['completed_positions']=len(report['positions'])
            write_json(output/'report.json',report)
        report['status']='COMPLETE'
    except TimeoutError as exc:report.update(status='INCOMPLETE',reason=str(exc))
    except BaseException as exc:
        report.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['summaries']=summarize(report['positions'])
        write_json(output/'report.json',report);render(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline-policy','candidate-policy','encounters','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('expected-baseline-sha256','expected-candidate-sha256','expected-library-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--candidate-label',default='P56');p.add_argument('--max-seconds',type=int,default=1800)
    p.add_argument('--baseline-analysis',type=Path)
    p.add_argument('--baseline-label',default='P32')
    if run(**vars(p.parse_args()))['status']!='COMPLETE':raise SystemExit(2)


if __name__=='__main__':main()
