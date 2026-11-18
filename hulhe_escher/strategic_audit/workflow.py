"""Bounded local multi-checkpoint comparison with immutable provenance and reports."""
import argparse
from datetime import datetime, timezone
import html
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from ..checkpointing import LoadedHULHEPolicy, sha256_file
from ..game import load_hulhe_game, serialisable_game_definition
from ..evaluate import UniformPolicy, AlwaysCallPolicy
from .library import VERSION, build_library, validate_library, digest, card_text, build_state
from .engine import PANELS, RANGES, CachedPolicy, RiverSpace, consistency_checks, probabilities, river_audit

CAVEAT = ('Balanced fixed-position diagnostics in big blinds per decision, not bb/100, '
          'whole-game exploitability, equilibrium certification or a visitation-weighted loss estimate. '
          'Root ranges are stipulated, not inferred from earlier actions. Opponents are synthetic '
          'frozen response models, not Fell Omen 2. Earlier-street actions are descriptive only. '
          'Suit permutations and overlapping decisions are not independent samples; no confidence '
          'intervals are fabricated. Assessment is held out from audit-driven selection, not training.')


class AlwaysRaisePolicy:
    def action_probabilities(self, state):
        legal = state.legal_actions(); selected = 2 if 2 in legal else 1
        return {a:float(a == selected) for a in legal}


class RegretPolicy:
    def __init__(self, state):
        from ..policy_replay_artifacts import read_metadata
        from ..features import encoder_from_metadata
        from ..diagnostic_probes import regret_models
        from types import SimpleNamespace
        manifest, metadata = read_metadata(state)
        if metadata['game'] != serialisable_game_definition():
            raise ValueError('Regret snapshot game mismatch')
        self.encoder = encoder_from_metadata(metadata['encoder'])
        self.models = regret_models(metadata, self.encoder)
        self.settings = SimpleNamespace(**{k:metadata['config'][k] for k in
            ('use_regret_matching_argmax', 'uniform_regret_ties')})
        self.iteration = metadata['solver']['num_iteration']

    def action_probabilities(self, state):
        from vr_deep_cfr.solver import RegretTrainer
        player = state.current_player()
        with torch.no_grad():
            x = torch.as_tensor(self.encoder.information_state(state,player),dtype=torch.float32)
            raw = self.models[player](x).numpy()
        p = RegretTrainer.regret_matching(self.settings,raw,state.legal_actions())
        return {a:float(p[a]) for a in state.legal_actions()}


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def selection(library, partition, smoke):
    positions = [p for p in library['positions'] if p['partition'] == partition]
    if not positions:
        raise ValueError('Selected library partition has no positions')
    if smoke:
        river = [p for p in positions if len(p['board']) == 5]
        context = [next(p for p in positions if len(p['board']) == size) for size in (0,3,4)
                   if any(len(p['board']) == size for p in positions)]
        positions = river[:6]+context
    return positions


def summary(rows, policy):
    own = [r for r in rows if r['policy'] == policy and r.get('oracle') == 'exact_river'
           and r['result']['status'] == 'COMPLETE']
    panels = {}
    for opponent in PANELS:
        for prior in RANGES:
            subset = [r for r in own if r['opponent'] == opponent and r['range'] == prior]
            if not subset: continue
            def metrics(part):
                return dict(positions=len(part),
                    mean_local_gap_bb=float(np.mean([r['result']['root']['local_gap_bb'] for r in part])),
                    mean_remaining_response_gap_bb=float(np.mean([r['result']['root']['remaining_response_gap_bb'] for r in part])))
            panels[f'{opponent}/{prior}'] = dict(**metrics(subset),
                by_seat={str(seat):metrics([r for r in subset if r['hero']==seat]) for seat in (0,1)
                         if any(r['hero']==seat for r in subset)},
                by_family={family:metrics([r for r in subset if r['family']==family])
                           for family in sorted({r['family'] for r in subset})},
                worst_positions=[dict(position=r['position'],local_gap_bb=r['result']['root']['local_gap_bb'])
                    for r in sorted(subset,key=lambda r:r['result']['root']['local_gap_bb'],reverse=True)[:10]])
    # Universal error is independent of the assumed opponent/range: count root once.
    exact = [r for r in own if r['opponent']==PANELS[0] and r['range']==RANGES[0]
             and r['result']['root']['provable_fold_probability'] is not None]
    return dict(panels=panels, provable_unbeatable_call_positions=len(exact),
        mean_provable_fold_probability=float(np.mean([r['result']['root']['provable_fold_probability'] for r in exact])) if exact else None,
        worst_provable_fold_probability=max((r['result']['root']['provable_fold_probability'] for r in exact),default=None),
        worst_provable_positions=[dict(position=r['position'],hole=r['hole'],board=r['board'],
            fold_probability=r['result']['root']['provable_fold_probability'],
            cost_lower_bound_bb=r['result']['root']['provable_fold_cost_lower_bound_bb'])
            for r in sorted(exact,key=lambda r:r['result']['root']['provable_fold_cost_lower_bound_bb'],reverse=True)[:10]])


def render_report(output, report):
    lines=['# Limit Hold’em strategic-position audit','',f"Status: **{report['status']}**",'',CAVEAT,'',
           '| Policy | Opponent / root range | Positions | Local gap (bb) | Remaining-river gap (bb) |',
           '|---|---|---:|---:|---:|']
    chart=[]
    for name, data in report['policies'].items():
        for panel,m in data['panels'].items():
            safe=lambda text:str(text).replace('|','\\|').replace('\n',' ')
            lines.append(f"| {safe(name)} | {panel} | {m['positions']} | {m['mean_local_gap_bb']:.5f} | {m['mean_remaining_response_gap_bb']:.5f} |")
            if panel == 'value/uniform':chart.append((name,m['mean_remaining_response_gap_bb']))
    lines += ['', 'Provably dominated folds (unbeatable hand facing a terminal call):','']
    for name,data in report['policies'].items():
        value=data['mean_provable_fold_probability']
        lines.append(f"- {name}: {data['provable_unbeatable_call_positions']} positions; mean fold probability "
                     + ('unavailable' if value is None else f'{value:.2%}'))
    lines += ['', '| Policy / position | Hole | Board | Fold probability | Guaranteed cost floor (bb/decision) |',
              '|---|---|---|---:|---:|']
    for name,data in report['policies'].items():
        for row in data['worst_provable_positions'][:5]:
            safe_name=str(name).replace('|','\\|').replace('\n',' ')
            lines.append(f"| {safe_name} / {row['position']} | {row['hole']} | {row['board']} | "
                         f"{row['fold_probability']:.2%} | {row['cost_lower_bound_bb']:.4f} |")
    lines += ['', 'See `positions.jsonl` for cards, full betting histories, action probabilities, exact action values,',
              'pot odds, conditional ranges and the largest continuation errors. See `checks.json` for privacy',
              'and all 24 suit permutations. Repeated assessment-driven tuning consumes the assessment holdout.', '']
    (output/'report.md').write_text('\n'.join(lines))
    scale=max((v for _,v in chart),default=1) or 1
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="'+str(45+len(chart)*40)+'" role="img" aria-label="Remaining-river gap against value model, uniform root range">']
    for i,(name,value) in enumerate(chart):
        y=20+i*40
        svg += [f'<text x="8" y="{y+16}" font-size="13">{html.escape(name[:28])}</text>',
                f'<rect x="225" y="{y}" width="{380*value/scale:.2f}" height="24" fill="#3d7a99"/>',
                f'<text x="620" y="{y+16}" font-size="13">{value:.4f} bb</text>']
    svg.append('</svg>'); graphic=''.join(svg)
    (output/'gaps.svg').write_text(graphic)
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Limit Hold’em strategic audit</title>'
        '<style>body{max-width:1100px;margin:40px auto;font:16px system-ui;color:#172a36}pre{white-space:pre-wrap}svg{max-width:100%}</style>'
        '<h1>Limit Hold’em strategic audit</h1><p>'+html.escape(CAVEAT)+'</p><h2>Value opponent / uniform root range</h2>'
        +graphic+'<pre>'+html.escape('\n'.join(lines))+'</pre>')


def run(output, policies, *, library=None, partition='development', unlock_assessment=False,
        smoke=False, max_seconds=600, provenance=None, require_suit_invariance=False):
    if partition not in ('development','assessment') or (partition=='assessment' and not unlock_assessment):
        raise ValueError('Assessment requires explicit --unlock-assessment after freezing model selection')
    if not policies or max_seconds <= 0 or not np.isfinite(max_seconds):
        raise ValueError('Policies and finite positive budget required')
    library=validate_library(build_library() if library is None else library)
    positions=selection(library,partition,smoke)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def deadline():
        if time.monotonic()-start >= max_seconds:raise TimeoutError('Strategic audit time budget exhausted')
    source_hashes={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    # Include reused game, ranker, encoder and inference code, not just the audit wrapper.
    source_hashes.update({str(p.relative_to(Path(__file__).parents[1])):sha256_file(p)
        for p in Path(__file__).parents[1].glob('*.py')})
    manifest=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        library_sha256=digest(library),partition=partition,assessment_unlocked=unlock_assessment,
        smoke=smoke,position_ids=[p['id'] for p in positions],policy_provenance=provenance or {},
        policy_ids=list(policies),opponents=list(PANELS),root_ranges=list(RANGES),
        source_sha256=source_hashes,versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                                                open_spiel=version('open_spiel')),
        threads=torch.get_num_threads(),max_seconds=max_seconds,caveat=CAVEAT)
    write_json(output/'manifest.json',manifest);write_json(output/'library.json',library)
    report=dict(status='RUNNING',manifest_sha256=sha256_file(output/'manifest.json'),policies={},completed_positions=0)
    rows=[];checks=[]
    write_json(output/'summary.json',report)
    try:
        with (output/'positions.jsonl').open('x') as journal:
            for name,policy in policies.items():
                for pos in positions:
                    deadline()
                    check=consistency_checks(policy,pos,deadline)
                    checks.append(dict(policy=name,position=pos['id'],**check))
                    if require_suit_invariance and not check['suit_invariant']:
                        raise ValueError('Policy violates required suit invariance')
                    cached=CachedPolicy(policy,deadline)
                    base=dict(policy=name,position=pos['id'],hero=pos['hero'],family=pos['family'],
                        hole=card_text(pos['hole']),board=card_text(pos['board']),actions=pos['actions'],tags=pos['tags'])
                    if len(pos['board'])!=5:
                        state,_=build_state(pos)
                        records=[dict(**base,oracle='context_only',street={0:'preflop',3:'flop',4:'turn'}[len(pos['board'])],
                            probabilities=probabilities(cached,state).tolist(),legal=state.legal_actions())]
                    else:
                        space=RiverSpace(pos)
                        records=[dict(**base,oracle='exact_river',opponent=opp,range=prior,
                            result=river_audit(space,cached,opp,space.weights(prior),deadline))
                            for opp in PANELS for prior in RANGES]
                    for row in records:
                        journal.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                    journal.flush()
                    report['completed_positions']+=1
                    write_json(output/'checks.json',checks)
                report['policies'][name]=summary(rows,name)
                write_json(output/'summary.json',report)
        report['status']='COMPLETE'
    except BaseException as exc:
        report.update(status='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        report['checks']=dict(position_checks=len(checks),max_suit_error=max((c['suit_max_error'] for c in checks),default=None))
        write_json(output/'summary.json',report);write_json(output/'checks.json',checks)
        render_report(output,report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    lib=commands.add_parser('library');lib.add_argument('--output',type=Path,required=True)
    job=commands.add_parser('run')
    job.add_argument('--policy',action='append',default=[],metavar='LABEL=CHECKPOINT')
    job.add_argument('--regret-state',action='append',default=[],metavar='LABEL=STATE_DIRECTORY')
    job.add_argument('--baseline',action='append',choices=['uniform','check_call','raise'],default=[])
    job.add_argument('--library',type=Path);job.add_argument('--output',type=Path,required=True)
    job.add_argument('--partition',choices=['development','assessment'],default='development')
    job.add_argument('--unlock-assessment',action='store_true');job.add_argument('--smoke',action='store_true')
    job.add_argument('--threads',type=int,default=1);job.add_argument('--max-seconds',type=float,default=600)
    job.add_argument('--require-suit-invariance',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='library':
        payload=build_library();args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(payload,f,indent=2)
        print(json.dumps(dict(positions=len(payload['positions']),sha256=digest(payload))))
        return
    if args.threads<1:parser.error('Positive thread count required')
    torch.set_num_threads(args.threads)
    policies={};provenance={}
    def add(name,policy,source):
        if not name or name in policies:raise ValueError('Nonempty unique policy labels required')
        policies[name]=policy;provenance[name]=source
    for spec in args.policy:
        name,path=spec.split('=',1);path=Path(path).resolve()
        add(name,LoadedHULHEPolicy(load_hulhe_game(),path),dict(type='average_policy',path=str(path),sha256=sha256_file(path)))
    for spec in args.regret_state:
        name,path=spec.split('=',1);path=Path(path).resolve()
        add(name,RegretPolicy(path),dict(type='instantaneous_regret_strategy',path=str(path),manifest_sha256=sha256_file(path/'manifest.json')))
    for name in args.baseline:
        add(name,{'uniform':UniformPolicy,'check_call':AlwaysCallPolicy,'raise':AlwaysRaisePolicy}[name](),dict(type='fixed_baseline',name=name))
    library=json.loads(args.library.read_text()) if args.library else None
    report=run(args.output,policies,library=library,partition=args.partition,
        unlock_assessment=args.unlock_assessment,smoke=args.smoke,max_seconds=args.max_seconds,
        provenance=provenance,require_suit_invariance=args.require_suit_invariance)
    print(json.dumps(dict(status=report['status'],elapsed_seconds=report['elapsed_seconds'],output=str(args.output))))
