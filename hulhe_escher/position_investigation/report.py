"""Compact analytical summary; no automated causal or playing-strength claims."""
import argparse
import html
import json
from pathlib import Path

from ..strategic_audit.workflow import write_json


def render(root):
    root=Path(root)
    def read(path):
        p=root/path
        return json.loads(p.read_text()) if p.exists() else dict(status='MISSING')
    encounters=read('encounters/report.json'); audit=read('analysis/report.json')
    probes=read('upstream/probes/report.json'); iteration=read('upstream/iteration/report.json')
    regression=read('regression/summary.json')
    stages={k:v.get('status','MISSING') for k,v in
            [('encounters',encounters),('analysis',audit),('upstream',probes),('iteration',iteration),('regression',regression)]}
    cloud=read('cloud_status.json')
    comparison=read('comparison/report.json')
    if cloud.get('comparison_requested'):
        stages['comparison']=comparison['status']
    complete=all(s=='COMPLETE' for s in stages.values())
    label=cloud.get('checkpoint_label',audit.get('checkpoint_label','P32'))
    rows=probes.get('positions',[])
    flags=[dict(id=r['id'],fold=f['fold'],actions=[a for a,b in zip(r['legal'],f['reference_outside_interval']) if b])
        for r in rows for f in r.get('folds',[]) if any(f['reference_outside_interval'])]
    summary=dict(status='FAILED' if cloud.get('status')=='FAILED' or 'FAILED' in stages.values()
                 else 'COMPLETE' if complete else 'INCOMPLETE',stages=stages,
        checkpoint_label=label,completed_hands=encounters.get('completed_hands',{}),
        audited_positions=audit.get('completed_positions',0),
        upstream_supported_positions=sum(r['status']=='COMPLETE' for r in rows),
        upstream_unsupported_positions=sum(r['status']=='UNSUPPORTED' for r in rows),
        estimator_flags=flags,ranked_families=audit.get('ranked_families',[]),
        limitations=audit.get('limitations',[])+[
            'Diagnostic flags need replication; finite Monte Carlo error and model-conditioned references do not establish estimator bias.',
            'Low-gap controls and high-gap positions are purposefully selected; no population inference from their comparison.',
            'No new playable policy is selected and no playing-strength improvement is claimed. Assessment remains locked.'])
    if 'reason' in cloud:summary['reason']=cloud['reason']
    if cloud.get('comparison_requested'):
        summary['paired_comparison']=dict(status=comparison['status'],
            completed_positions=comparison.get('completed_positions',0),summaries=comparison.get('summaries',[]),
            report='comparison/report.md')
    write_json(root/'summary.json',summary)
    lines=['# Limit Hold’em position investigation','',f"Status: **{summary['status']}**",'',
        f"Hands: {summary['completed_hands']}; audited development positions: {summary['audited_positions']}; "
        f"supported upstream positions: {summary['upstream_supported_positions']}.",'',
        '## Common decision families','',
        '| Source | Family | Encounter decisions | Audited | Exact replay matches | Conditional local gap, bb |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary['ranked_families'][:30]:
        gap='unscored' if r['mean_local_gap_bb'] is None else f"{r['mean_local_gap_bb']:.5f}"
        lines.append(f"| {r['source']} | {r['family']} | {r['encounters']} | {r['audited_positions']} | {r['exact_replay_matches']} | {gap} |")
    lines += ['', f'These are costs conditional on a frozen {label} opponent and its history-conditioned range. '
        'The frequency ranking is triage, not an additive estimate of whole-game losses. Earlier streets have no oracle score.',
        '', '## Upstream checks','',f"{len(flags)} position/fold combinations have at least one flagged action interval. "
        'Inspect the reference, uncertainty and support before interpreting these as possible estimator discrepancies.',
        '', 'The iteration report contains actual collected regret/critic targets and before/after training errors. '
        'Probe targets separately freeze local strategies before the iteration; they are not assumed to have occurred in replay.',
        '', '## Interpretation','',
        '- Missing exact matches: inspect family coverage; retained replay cannot prove a state was never visited.',
        '- Large training KL: approximation/optimization candidate; low training and larger historical-validation KL: generalization candidate.',
        '- Estimator reference discrepancy: check implementation/conditioning, then replicate with more independent samples.',
        '- Broad estimator intervals: noisy supervision candidate, not proof of biased supervision.',
        '- Good targets but large actual training residuals: investigate regret fitting; synthetic probe changes alone do not establish this.',
        '', *['- '+s for s in summary['limitations']], '']
    if cloud.get('comparison_requested'):
        lines += ['## Paired comparison with P32','',
            'See comparison/report.md for both policies on identical original P32 development positions, '
            'with a shared frozen P32 opponent and posterior. Positive gap reductions mean improvement.', '',
            '| Panel | Positions | Remaining-river gap reduction, bb/decision |', '|---|---:|---:|']
        for row in comparison.get('summaries',[]):
            if row['scope']=='all':lines.append(f"| {row['panel']} | {row['positions']} | {row['weighted']['remaining_gap_reduction_bb']:+.6f} |")
    text='\n'.join(lines); (root/'report.md').write_text(text)
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Position investigation</title>'
        '<style>body{max-width:1100px;margin:40px auto;font:16px system-ui;padding:20px}pre{white-space:pre-wrap}</style>'
        '<pre>'+html.escape(text)+'</pre>')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    render(p.parse_args().root)

if __name__=='__main__': main()
