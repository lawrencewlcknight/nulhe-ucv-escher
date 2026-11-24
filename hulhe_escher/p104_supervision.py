"""Frozen P104 supervision audit; no fitting, replay writes or policy exports."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from .checkpointing import sha256_file
from .train import write_json
from .supervision_runtime import load_runtime, ARMS
from .supervision_positions import encounters, select_positions, analyse_position
from .supervision_statistics import summarise_position, select, aggregate, scores


_RUNTIME = None
_ERROR = None
_DEADLINE = None


def initialise(state, policy, smoke, deadline):
    global _RUNTIME, _ERROR, _DEADLINE
    _DEADLINE = deadline
    try:
        torch.set_num_threads(1)
        _RUNTIME = load_runtime(state,policy,smoke=smoke)
    except Exception as exc:
        _ERROR = repr(exc)  # Fail in task, rather than endlessly respawning initializers.


def check():
    if _ERROR is not None:
        raise RuntimeError(_ERROR)
    if time.monotonic() >= _DEADLINE:
        raise TimeoutError('Supervision audit stage budget exhausted; sample size not reduced')


def collect_task(task):
    check()
    return encounters(_RUNTIME,*task,check=check)


def analyse_task(task):
    check()
    position, options, output = task
    report, arrays = analyse_position(_RUNTIME,position,check=check,**options)
    output = Path(output)
    artifact = output/(position['id']+'.npz')
    np.savez_compressed(artifact,**arrays)
    report['artifact'] = artifact.name
    report['artifact_sha256'] = sha256_file(artifact)
    if not options.get('pilot',False):
        summary, replicas = summarise_position(report,arrays)
        report['statistics'] = summary
    else:
        replicas = {}
    write_json(output/(position['id']+'.json'),report)
    return report,replicas


def results(pool, function, tasks, deadline):
    iterator = pool.imap_unordered(function,tasks,chunksize=1)
    for _ in tasks:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Audit budget exhausted')
        yield iterator.next(timeout=remaining)


def forecast(pilot, positions, *, workers, hidden, repeats, samples):
    by_street = {}
    for street in (3,4,5):
        values = []
        for row in pilot:
            if len(row['position']['board']) == street:
                stages = row['stage_cpu_seconds']
                values.append(stages['ranges']+stages['reference']*(1 if street==5 else samples/64)
                    +stages['sampling']*hidden*repeats/8)
        if not values:
            raise ValueError('Pilot must cover every street')
        by_street[street] = max(values)  # Conservative seat-specific maximum.
    cpu = sum(by_street[len(p['board'])] for p in positions)
    return dict(estimated_cpu_seconds=cpu, conservative_remaining_wall_seconds=1.5*cpu/workers+300,
                factor=1.5, caveat='Pilot extrapolation, not a runtime guarantee; no silent sample reduction.')


def run(state, policy, output, *, workers=8, max_seconds=6.5*3600, smoke=False):
    state,policy,output = Path(state),Path(policy),Path(output)
    if workers < 1 or max_seconds <= 0:
        raise ValueError('Positive workers and time budget required')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('pilot','positions'):
        (output/name).mkdir()
    started = time.monotonic()
    deadline = started+max_seconds
    before = {p:sha256_file(p) for p in state.rglob('*') if p.is_file()}
    before[policy] = sha256_file(policy)
    status = dict(status='RUNNING',smoke=smoke,workers=workers,
        production_traversals=512 if smoke else 20000,hidden_draws=8 if smoke else 32,
        continuation_repeats=2 if smoke else 8,reference_samples=8 if smoke else 4096,
        source_read_only=True,training_iterations=0,arms=list(ARMS),
        beta_one='Alias of current P104 beta=1; not fitted or sampled twice.',
        objective='Precision and mean correctness of prospective frozen-iteration supervision, not playing strength.')
    write_json(output/'protocol.json',status)
    reports,replicas = [],{}
    pool = mp.get_context('spawn').Pool(workers,initializer=initialise,
        initargs=(state,policy,smoke,deadline))
    try:
        tasks = [(start,min(100,status['production_traversals']-start))
                 for start in range(0,status['production_traversals'],100)]
        encounter_rows = []
        for batch in results(pool,collect_task,tasks,deadline):
            encounter_rows.extend(batch)
        panel = select_positions(encounter_rows,smoke=smoke)
        write_json(output/'panel.json',panel)
        if panel['status'] != 'COMPLETE':
            raise RuntimeError('Insufficient outcome-blind position coverage; inspect panel.json')
        positions = panel['positions']
        pilot_positions = [next(p for p in positions if p['partition']=='development'
                           and len(p['board'])==street and p['hero']==hero)
                           for street in (3,4,5) for hero in (0,1)]
        pilot_tasks = [(p,dict(hidden=4,repeats=2,samples=64,arms=ARMS,pilot=True),output/'pilot')
                       for p in pilot_positions]
        pilot = [r for r,_ in results(pool,analyse_task,pilot_tasks,deadline)]
        projection = forecast(pilot,positions,workers=workers,hidden=status['hidden_draws'],
            repeats=status['continuation_repeats'],samples=status['reference_samples'])
        write_json(output/'runtime_projection.json',projection)
        if not smoke and projection['conservative_remaining_wall_seconds'] > deadline-time.monotonic():
            raise TimeoutError('Pilot projects insufficient remaining budget; full sampling was not started')
        selection = None
        for phase in ('development_validation','assessment'):
            phase_positions = [p for p in positions if (p['partition']=='assessment') == (phase=='assessment')]
            arms = ARMS if selection is None else tuple(dict.fromkeys(('current',selection['selected'])))
            options = dict(hidden=status['hidden_draws'],repeats=status['continuation_repeats'],
                           samples=status['reference_samples'],arms=arms)
            tasks = [(p,options,output/'positions') for p in phase_positions]
            for report,replica in results(pool,analyse_task,tasks,deadline):
                reports.append(report)
                replicas[report['id']] = replica
                write_json(output/'progress.json',dict(completed_positions=len(reports),phase=phase))
            if selection is None:
                selection = select([r for r in reports if r['position']['partition']=='validation'])
                # This durable decision precedes any assessment target calculation.
                write_json(output/'selection.json',selection)
        assessment = [r for r in reports if r['position']['partition']=='assessment']
        primary = aggregate(assessment,replicas,selection['selected'])
        streets = {str(street):aggregate([r for r in assessment if len(r['position']['board'])==street],
                    replicas,selection['selected']) for street in (3,4,5)}
        regression = any(v['estimated_equal_cpu_mse_reduction'] < -.10 for v in streets.values())
        primary['material_point_estimate_street_regression'] = regression
        primary['evidence_for_short_continuation'] &= not regression and not smoke
        families = {}
        for family in sorted({r['position']['family'] for r in assessment}):
            rows = [r for r in assessment if r['position']['family']==family]
            families[family] = dict(positions=len(rows),
                current_equal_cpu_mse_bb2=scores(rows,'current'),
                candidate_equal_cpu_mse_bb2=scores(rows,selection['selected']),
                caveat='Descriptive small subgroup; not an independent confirmatory test.')
        status.update(status='COMPLETE',selection=selection,assessment=primary,by_street=streets,
            by_family=families,
            positions=sorted(reports,key=lambda r:r['id']),runtime_projection=projection,
            conclusion=('Supports a short matched continuation, not a production change.'
                        if primary['evidence_for_short_continuation'] else
                        'Inconclusive or no qualifying improvement; inspect precision, mean discrepancy and variance separately.'))
    except BaseException as exc:
        status.update(status='INCOMPLETE',reason=f'{type(exc).__name__}: {exc}',positions=reports)
        raise
    finally:
        pool.terminate()
        pool.join()
        preserved = all(p.exists() and sha256_file(p)==value for p,value in before.items())
        status.update(source_preserved=preserved,elapsed_seconds=time.monotonic()-started)
        if not preserved:
            status.update(status='FAILED_SOURCE_INTEGRITY')
        write_json(output/'summary.json',status)
        lines = ['# P104 regret-supervision audit','',status['status'],'',
                 status.get('conclusion',status.get('reason','')),'']
        if status['status']=='COMPLETE':
            primary = status['assessment']
            lines += [f"Selected on validation: **{status['selection']['selected']}**.",
                f"Assessment: {primary['positions']} positions in {primary['components']} board components.",
                f"Estimated equal-CPU MSE reduction: **{100*primary['estimated_equal_cpu_mse_reduction']:.1f}%**; "
                f"95% interval [{100*primary['interval95'][0]:.1f}%, {100*primary['interval95'][1]:.1f}%].",'',
                '| Street | MSE reduction | 95% interval |','|---|---:|---|']
            for street,value in status['by_street'].items():
                lines.append(f"| { {'3':'Flop','4':'Turn','5':'River'}[street]} | "
                    f"{100*value['estimated_equal_cpu_mse_reduction']:.1f}% | "
                    f"[{100*value['interval95'][0]:.1f}%, {100*value['interval95'][1]:.1f}%] |")
            lines += ['', 'Subgroups are descriptive. Precision/bias screening can remain unresolved even when variance falls.','']
        lines += ['No training or replacement policy. Detailed measurements are in summary.json; raw diagnostic arrays are optional.']
        (output/'report.md').write_text('\n'.join(lines)+'\n')
    if not status['source_preserved']:
        raise RuntimeError('Source integrity check failed')
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state','policy','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-seconds',type=float,default=6.5*3600)
    p.add_argument('--smoke',action='store_true')
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
