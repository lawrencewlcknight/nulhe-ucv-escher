"""Bounded deterministic visitation sampling; no learner updates or match claims."""
import argparse
from collections import Counter, defaultdict
import heapq
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .. import learning_evaluate as players
from ..checkpointing import sha256_file
from ..fo2_setup import verify_build
from ..game import serialisable_game_definition
from ..strategic_audit.engine import probabilities
from ..strategic_audit.library import VERSION, digest, validate_library
from ..strategic_audit.workflow import write_json
from .common import visible_position, families, partition, priority, lock_partitions


def batch(task):
    seed, source, start, count = task
    if players._initialisation_error:
        raise RuntimeError(players._initialisation_error)
    game, policies = players._game, players._policies
    encoder = policies['P32'].feature_encoder
    rows, adaptations = [], 0
    for hand in range(start, start+count):
        stream = 0 if source == 'self' else 1
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream, hand]))
        deal = iter(rng.choice(52, 9, replace=False).tolist())
        state = game.new_initial_state()
        hero_seat = hand % 2
        decision = 0
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(next(deal)); continue
            actor = state.current_player()
            name = 'P32' if source == 'self' or actor == hero_seat else 'opponent'
            p = probabilities(policies[name], state)
            if name == 'opponent':
                adaptations += (getattr(policies[name], 'last_diagnostics', None) or {}).get('legalization') == 'explicit_free_fold_to_check'
            if name == 'P32':
                position = visible_position(state)
                family = families(encoder.information_state(state, actor), encoder)[0]
                rows.append(dict(**position, id=f'{source}-{hand}-{decision}', source=source,
                    hand=hand, decision=decision, family=family, tags=[family],
                    priority=priority(seed,source,hand,decision),
                    stratum=f'{source}/{len(position["board"])}/p{actor}'))
            state.apply_action(int(rng.choice(3, p=p)))
            decision += 1
    return dict(rows=rows, hands=count, source=source, adaptations=int(adaptations))


class Reservoir:
    """Bottom-k independent event hashes; merge order and worker count do not matter."""
    def __init__(self, capacity=512):
        self.capacity = capacity
        self.heaps = defaultdict(list)
        self.strata = Counter()
        self.families = Counter()

    def add(self, row):
        key = row['stratum']
        self.strata[key] += 1
        self.families[row['source']+'|'+row['family']] += 1
        item = (-int(row['priority'],16), row['id'], row)
        heap = self.heaps[key]
        if len(heap) < self.capacity:
            heapq.heappush(heap,item)
        elif item > heap[0]:
            heapq.heapreplace(heap,item)

    def library(self, river=128, earlier=128):
        quotas = {0:earlier//3, 3:earlier//3, 4:earlier-2*(earlier//3), 5:river}
        positions = []
        shortages = {}
        for size, quota in quotas.items():
            keys = sorted(k for k in self.heaps if int(k.split('/')[1]) == size)
            if not keys:
                shortages[str(size)] = quota; continue
            for index,key in enumerate(keys):
                n = quota//len(keys)+(index < quota%len(keys))
                chosen = sorted((r for _,_,r in self.heaps[key]),key=lambda r:r['priority'])[:n]
                if len(chosen)<n: shortages[key] = n-len(chosen)
                for row in chosen:
                    row = dict(row, partition=partition(row),
                               encounter_weight=self.strata[key]/len(chosen))
                    positions.append(row)
        return validate_library(dict(version=VERSION, game=serialisable_game_definition(),
            positions=lock_partitions(positions), weighting='Uniform events within source/street/seat; encounter_weight=N_stratum/n_selected. '
                'Repeated information states remain separate encounter events. Not independent statistical observations.',
            shortages=shortages, partition_note='Flop orbit held together across later streets; 25% hash assessment, never scored by default.'))


def run(policy, output, *, assets=None, opponent='fo2', hands=20000, workers=8,
        seed=481701, max_seconds=2400, river=128, earlier=128):
    if (hands < 2 or hands%2 or workers < 1 or seed < 0 or min(river,earlier)<1
            or not np.isfinite(max_seconds) or max_seconds<=0):
        raise ValueError('Even positive hands per matchup, positive budgets and workers required')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    report=dict(status='COLLECTING', requested_hands_per_source=hands, completed_hands={},
        source_policy_sha256=sha256_file(policy), seed=seed, workers=workers, opponent=opponent,
        selection='Frozen before error measurement; bottom-k event hashes within source/street/seat',
        smoke=opponent!='fo2', adaptations=0)
    write_json(output/'report.json', report)
    specs={'P32':dict(kind='checkpoint',checkpoint=str(Path(policy).resolve())),
           'opponent':dict(kind=opponent,assets=str(Path(assets).resolve()) if assets else '')}
    started=time.monotonic(); end=started+max_seconds; reservoir=Reservoir()
    try:
        if opponent == 'fo2':
            report['fo2_build']=verify_build(Path(assets))
            write_json(output/'report.json',report)
        with mp.get_context('spawn').Pool(workers,initializer=players.initialise,initargs=(specs,)) as pool:
            # Equal source exposure; bounded dispatch prevents unbounded queued work.
            for start in range(0,hands,workers*16):
                tasks=[(seed,source,i,min(16,hands-i)) for source in ('self','opponent')
                       for i in range(start,min(hands,start+workers*16),16)]
                pending=pool.imap(batch,tasks)
                for _ in tasks:
                    try: result=pending.next(timeout=max(.001,end-time.monotonic()))
                    except mp.TimeoutError: raise TimeoutError('Encounter collection time guard')
                    for row in result['rows']: reservoir.add(row)
                    source=result['source']
                    report['completed_hands'][source]=report['completed_hands'].get(source,0)+result['hands']
                    report['adaptations']+=result['adaptations']
                write_json(output/'report.json',report)
                if time.monotonic()>=end: raise TimeoutError('Encounter collection time guard')
        if opponent == 'fo2' and verify_build(Path(assets))!=report['fo2_build']:
            raise ValueError('FO2 assets changed during collection')
        library=reservoir.library(river,earlier)
        write_json(output/'library.json',library)
        report.update(status='COMPLETE' if not library['shortages'] else 'INCOMPLETE_COVERAGE',
            library_sha256=digest(library), selected_positions=len(library['positions']),
            development=sum(p['partition']=='development' for p in library['positions']))
    except TimeoutError as exc:
        report.update(status='INCOMPLETE',reason=str(exc))
    except BaseException as exc:
        report.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
            encounter_counts=dict(reservoir.strata), family_counts=dict(reservoir.families))
        write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('policy','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--assets',type=Path); p.add_argument('--opponent',choices=['fo2','check_call','random'],default='fo2')
    for n,d in [('hands',20000),('workers',8),('seed',481701),('max-seconds',2400),('river',128),('earlier',128)]:
        p.add_argument('--'+n,type=int,default=d)
    if run(**vars(p.parse_args()))['status']!='COMPLETE': raise SystemExit(2)

if __name__=='__main__': main()
