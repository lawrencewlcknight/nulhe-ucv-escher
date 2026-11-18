"""Independent OpenSpiel payoffs, hidden-information choices and audit lifecycle."""
from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hulhe_escher.evaluate import UniformPolicy, AlwaysCallPolicy
from hulhe_escher.checkpointing import LoadedHULHEPolicy, sha256_file
from hulhe_escher.game import load_hulhe_game
from hulhe_escher.strategic_audit.library import (build_library, build_state, validate_library,
                                                canonical_board, digest, replace_cards)
from hulhe_escher.strategic_audit.engine import (RiverSpace, river_audit, probabilities,
    consistency_checks, opponent_probabilities, CachedPolicy)
from hulhe_escher.strategic_audit.workflow import run, main, RegretPolicy
from test_policy_sampling_experiment import frozen_source


@pytest.fixture(scope='module')
def library():
    return build_library()


def position(library,family='dry_top_pair',hero=0,mode='capped',hand=0):
    return next(p for p in library['positions'] if p['id']==f'{family}-{hand}-p{hero}-{mode}')


def test_frozen_coverage_partitions_and_legal_states(library):
    assert digest(library)==digest(build_library())
    assert len(library['positions'])==270
    assert {len(p['board']) for p in library['positions']}=={0,3,4,5}
    assert {p['hero'] for p in library['positions']}=={0,1}
    assert {p['partition'] for p in library['positions']}=={'development','assessment'}
    assert all(not build_state(p)[0].is_terminal() for p in library['positions'])
    bad=deepcopy(library);bad['positions'][1]['partition']='assessment'
    with pytest.raises(ValueError,match='leakage'):validate_library(bad)
    bad=deepcopy(library);bad['positions'][0]['board'][0]=bad['positions'][0]['hole'][0]
    with pytest.raises(ValueError,match='cards'):validate_library(bad)
    bad=deepcopy(library);bad['positions'][0]['actions'][-1]+=[0]
    with pytest.raises(ValueError):validate_library(bad)
    for p in permutations(range(4)):
        assert canonical_board((0,5,10,19,24))==canonical_board(tuple(4*(c//4)+p[c%4] for c in (0,5,10,19,24)))


@pytest.mark.parametrize('hero',[0,1])
@pytest.mark.parametrize('family',['dry_top_pair','board_nuts','four_flush','paired_board'])
def test_exact_terminal_call_values_against_all_990_openspiel_worlds(library,hero,family):
    p=position(library,family,hero)
    space=RiverSpace(p)
    weights=space.weights('strong');weights/=weights.sum()
    actual=river_audit(space,UniformPolicy(),weights=weights)['root']
    calls=[];folds=[]
    for hand in space.hands:
        state,spent=build_state(p,tuple(map(int,hand)))
        assert state.legal_actions()==[0,1] and state.child(1).is_terminal()
        calls.append(state.child(1).returns()[hero]/100)
        folds.append(state.child(0).returns()[hero]/100)
    call,fold=float(weights@calls),float(weights@folds)
    assert actual['q_bb']['1']==pytest.approx(call,abs=1e-10)
    assert actual['q_bb']['0']==pytest.approx(fold,abs=1e-10)
    assert actual['local_gap_bb']==pytest.approx(max(call,fold)-.5*(call+fold),abs=1e-10)
    assert call-fold==pytest.approx(actual['equity']*(actual['pot_bb']+actual['call_cost_bb'])-actual['call_cost_bb'])


def test_maximisation_cannot_see_hidden_hand(library):
    space=RiverSpace(position(library))
    win=np.flatnonzero(space.outcome>0)[0];loss=np.flatnonzero(space.outcome<0)[0]
    weights=np.zeros(990);weights[win]=.01;weights[loss]=.99
    result=river_audit(space,UniformPolicy(),weights=weights)
    root=result['root'];assert root['best_local_actions']==[0]
    clairvoyant=0.
    for idx in (win,loss):
        state,_=build_state(space.position,tuple(map(int,space.hands[idx])))
        clairvoyant+=weights[idx]*max(state.child(a).returns()[space.hero]/100 for a in state.legal_actions())
    assert clairvoyant > result['best_response_value_bb']+.01
    assert result['best_response_value_bb']==pytest.approx(root['q_bb']['0'])


def scalar_reference(space, candidate, opponent, ids, weights):
    """Independent list-of-OpenSpiel-worlds recursion; no ranker or chip arithmetic."""
    states=[build_state(space.position,tuple(map(int,space.hands[i])))[0] for i in ids]
    def recurse(worlds,mass,spent):
        if worlds[0].is_terminal():
            value=sum(w*s.returns()[space.hero]/100 for w,s in zip(mass,worlds))
            return value,value
        actor=worlds[0].current_player();legal=worlds[0].legal_actions();q=[];br=[]
        for action in legal:
            next_mass=np.array(mass)
            if actor!=space.hero:
                next_mass*=opponent_probabilities(opponent,worlds[0],space.strength[ids],spent)[:,action]
            next_spent=spent.copy()
            if action==1:next_spent[actor]=max(spent)
            if action==2:next_spent[actor]=max(spent)+200
            v,b=recurse([s.child(action) for s in worlds],next_mass,next_spent)
            q.append(v);br.append(b)
        if actor!=space.hero:return sum(q),sum(br)
        p=probabilities(candidate,worlds[0])
        return float(p[legal]@q),max(br)
    return recurse(states,weights,space.spent)


@pytest.mark.parametrize('hero',[0,1])
@pytest.mark.parametrize('opponent',['check_call','value','pressure'])
def test_complete_river_tree_matches_independent_scalar_worlds(library,hero,opponent):
    space=RiverSpace(position(library,hero=hero,mode='open'))
    ids=np.array([0,75,200,490,700,989]);weights=np.array([1,2,4,8,2,3],float);weights/=weights.sum()
    full=np.zeros(990);full[ids]=weights
    result=river_audit(space,UniformPolicy(),opponent,full)
    value,best=scalar_reference(space,UniformPolicy(),opponent,ids,weights)
    assert result['candidate_value_bb']==pytest.approx(value,abs=1e-10)
    assert result['best_response_value_bb']==pytest.approx(best,abs=1e-10)
    assert result['root']['remaining_response_gap_bb']>=result['root']['local_gap_bb']-1e-12


def test_structural_and_range_conditioned_errors_are_separate(library):
    nuts=RiverSpace(position(library,'board_nuts'))
    row=river_audit(nuts,UniformPolicy())['root']
    assert row['globally_unbeatable'] and row['provable_fold_probability']==.5
    assert row['equity']==pytest.approx(.5) and row['best_local_actions']==[1]
    assert row['provable_fold_cost_lower_bound_bb']==pytest.approx(row['local_gap_bb'])
    assert row['range_losing_call_probability'] is None
    space=RiverSpace(position(library))
    weights=(space.outcome<0).astype(float)
    row=river_audit(space,UniformPolicy(),weights=weights)['root']
    assert not row['globally_unbeatable'] and row['provable_fold_probability'] is None
    assert row['range_losing_call_probability']==.5
    assert river_audit(space,UniformPolicy(),weights=np.zeros(990))['status']=='UNSUPPORTED'
    with pytest.raises(ValueError):river_audit(space,UniformPolicy(),weights=-np.ones(990))
    early=next(p for p in library['positions'] if len(p['board'])==3)
    with pytest.raises(ValueError,match='river'):RiverSpace(early)


def test_policy_contracts_catch_hidden_information_and_bad_probabilities(library):
    p=position(library)
    result=consistency_checks(UniformPolicy(),p)
    assert result['suit_invariant'] and result['suit_permutations']==24
    class Leaky:
        def action_probabilities(self,s):
            selected=s.history()[2*(1-s.current_player())] % 2
            return {a:float(a==selected) for a in s.legal_actions()}
    with pytest.raises(ValueError,match='contract'):consistency_checks(Leaky(),p)
    class Broken:
        def action_probabilities(self,s):return {1:float('nan')}
    with pytest.raises(ValueError,match='probabilities'):consistency_checks(Broken(),p)
    class SuitBiased:
        def action_probabilities(self,s):
            selected=s.history()[2*s.current_player()] % 2
            return {a:float(a==selected) for a in s.legal_actions()}
    assert not consistency_checks(SuitBiased(),p)['suit_invariant']


def test_run_is_reproducible_readonly_and_reports_context_without_scores(tmp_path,library):
    np_before=deepcopy(np.random.get_state());torch_before=torch.get_rng_state().clone()
    for name in ('a','b'):
        result=run(tmp_path/name,{'uniform':UniformPolicy(),'check_call':AlwaysCallPolicy()},library=library,smoke=True)
        assert result['status']=='COMPLETE' and result['completed_positions']==18
        assert result['policies']['check_call']['mean_provable_fold_probability']==0
        assert (tmp_path/name/'report.html').exists() and (tmp_path/name/'gaps.svg').exists()
    assert (tmp_path/'a/positions.jsonl').read_bytes()==(tmp_path/'b/positions.jsonl').read_bytes()
    np.testing.assert_equal(np_before,np.random.get_state());assert torch.equal(torch_before,torch.get_rng_state())
    rows=[json.loads(s) for s in (tmp_path/'a/positions.jsonl').read_text().splitlines()]
    early=[r for r in rows if r['oracle']=='context_only']
    assert len(early)==6 and all('result' not in r for r in early)
    with pytest.raises(FileExistsError):run(tmp_path/'a',{'u':UniformPolicy()},smoke=True)
    with pytest.raises(ValueError,match='unlock'):run(tmp_path/'locked',{'u':UniformPolicy()},partition='assessment')
    assert not (tmp_path/'locked').exists()
    result=run(tmp_path/'unlocked',{'u':UniformPolicy()},partition='assessment',unlock_assessment=True,smoke=True)
    assert result['status']=='COMPLETE'
    assert json.loads((tmp_path/'unlocked/manifest.json').read_text())['assessment_unlocked']


def test_timeout_and_bad_policy_never_publish_success(tmp_path):
    with pytest.raises(TimeoutError):run(tmp_path/'timeout',{'u':UniformPolicy()},smoke=True,max_seconds=1e-12)
    assert json.loads((tmp_path/'timeout/summary.json').read_text())['status']=='INCOMPLETE'
    class Broken:
        def action_probabilities(self,s):raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError):run(tmp_path/'bad',{'broken':Broken()},smoke=True)
    assert json.loads((tmp_path/'bad/summary.json').read_text())['status']=='FAILED'


def test_custom_library_empty_partition_rejected_and_river_only_smoke_allowed(tmp_path,library):
    custom=deepcopy(library);custom['positions']=[position(library)]
    report=run(tmp_path/'custom',{'u':UniformPolicy()},library=custom,smoke=True)
    assert report['completed_positions']==1
    with pytest.raises(ValueError,match='no positions'):
        run(tmp_path/'empty',{'u':UniformPolicy()},library=custom,partition='assessment',unlock_assessment=True)
    assert not (tmp_path/'empty').exists()


def test_real_checkpoint_cli_preserves_source(frozen_source,tmp_path,capsys):
    state,policy=frozen_source
    before=sha256_file(policy)
    main(['run','--policy',f'checkpoint={policy}','--baseline','check_call',
          '--smoke','--output',str(tmp_path/'cli'),'--threads','1'])
    report=json.loads((tmp_path/'cli/summary.json').read_text())
    assert report['status']=='COMPLETE' and sha256_file(policy)==before
    manifest=json.loads((tmp_path/'cli/manifest.json').read_text())
    assert manifest['policy_provenance']['checkpoint']['sha256']==before
    capsys.readouterr()


def test_instantaneous_regret_adapter_preserves_metadata(frozen_source,tmp_path):
    from test_bottleneck import add_regrets
    state,_=frozen_source;add_regrets(state)
    before=sha256_file(state/'manifest.json')
    policy=RegretPolicy(state)
    result=run(tmp_path/'regret',{'instantaneous':policy},smoke=True,max_seconds=60)
    assert result['status']=='COMPLETE' and sha256_file(state/'manifest.json')==before
