"""Independent conditional-value, partition, replay and archived-solver checks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from hulhe_escher.checkpointing import LoadedHULHEPolicy,sha256_file
from hulhe_escher.evaluate import AlwaysCallPolicy,UniformPolicy
from hulhe_escher.game import load_hulhe_game
from hulhe_escher.position_investigation.common import visible_position,partition,families,priority,lock_partitions
from hulhe_escher.position_investigation.collect import Reservoir,run as collect
from hulhe_escher.position_investigation.oracle import LearnedOpponent,with_hand,learned_audit
from hulhe_escher.position_investigation.analyse import replay_scan,run as analyse
from hulhe_escher.position_investigation.upstream import mean_diagnostic,read_only_traversal
from hulhe_escher.strategic_audit.engine import RiverSpace,river_audit
from hulhe_escher.strategic_audit.library import build_library,build_state,validate_library,digest
from hulhe_escher.features import make_feature_encoder
from test_policy_sampling_experiment import frozen_source


def position(mode='capped',hero=0):
    return next(p for p in build_library()['positions'] if p['family']=='dry_top_pair' and
                p['hero']==hero and p['id'].endswith(mode))


@pytest.mark.parametrize('hero',[0,1])
def test_visible_serialization_never_keeps_other_cards(hero):
    p=position(hero=hero);s,_=build_state(p)
    visible=visible_position(s)
    assert visible=={k:p[k] for k in ('hero','hole','board','actions')}
    s2,_=build_state(p,opponent=[c for c in range(52) if c not in p['hole']+p['board']][-2:])
    assert visible_position(s2)==visible
    assert build_state(visible)[0].information_state_string()==s.information_state_string()


def test_orbit_partition_holds_flop_extensions_and_suits_together():
    p=position();q=deepcopy(p);q['board']=q['board'][:3]
    r=deepcopy(p);r['board']=r['board'][:4]
    assert partition(p)==partition(q)==partition(r)
    for row in (p,q,r):
        row['board']=[4*(c//4)+(c%4+1)%4 for c in row['board']]
        row['hole']=[4*(c//4)+(c%4+1)%4 for c in row['hole']]
        assert partition(row)==partition(p)


def test_event_reservoir_is_deterministic_across_merge_orders():
    base=position();rows=[]
    for i in range(100):
        rows.append(dict(base,id=f'event-{i}',source='self',family='river',
            stratum='self/5/p0',priority=priority(10,'self',i,0)))
    a,b=Reservoir(10),Reservoir(10)
    for row in rows:a.add(row)
    for row in rows[::-1]:b.add(row)
    assert sorted(x[1] for x in a.heaps['self/5/p0'])==sorted(x[1] for x in b.heaps['self/5/p0'])
    assert a.strata==b.strata and a.families==b.families


class OwnRankPolicy:
    def action_probabilities(self,state):
        raw=np.asarray(state.information_state_tensor(state.current_player()))
        hole=np.flatnonzero(raw[2:54])
        strength=(max(hole)//4+1)/14
        p=np.array([.1,.2+strength,.1+.5*strength])
        p[[a for a in range(3) if a not in state.legal_actions()]]=0
        p/=p.sum()
        return {a:float(p[a]) for a in state.legal_actions()}


@pytest.mark.parametrize('hero',[0,1])
def test_learned_posterior_matches_independent_history_likelihood(hero):
    space=RiverSpace(position(hero=hero));policy=OwnRankPolicy()
    actual=LearnedOpponent(space,policy).posterior()
    expected=[]
    for hand in space.hands:
        root,_=build_state(space.position,opponent=list(map(int,hand)))
        replay=load_hulhe_game().new_initial_state();weight=1.
        for a in root.history():
            if not replay.is_chance_node() and replay.current_player()!=hero:
                weight*=policy.action_probabilities(replay)[a]
            replay.apply_action(a)
        expected.append(weight)
    expected=np.asarray(expected);expected/=expected.sum()
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
    assert not np.allclose(actual,np.full(990,1/990))


def test_zero_history_support_is_not_replaced_with_uniform():
    space=RiverSpace(position())
    response=LearnedOpponent(space,AlwaysCallPolicy())
    assert response.posterior().sum()==0 # History contains raises.
    assert learned_audit(space.position,UniformPolicy(),AlwaysCallPolicy())['status']=='UNSUPPORTED'


@pytest.mark.parametrize('hero',[0,1])
def test_learned_oracle_matches_scalar_open_spiel_values(hero):
    p=position(hero=hero);space=RiverSpace(p)
    opponent=OwnRankPolicy();candidate=UniformPolicy()
    response=LearnedOpponent(space,opponent);weights=response.posterior()
    result=river_audit(space,candidate,response,weights)
    # Capped terminal decisions independently integrate native terminal payoffs.
    for action in space.state.legal_actions():
        native=[]
        for hand in space.hands:
            s,_=build_state(p,opponent=list(map(int,hand)))
            child=s.child(action);assert child.is_terminal()
            native.append(child.returns()[hero]/100)
        assert result['root']['q_bb'][str(action)]==pytest.approx(weights@native,abs=1e-10)


def test_invalid_learned_response_rejected():
    space=RiverSpace(position('open',1))
    with pytest.raises(ValueError,match='Invalid conditional'):
        river_audit(space,UniformPolicy(),lambda s:np.zeros((990,3)))


def test_exact_replay_matches_weighted_target_without_neighbour_imputation(frozen_source):
    from types import SimpleNamespace
    from hulhe_escher.policy_replay_artifacts import read_metadata
    state,policy_path=frozen_source
    policy=LoadedHULHEPolicy(load_hulhe_game(),policy_path);_,metadata=read_metadata(state)
    p=position();q=deepcopy(p);q['hole']=[1,4]
    encoder=policy.feature_encoder
    x=encoder.information_state(build_state(p)[0],p['hero'])
    y=np.array([[.1,.9,0],[.8,.2,0]],np.float32)
    buffer=SimpleNamespace(infostate_buf=np.stack([x,x]),q_value_buf=y,
        q_value_mask_buf=np.array([[1,1,0]]*2,np.float32),iteration_buf=np.array([[1],[10]],np.float32),
        cur_id=2,buffer_size=2,action_size=3)
    matches,summary,coverage=replay_scan(buffer,metadata,policy,[p,q])
    w=(np.array([1,10])/10*2)**metadata['config']['gamma']
    np.testing.assert_allclose(matches[0]['target'],(w[:,None]*y).sum(0)/w.sum(),rtol=1e-6)
    assert matches[0]['retained_rows']==2 and matches[1] is None
    assert sum(r['retained_rows'] for r in summary.values())==2
    assert families(x,encoder)[0].startswith('river/p0/')


def test_mean_diagnostic_detects_controlled_offset_and_reports_uncertainty():
    rng=np.random.default_rng(1);samples=rng.normal(0,.1,(10000,2))
    good=mean_diagnostic(samples,[0,0],.01)
    assert not any(good['reference_outside_interval'])
    bad=mean_diagnostic(samples+1,[0,0],.01)
    assert all(bad['reference_outside_interval'])
    assert min(bad['standard_error'])>0


@pytest.fixture(scope='module')
def recovery(tmp_path_factory):
    output=tmp_path_factory.mktemp('position-recovery')/'training'
    repo=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','hulhe_escher.train','--smoke','--seed','53','--threads','1',
        '--policy-every','1','--checkpoint-every','1','--recovery-seconds','0','--output',str(output),'--iterations','1'],
        cwd=repo,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-2000:]
    return output


def test_real_collection_analysis_preserves_source_and_locks_assessment(recovery,tmp_path):
    state=recovery/'state_000001';policy=recovery/'policy_000001.pkl'
    before=sha256_file(state/'manifest.json'),sha256_file(policy)
    encounter=tmp_path/'encounters'
    result=collect(policy,encounter,opponent='check_call',hands=100,workers=1,max_seconds=90,river=8,earlier=12)
    assert result['status']=='COMPLETE' and result['completed_hands']==dict(self=100,opponent=100)
    library=validate_library(json.loads((encounter/'library.json').read_text()))
    assert len(library['positions'])==20
    report=analyse(state,policy,encounter,tmp_path/'analysis',threads=1,max_seconds=120,upstream_positions=2)
    assert report['status']=='COMPLETE'
    assert all(r['partition']=='development' for r in report['positions'])
    assert all('board' not in r or len(r['board'])<=5 for r in report['positions'])
    assert before==(sha256_file(state/'manifest.json'),sha256_file(policy))
    # Tampering with a frozen library must be detected before any scoring.
    library['positions'][0]['tags'].append('changed')
    (encounter/'library.json').write_text(json.dumps(library))
    with pytest.raises(ValueError,match='changed'):
        analyse(state,policy,encounter,tmp_path/'tampered')


def test_archived_position_probes_and_actual_iteration_targets(recovery,tmp_path):
    repo=Path(__file__).resolve().parents[1];state=recovery/'state_000001';policy=recovery/'policy_000001.pkl'
    before={str(p):sha256_file(p) for p in state.rglob('*') if p.is_file()}
    # Use a low-action-probability but full-support initial policy history.
    p=dict(position(),hero=1,actions=[[1,1],[1,1],[1,1],[]],partition='development')
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(source_policy_sha256=sha256_file(policy),
        source_manifest_sha256=sha256_file(state/'manifest.json'),positions=[dict(position=p,selection_group='control')])) )
    output=tmp_path/'upstream'
    result=subprocess.run([sys.executable,str(repo/'hulhe_escher/position_upstream.py'),
        '--source',str(repo),'--state',str(state),'--policy',str(policy),'--contract',str(recovery/'manifest.json'),
        '--selection',str(selection),'--output',str(output),'--repeats','8','--max-seconds','90'],
        cwd=repo,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-5000:]+result.stderr[-5000:]
    report=json.loads((output/'probes/report.json').read_text())
    assert report['status']=='COMPLETE' and report['iteration_after']==2
    assert len(report['positions'])==1
    assert report['positions'][0]['status']=='COMPLETE'
    assert len(report['positions'][0]['folds'])==2
    assert 'synthetic_after_mse' in report['positions'][0]
    assert json.loads((output/'iteration/report.json').read_text())['status']=='COMPLETE'
    assert json.loads((output/'iteration/family_errors.json').read_text())['by_player_family']
    assert before=={str(p):sha256_file(p) for p in state.rglob('*') if p.is_file()}
    # Probe instrumentation must leave the subsequent production iteration exactly
    # reproducible, including traversal replay, frozen labels and fitted outputs.
    baseline=tmp_path/'unprobed'
    plain=subprocess.run([sys.executable,str(repo/'hulhe_escher/bottleneck_iteration.py'),
        '--source',str(repo),'--state',str(state),'--policy',str(policy),'--contract',str(recovery/'manifest.json'),
        '--output',str(baseline)],cwd=repo,capture_output=True,text=True,timeout=90)
    assert plain.returncode==0,plain.stdout[-2000:]+plain.stderr[-2000:]
    for actual in (output/'iteration').glob('*.npz'):
        a,b=np.load(actual),np.load(baseline/actual.name)
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])


def test_read_only_traversal_restores_state_on_exception():
    from types import SimpleNamespace
    import unbiased_escher.solver as module
    calls=[]
    def write(*a,**k): calls.append(1)
    learner=SimpleNamespace(add_data=write)
    q=SimpleNamespace(add_data=write,observe_control_return=write,active_fold=1)
    solver=SimpleNamespace(regret_trainers=[learner],ave_policy_trainer=learner,q_value_trainer=q,
        calibration_trainer=None,nodes_touched=10,episode=12,_minimum_sample_probability=.1,
        _architecture_stats={'n':2},_record_estimate_diagnostics=write,
        _capture_rng_state=np.random.get_state,_restore_rng_state=np.random.set_state)
    saved=np.random.get_state(); original=module.control_variate_advantage
    with pytest.raises(RuntimeError):
        with read_only_traversal(solver,lambda *a:None):
            np.random.random();solver.nodes_touched=100;q.active_fold=0
            solver._architecture_stats['n']=50
            q.observe_control_return();learner.add_data()
            raise RuntimeError('test failure')
    assert not calls and q.active_fold==1 and solver.nodes_touched==10
    assert solver._architecture_stats=={'n':2}
    assert module.control_variate_advantage is original
    assert np.random.get_state()[2:]==saved[2:]
    np.testing.assert_array_equal(np.random.get_state()[1],saved[1])
    learner.add_data();assert calls==[1]


def test_locked_components_join_full_boards_and_flop_extensions():
    p=position();q=deepcopy(p);q['board']=q['board'][2:]+q['board'][:2]
    r=deepcopy(q);r['board']=r['board'][:3]
    locked=lock_partitions([p,q,r])
    assert len({x['partition'] for x in locked})==1


def test_collection_failure_publishes_failed_status(tmp_path,monkeypatch):
    import hulhe_escher.position_investigation.collect as module
    policy=tmp_path/'policy';policy.write_text('dummy')
    def fail(*a):raise ValueError('bad build')
    monkeypatch.setattr(module,'verify_build',fail)
    with pytest.raises(ValueError,match='bad build'):
        collect(policy,tmp_path/'results',assets=tmp_path,hands=2)
    assert json.loads((tmp_path/'results/report.json').read_text())['status']=='FAILED'


def test_position_batch_dry_run_and_submission(tmp_path,monkeypatch,capsys):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_position_investigation as launcher
    args=['--project','test','--bucket','gs://test-bucket/hulhe','--service-account','runner@test.iam.gserviceaccount.com',
        '--run-name','hulhe-positions-test','--source-uri','gs://test-bucket/source','--expected-policy-sha256','a'*64]
    calls=[];monkeypatch.setattr(launcher.subprocess,'run',lambda argv,check:calls.append(argv))
    launcher.main(args+['--output',str(tmp_path/'dry')]);assert not calls
    plan=json.loads((tmp_path/'dry/launch.json').read_text())
    job=json.loads((tmp_path/'dry/job.json').read_text())
    assert plan['hands_per_matchup']==20000 and plan['upstream_positions']==24
    assert job['taskGroups'][0]['taskSpec']['maxRetryCount']==0
    assert job['labels']['experiment']=='hulhe-positions'
    assert 'hulhe_escher.position_cloud' in (tmp_path/'dry/batch.sh').read_text()
    launcher.main(args+['--output',str(tmp_path/'submitted'),'--execute'])
    assert len(calls)==4 and calls[-1][:4]==['gcloud','batch','jobs','submit']
    capsys.readouterr()


def test_cloud_incomplete_is_visible_and_only_results_are_uploaded(tmp_path,monkeypatch):
    import hulhe_escher.position_cloud as cloud
    transfers=[]
    monkeypatch.setattr(cloud,'storage',lambda *a:transfers.append(a))
    monkeypatch.setattr(cloud,'prepare_fo2',lambda _:dict(fingerprint='test'))
    monkeypatch.setattr(cloud,'prepare_full',lambda *a:(tmp_path/'state',tmp_path/'policy',tmp_path/'contract',tmp_path/'archive',{}))
    monkeypatch.setattr(cloud,'run_child',lambda *a,**k:dict(returncode=2,timed_out=False))
    result=cloud.run_cloud(tmp_path/'work','gs://bucket/source','gs://bucket/diagnostic','a'*64)
    assert result['status']=='INCOMPLETE'
    assert all('isolated-source' not in str(t) for t in transfers)
    assert transfers[-1][0]=='cp' and str(transfers[-1][1]).endswith('cloud_status.json')
    with pytest.raises(ValueError):cloud.run_cloud(tmp_path/'bad','gs://bucket/source','gs://bucket/source/child','a'*64)
