"""Independent checks of leakage, exports, exact probes and archived iteration capture."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_policy_sampling_experiment import frozen_source
from hulhe_escher.bottleneck_refit import run_refits, splits, metrics, select_arm
from hulhe_escher.bottleneck_evaluate import evaluate, verified_candidate, observations
from hulhe_escher.checkpointing import LoadedHULHEPolicy, sha256_file, save_policy_checkpoint
from hulhe_escher.diagnostic_model import ResidualPolicy
from hulhe_escher.diagnostic_probes import positions
from hulhe_escher.features import make_feature_encoder, StructuredHULHEMLP
from hulhe_escher.game import load_hulhe_game
from hulhe_escher.policy_replay_artifacts import read_metadata
from unbiased_escher.replay_grouping import IndexedFeatureRows


def add_regrets(state):
    manifest,metadata=read_metadata(state)
    encoder=make_feature_encoder(metadata['config']['feature_encoder_id'])
    c=metadata['config']
    model=StructuredHULHEMLP(encoder.policy_layout,[c['num_hiddens']]*c['num_layers'],3,branch_width=c['structured_branch_width'])
    metadata['regret_trainers']=[dict(model=model.state_dict()),dict(model=model.state_dict())]
    path=state/manifest['metadata_path']; torch.save(metadata,path)
    manifest.update(metadata_sha256=sha256_file(path),metadata_size_bytes=path.stat().st_size)
    (state/'manifest.json').write_text(json.dumps(manifest))


def test_exact_river_probes_are_unbeatable_and_call_dominates_fold():
    rows=positions()
    assert len(rows)==6
    for row in rows[:4]:
        s=row['state']; player=s.current_player()
        assert row['exact'] and row['opponent_combinations']==990
        assert s.legal_actions()==[0,1]
        assert s.child(1).returns()[player]>s.child(0).returns()[player]
    assert all(not r['exact'] for r in rows[4:])


def test_split_disjoint_deterministic_and_audit_not_training():
    x=np.random.default_rng(3).normal(size=(1000,8)).astype(np.float32)
    x[1]=x[0]
    features=IndexedFeatureRows(x,np.arange(len(x)),'cpu')
    a=splits(features,.2);b=splits(features,.2)
    assert all(np.array_equal(c,d) for c,d in zip(a,b))
    assert len(set(np.concatenate(a)))==len(x) and sum(map(len,a))==len(x)
    assert all((0 in part)==(1 in part) for part in a)


def test_metrics_kl_removes_entropy_and_frequency_uses_counts():
    x=np.ones((4,2),np.float32); features=IndexedFeatureRows(x,np.arange(4),'cpu')
    model=torch.nn.Linear(2,3); torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
    data=(features,torch.full((4,3),1/3),torch.ones(4,3),torch.tensor([1.,2.,3.,4.]))
    result=metrics(model,data,np.arange(4),np.arange(4),np.array([1,2,10,100]))
    assert abs(result['overall']['weighted_kl'])<1e-6
    assert result['overall']['target_entropy']==pytest.approx(np.log(3))
    assert all(x['groups']==1 for x in result['by_frequency'].values())


def test_refits_selection_reload_and_fresh_evaluation(frozen_source,tmp_path):
    state,policy=frozen_source; add_regrets(state)
    before=sha256_file(state/'manifest.json'); original=sha256_file(policy)
    root=tmp_path/'fit'
    result=run_refits(state,policy,root,threads=1,smoke=True)
    assert result['status']=='COMPLETE' and len(result['results'])==6
    assert sha256_file(state/'manifest.json')==before and sha256_file(policy)==original
    for seed in result['fit_seeds']:
        rows=[r for r in result['results'] if r['fit_seed']==seed and r['arm']!='residual_low_lr']
        assert rows[0]['initial_model_sha256']==rows[1]['initial_model_sha256']
    for row in result['results']:
        loaded=LoadedHULHEPolicy(load_hulhe_game(),root/row['checkpoint'])
        s=positions()[-1]['state'];p=loaded.action_probabilities(s)
        assert sum(p.values())==pytest.approx(1)
        assert row['fit']['training_groups']==result['partition_sizes']['train']
        assert row['fit']['validation_groups']==result['partition_sizes']['validation']
        assert row['audit']['overall']['groups']==result['partition_sizes']['audit']
    arm,_=select_arm(result['results']); assert result['selection']['arm']==arm
    assert result['selection']['fit_seed']==2701
    report=evaluate(root,policy,tmp_path/'evaluation',workers=1,opponent='check_call',
                    pilot_pairs=4,minimum=8,maximum=8,max_seconds=60)
    assert report['status']=='COMPLETE' and report['completed_groups']==8
    assert report['completed_hands']==48 and not report['results']['head_to_head']['confirmatory']
    # Selection manifest is not allowed to switch to a lucky second fit seed.
    selection=json.loads((root/'selection.json').read_text());selection['fit_seed']=2702
    (root/'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError):verified_candidate(root,policy)


def test_deal_group_difference_not_independent_hands():
    rows=[{'head':{'mean_bb':x},'candidate_fo2':{'mean_bb':2*x},'p32_fo2':{'mean_bb':x}} for x in (1,2,3)]
    result=observations(rows)
    np.testing.assert_array_equal(result['fo2_improvement'],[1,2,3])
    np.testing.assert_array_equal(result['head_to_head'],[1,2,3])


def test_launcher_bounded_and_source_allowlisted(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from launch_bottleneck import build_bundle
    output=tmp_path/'launch'
    plan=build_bundle(output,project='test',region='europe-west1',bucket='gs://test-bucket/hulhe',
        service_account='runner@test.iam.gserviceaccount.com',run_name='hulhe-bottleneck-test',
        source_uri='gs://test-bucket/original',expected_policy_sha256='a'*64)
    assert plan['max_task_hours']==4 and plan['collection_workers']==8
    job=json.loads((output/'job.json').read_text())
    group=job['taskGroups'][0]; spec=group['taskSpec']
    assert group['taskCount']==group['parallelism']==group['taskCountPerNode']==1
    assert spec['maxRetryCount']==0 and spec['maxRunDuration']=='14400s'
    assert spec['computeResource']==dict(cpuMilli=16000,memoryMib=60000)
    machine=job['allocationPolicy']['instances'][0]['policy']
    assert machine['machineType']=='c4-standard-16' and machine['provisioningModel']=='STANDARD'
    assert machine['bootDisk']['sizeGb']==150 and machine['bootDisk']['type']=='hyperdisk-balanced'
    assert 'batch-cos' in machine['bootDisk']['image']
    assert job['logsPolicy']['destination']=='CLOUD_LOGGING'
    assert job['labels']['experiment']=='hulhe-bottleneck'
    assert job['allocationPolicy']['serviceAccount']['email']=='runner@test.iam.gserviceaccount.com'
    script=(output/'batch.sh').read_text()
    container=spec['runnables'][0]['container']
    assert container['commands']==['-c',script] and '@sha256:' in container['imageUri']
    assert "'.[parallel]'" in script and 'hulhe_escher.bottleneck_cloud' in script
    assert '--iteration 1132' in script and plan['expected_policy_sha256'] in script
    assert plan['source_sha256'] in script and 'sha256sum --check' in script
    assert 'shutdown' not in script and 'for attempt in 1 2 3 4' in script
    assert 'probe_bottleneck' not in script # coordinator runs the archived child
    subprocess.run(['bash','-n',str(output/'batch.sh')],check=True)
    import tarfile
    with tarfile.open(output/'source.tar.gz') as archive:
        names=archive.getnames()
    assert 'hulhe_escher/bottleneck_iteration.py' in names
    assert not any('/outputs/' in n or '.env' in n for n in names)


def test_batch_submission_requires_execute_and_stops_on_upload_failure(tmp_path,monkeypatch,capsys):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_bottleneck as launcher
    args=['--project','test','--bucket','gs://test-bucket/hulhe',
        '--service-account','runner@test.iam.gserviceaccount.com','--run-name','hulhe-bottleneck-test',
        '--source-uri','gs://test-bucket/original','--expected-policy-sha256','a'*64]
    calls=[]
    def command(argv,check):
        assert check
        calls.append(argv)
    monkeypatch.setattr(launcher.subprocess,'run',command)
    launcher.main(args+['--output',str(tmp_path/'dry')])
    assert not calls
    launcher.main(args+['--output',str(tmp_path/'submit'),'--execute'])
    plan=json.loads((tmp_path/'submit/launch.json').read_text())
    assert calls==plan['commands'] and len(calls)==4
    assert all(cmd[:4]==['gcloud','storage','cp','--if-generation-match=0'] for cmd in calls[:3])
    assert calls[-1][:5]==['gcloud','batch','jobs','submit','hulhe-bottleneck-test']
    assert calls[-1][calls[-1].index('--location')+1]=='europe-west1'
    calls.clear()
    def fail_upload(argv,check):
        calls.append(argv)
        raise subprocess.CalledProcessError(1,argv)
    monkeypatch.setattr(launcher.subprocess,'run',fail_upload)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.main(args+['--output',str(tmp_path/'failed'),'--execute'])
    assert len(calls)==1 and 'submit' not in calls[0]
    capsys.readouterr()


@pytest.mark.parametrize('override',[
    dict(source_uri='gs://test-bucket/hulhe'),
    dict(source_uri='gs://test-bucket/hulhe/hulhe-bottleneck-test'),
    dict(source_uri='gs://test-bucket/hulhe/hulhe-bottleneck-test/child'),
    dict(expected_policy_sha256='not-a-checksum'),dict(iteration=0),dict(run_name='Invalid'),
])
def test_batch_rejects_unsafe_or_invalid_inputs_before_build(tmp_path,override):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from launch_bottleneck import build_bundle
    args=dict(project='test',bucket='gs://test-bucket/hulhe',
        service_account='runner@test.iam.gserviceaccount.com',run_name='hulhe-bottleneck-test',
        source_uri='gs://test-bucket/original',expected_policy_sha256='a'*64)
    args.update(override)
    output=tmp_path/'bundle'
    with pytest.raises(ValueError):build_bundle(output,**args)
    assert not output.exists()


@pytest.mark.parametrize('all_regrets', [False, True])
def test_archived_iteration_capture_leaves_source_unchanged(tmp_path, all_regrets):
    repository=Path(__file__).resolve().parents[1]
    source=tmp_path/'source'
    result=subprocess.run([sys.executable,'-m','hulhe_escher.train','--smoke','--seed','53',
        '--threads','1','--policy-every','1','--checkpoint-every','1','--recovery-seconds','0',
        '--output',str(source),'--iterations','1'],cwd=repository,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout[-3000:]+result.stderr[-3000:]
    state=source/'state_000001'; before={str(p):sha256_file(p) for p in state.rglob('*') if p.is_file()}
    output=tmp_path/'upstream'
    command=[sys.executable,str(repository/'hulhe_escher/bottleneck_iteration.py'),
        '--source',str(repository),'--state',str(state),'--contract',str(source/'manifest.json'),
        '--policy',str(source/'policy_000001.pkl'),'--output',str(output)]
    if all_regrets: command += ['--capture-all-regrets','--save-regret-models','--skip-control-probe']
    result=subprocess.run(command,
        cwd=repository,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout[-5000:]+result.stderr[-5000:]
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='COMPLETE' and report['iteration']==2
    assert len(report['captures'])==4
    assert before=={str(p):sha256_file(p) for p in state.rglob('*') if p.is_file()}
    for row in report['captures']:
        assert np.isfinite(row['after_mse'])
        if all_regrets and row['name'].startswith('regret_'):
            assert row['sampled_rows']==row['rows']
            for phase,checksum in row['model_snapshots'].items():
                path=output/f'{row["name"]}_models_{phase}.pt'
                assert sha256_file(path)==checksum
                snapshot=torch.load(path,map_location='cpu',weights_only=False)
                assert snapshot['kind']=='diagnostic_regret_snapshot' and snapshot['iteration']==2
            assert sha256_file(output/f'{row["name"]}_after.npz')==row['after_sha256']
        else:
            assert row['sampled_rows']<=4096
        arrays=np.load(output/row['artifact'])
        assert np.isfinite(arrays['target']).all()
    assert not list(output.glob('state_*'))
    if all_regrets:
        assert not report['trajectory_ids_available'] and 'root_control_variance' not in report
        from hulhe_escher.p104_regret import run as diagnose
        diagnostic=diagnose(output,state,source/'policy_000001.pkl',tmp_path/'diagnostic',
                            threads=1,smoke=True,max_seconds=60)
        assert diagnostic['status']=='COMPLETE' and len(diagnostic['exports'])==2
        assert diagnostic['capture_iteration']==2
        assert before=={str(p):sha256_file(p) for p in state.rglob('*') if p.is_file()}


def test_cloud_publishes_results_only_and_surfaces_failures(tmp_path,monkeypatch):
    import hulhe_escher.bottleneck_cloud as cloud
    transfers=[];children=[]
    monkeypatch.setattr(cloud,'storage',lambda *args:transfers.append(args))
    monkeypatch.setattr(cloud,'prepare_fo2',lambda path:dict(fingerprint='test'))
    def prepare(uri,root,expected,iteration):
        root.mkdir()
        return root/'state',root/'P32.pkl',root/'contract.json',root/'archive',dict(full_checkpoint_bytes=123)
    monkeypatch.setattr(cloud,'prepare_full',prepare)
    def child(command,log,**kwargs):
        children.append((command,kwargs['timeout']))
        output=Path(command[command.index('--output')+1]);output.mkdir()
        if output.name=='upstream': data=dict(status='COMPLETE',iteration_seconds=1)
        elif output.name=='fit': data=dict(status='COMPLETE',selection=dict(arm='test'),elapsed_seconds=1)
        else:data=dict(status='COMPLETE')
        (output/('experiment.json' if output.name=='fit' else 'report.json')).write_text(json.dumps(data))
        return dict(returncode=0,timed_out=False)
    monkeypatch.setattr(cloud,'run_child',child)
    root=tmp_path/'run'
    report=cloud.run_cloud(root,'gs://bucket/source','gs://bucket/diagnostic','a'*64)
    assert report['status']=='COMPLETE' and [t for _,t in children]==[900,3600,3900]
    assert all('isolated-source' not in str(args) for args in transfers)
    assert transfers[-1][0]=='cp' and str(transfers[-1][1]).endswith('cloud_status.json')
    assert json.loads((root/'results/summary.json').read_text())['status']=='COMPLETE'
    with pytest.raises(ValueError): cloud.run_cloud(tmp_path/'bad','gs://bucket/source','gs://bucket/source/sub','a'*64)
    monkeypatch.setattr(cloud,'run_child',lambda *a,**kw:dict(returncode=1,timed_out=False))
    failed=tmp_path/'failed'
    with pytest.raises(RuntimeError):cloud.run_cloud(failed,'gs://bucket/source','gs://bucket/failure','a'*64)
    assert json.loads((failed/'results/cloud_status.json').read_text())['status']=='FAILED'
