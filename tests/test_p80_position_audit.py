"""Pinned P80 deployment, baseline routing and analytical reporting contracts."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hulhe_escher.checkpointing import sha256_file


def test_p80_preset_pins_full_source_and_p56_library(tmp_path,monkeypatch,capsys):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_p80_position_audit as launcher
    calls=[]
    with monkeypatch.context() as context:
        context.setattr(launcher.subprocess,'run',lambda argv,check:calls.append(argv))
        args=['--run-name','hulhe-positions-p80-test']
        launcher.main(args+['--output',str(tmp_path/'dry')])
        assert not calls
        launcher.main(args+['--output',str(tmp_path/'submitted'),'--execute'])
    assert len(calls)==4 and calls[-1][:4]==['gcloud','batch','jobs','submit']
    plan=json.loads((tmp_path/'dry/launch.json').read_text())
    assert plan['iteration']==2711 and plan['checkpoint_label']=='P80'
    assert plan['comparison_label']=='P56'
    assert plan['source_uri']==launcher.SOURCE_URI
    assert plan['expected_policy_sha256']==launcher.P80_SHA256
    assert plan['comparison_uri']==launcher.P56_AUDIT_URI
    assert plan['comparison_policy_uri']==launcher.P56_POLICY_URI
    assert plan['comparison_policy_sha256']==launcher.P56_SHA256
    assert plan['comparison_library_sha256']==launcher.P56_LIBRARY_SHA256
    assert plan['hands_per_matchup']==20000 and plan['upstream_positions']==24
    assert plan['estimator_repeats']==64 and plan['machine']=='c4-standard-16'
    assert 'frozen P56' in plan['comparison'] and 'P32' not in plan['comparison']
    job=json.loads((tmp_path/'dry/job.json').read_text())
    task=job['taskGroups'][0]['taskSpec']
    assert task['maxRetryCount']==0 and task['maxRunDuration']=='14400s'
    assert job['labels']==dict(experiment='hulhe-positions',checkpoint='p80')
    script=(tmp_path/'dry/batch.sh').read_text()
    assert '--checkpoint-label P80' in script and '--comparison-label P56' in script
    subprocess.run(['bash','-n',str(tmp_path/'dry/batch.sh')],check=True)
    capsys.readouterr()


def test_cloud_routes_p56_into_comparison_and_regression(tmp_path,monkeypatch):
    import hulhe_escher.position_cloud as cloud
    reference=tmp_path/'baseline.pkl';reference.write_bytes(b'pinned baseline')
    commands=[]
    def storage(*args):
        if args[0]=='cp' and str(args[1]).startswith('gs://'):
            Path(args[2]).write_bytes(reference.read_bytes() if str(args[2]).endswith('.pkl') else b'{}')
    monkeypatch.setattr(cloud,'storage',storage)
    monkeypatch.setattr(cloud,'prepare_fo2',lambda _:dict(fingerprint='test'))
    monkeypatch.setattr(cloud,'prepare_full',lambda *a:(tmp_path/'state',tmp_path/'policy',tmp_path/'contract',tmp_path/'archive',{}))
    def child(command,*args,**kwargs):
        commands.append(command)
        return dict(returncode=0,timed_out=False)
    monkeypatch.setattr(cloud,'run_child',child)
    monkeypatch.setattr(cloud,'render',lambda _:dict(status='COMPLETE'))
    result=cloud.run_cloud(tmp_path/'work','gs://bucket/p80','gs://bucket/audit','a'*64,
        iteration=2711,checkpoint_label='P80',comparison_label='P56',
        comparison_uri='gs://bucket/p56-audit',comparison_policy_uri='gs://bucket/p56.pkl',
        comparison_policy_sha256=sha256_file(reference),comparison_library_sha256='b'*64)
    assert result['status']=='COMPLETE' and result['comparison_label']=='P56'
    comparison=next(c for c in commands if 'hulhe_escher.position_investigation.compare' in c)
    assert comparison[comparison.index('--baseline-label')+1]=='P56'
    assert comparison[comparison.index('--candidate-label')+1]=='P80'
    regression=next(c for c in commands if 'hulhe_escher.strategic_audit' in c)
    assert any(v.startswith('P56=') for v in regression)
    assert any(v.startswith('P80=') for v in regression)
    assert not any(v.startswith('P32=') for v in regression)
    provenance=json.loads((tmp_path/'work/results/comparison_inputs.json').read_text())
    assert provenance['baseline_label']=='P56'


def test_summary_uses_p56_baseline_and_propagates_failed_comparison(tmp_path):
    from hulhe_escher.position_investigation.report import render
    for file in ('encounters/report.json','analysis/report.json','upstream/probes/report.json',
                 'upstream/iteration/report.json','regression/summary.json'):
        path=tmp_path/file;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(status='COMPLETE')))
    (tmp_path/'cloud_status.json').write_text(json.dumps(dict(status='COMPARISON',
        checkpoint_label='P80',comparison_label='P56',comparison_requested=True)))
    (tmp_path/'comparison').mkdir()
    (tmp_path/'comparison/report.json').write_text(json.dumps(dict(status='FAILED',baseline_label='P56')))
    report=render(tmp_path)
    assert report['status']=='FAILED' and report['paired_comparison']['baseline_label']=='P56'
    text=(tmp_path/'report.md').read_text()
    assert 'Paired comparison with P56' in text and 'frozen P80 opponent' in text
    assert 'P32' not in text


@pytest.mark.parametrize('label',['P56;echo bad','56',''])
def test_invalid_comparison_labels_rejected_before_work(tmp_path,label):
    from hulhe_escher.position_cloud import run_cloud
    with pytest.raises(ValueError,match='Checkpoint labels'):
        run_cloud(tmp_path/'work','gs://bucket/source','gs://bucket/output','a'*64,comparison_label=label)
    assert not (tmp_path/'work').exists()
