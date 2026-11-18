#!/usr/bin/env python3
"""Run new instrumentation against the verified archived production trainer."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','state','contract','policy','selection','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=64);p.add_argument('--max-seconds',type=int,default=1200)
    args=p.parse_args()
    current=Path(__file__).resolve().parent
    # Load the current instrumentation file without importing the current trainer.
    spec=importlib.util.spec_from_file_location('archived_iteration_instrumentation',current/'bottleneck_iteration.py')
    instrument=importlib.util.module_from_spec(spec);spec.loader.exec_module(instrument)
    sys.path.insert(0,str(args.source.resolve()))
    import hulhe_escher
    hulhe_escher.__path__.append(str(current))
    from hulhe_escher.position_investigation.upstream import Hooks,summarize_captures
    from hulhe_escher.checkpointing import sha256_file
    selection=json.loads(args.selection.read_text())
    if (selection['source_policy_sha256']!=sha256_file(args.policy)
            or selection['source_manifest_sha256']!=sha256_file(args.state/'manifest.json')):
        raise ValueError('Probe selection belongs to another checkpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    hooks=Hooks(args.selection,args.output/'probes',repeats=args.repeats,max_seconds=args.max_seconds)
    instrument.run(args.source,args.state,args.contract,args.policy,args.output/'iteration',
                   before_iteration=hooks.before,after_iteration=hooks.after)
    summarize_captures(args.output/'iteration',args.state)
    if hooks.report['status']!='COMPLETE': raise SystemExit(2)

if __name__=='__main__': main()
