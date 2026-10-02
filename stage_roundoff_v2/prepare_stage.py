"""Label-blind continuation importing every successful arm from stopped stage_v1."""
import os,time,json
from common_stage import *
from agent import window
from roundoff_fix import valid_interval


def main():
    assert not (HERE/'runs/manifest.json').exists(), 'No overwrite'
    prior=HERE.parent/'stage_v1'
    state=json.loads((prior/'runs/state.json').read_text())
    assert state['status']=='needs_diagnosis'
    for p in (prior/'runs').glob('worker_gpu*.json'):
        assert not Path('/proc/'+str(json.loads(p.read_text())['pid'])).exists()
    frozen=json.loads((V5/'manifests/code_freeze.json').read_text())
    assert frozen['python_sha256']=={p.name:sha(p) for p in sorted(V5.glob('*.py'))}
    old=json.loads((prior/'runs/manifest.json').read_text())
    assert old['stage_code_sha256']=={p.name:sha(p) for p in sorted(prior.glob('*.py'))}
    for rel,h in old['inherited_sha256'].items():assert sha(prior/rel)==h
    plan={};inherited_hashes={};source_hashes={};errors=[];bad_windows=[];windows_checked=0
    for ds in ['avspeaker','joint']:
        rows=readlines(V5/f'data/{ds}.jsonl');ids=[r['qid'] for r in rows]
        durations=json.loads((V5/f'manifests/{ds}_durations.json').read_text())
        for row in rows:
            v=durations[row['qid']];d=float(v['duration_seconds'] if isinstance(v,dict) else v)
            for bin_id in range(8):
                w=window(d*(bin_id+.5)/8,d);windows_checked+=1
                assert valid_interval(w), (row['qid'],bin_id,w)
                if not(0<=w[0]<w[1] and w[1]-w[0]<=12.001):
                    bad_windows.append(dict(dataset=ds,qid=row['qid'],bin=bin_id,interval=w,width=w[1]-w[0]))
        for model in ['omni','videollama']:
            for mode in ['baseline','fixed','agent']:
                dest=prior/f'runs/{ds}/{model}/{mode}';found={}
                for path in [dest/'inherited.jsonl']+sorted(dest.glob('gpu*/predictions.jsonl')):
                    digest=sha(path);source_hashes[str(path.relative_to(ROOT))]=digest
                    for line,r in enumerate(readlines(path),1):
                        if r['status']!='ok':errors.append(r);continue
                        assert r['qid'] not in found and r['qid'] in ids
                        r['continuation_source']=dict(path=str(path.relative_to(ROOT)),sha256=digest,line=line)
                        found[r['qid']]=r
                inherited=[found[q] for q in ids if q in found]
                out=HERE/f'runs/{ds}/{model}/{mode}/inherited.jsonl';jsonlines(out,inherited)
                inherited_hashes[str(out.relative_to(HERE))]=sha(out)
                a,b=split_missing(ids,found)
                assert not(a or b) or (ds,model,mode)==('joint','videollama','agent')
                plan[ds+'/'+model+'/'+mode]=dict(planned=len(ids),inherited=len(found),gpu0=a,gpu3=b)
    jsonlines(HERE/'runs/historical_errors.jsonl',errors)
    assert len(errors)==2 and all(e['error_type']=='AssertionError' for e in errors)
    repair=dict(time=time.time(),cause='Binary float width 12.001000000000001 exceeds intended inclusive 12.001 tolerance',
                change='math.isclose(width,12.001,rel_tol=0,abs_tol=1e-12) added to interval guard only; crop coordinates, ffmpeg arguments, prompts, model, decoding and scoring unchanged',
                all_candidate_windows_checked=windows_checked,previously_rejected_windows=bad_windows,
                source_state=state,prior_archive_sha256=sha(HERE.parent/'stage_v1_before_roundoff_repair.tar.gz'),
                historical_failure_count=len(errors),original_scores_preserved=True)
    write(HERE/'runs/repair_record.json',repair)
    write(HERE/'runs/manifest.json',dict(created_at=time.time(),stage_code_sha256=code_hashes(),
          native_freeze_sha256=sha(V5/'manifests/code_freeze.json'),input_sha256=frozen['data_sha256'],
          inherited_sha256=inherited_hashes,source_sha256=source_hashes,plan=plan,
          repair_record_sha256=sha(HERE/'runs/repair_record.json'),
          rule='Import ALL successes without labels; only missing Joint VideoLLaMA Agent questions; numerical guard repair in separate version.'))
    print(json.dumps(dict(plan={k:dict(inherited=v['inherited'],gpu0=len(v['gpu0']),gpu3=len(v['gpu3'])) for k,v in plan.items()},repair=repair),indent=2))


if __name__=='__main__':main()
