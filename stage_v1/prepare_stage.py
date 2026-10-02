"""Import immutable completed arms and plan missing work without reading labels."""
import json,time
from common_stage import *


def main():
    assert not (HERE/'runs/manifest.json').exists(),'Already prepared; no overwrite'
    stop=json.loads((V5/'runs/stop_for_stage_schedule.json').read_text())
    assert not stop['still_alive']
    f=json.loads((V5/'manifests/code_freeze.json').read_text())
    assert f['python_sha256']=={p.name:sha(p) for p in sorted(V5.glob('*.py'))}
    plan={};audit={};inherited_hashes={}
    for ds in ['avspeaker','joint']:
        rows=readlines(V5/'data'/f'{ds}.jsonl');ids=[r['qid'] for r in rows]
        assert len(ids)==len(set(ids)) and len(ids)=={'joint':571,'avspeaker':642}[ds]
        for model in ['omni','videollama']:
            old=source_records(ds,model);assert set(ids)<=set(old)
            path=V5/f'runs/{ds}/{model}/dev20/predictions.jsonl'
            path_digest=sha(path)
            assert path_digest==stop['predictions'][str(path.relative_to(V5))]['sha256']
            done={r['qid']:(i,r) for i,r in enumerate(readlines(path),1) if r['status']=='ok'}
            assert set(done)<=set(ids)
            parity=['model_output','generated_tokens','input_tokens','audio_tokens','video_tokens','audio_sha256','video_tensor_sha256','raw_frames_sha256']
            diffs={k:sum(v['arms']['baseline']['calls'][0].get(k)!=old[q][0].get(k) for q,(_,v) in done.items()) for k in parity}
            assert not any(diffs.values()),('Historical baseline mismatch',ds,model,diffs)
            audit[ds+'/'+model]=dict(overlap=len(done),differences=diffs)
            for mode in ['baseline','fixed','agent']:
                inherited=[]
                for row in rows:
                    q=row['qid']
                    if mode=='baseline':
                        r,origin=old[q]
                        assert r['question_prompt']==row['question_prompt']
                        elapsed=r.get('seconds',r.get('elapsed_seconds'))
                        assert elapsed is not None and elapsed>0
                        arm=dict(model_output=r['model_output'],seconds=elapsed,calls=[r],review=False,
                                 timing_kind='historical_whole_question_including_native_preparation')
                    elif q in done:
                        line,r=done[q];arm=r['arms'][mode]
                        origin=dict(path=str(path.relative_to(ROOT)),sha256=path_digest,line=line,kind='v5_completed_arm')
                    else:continue
                    inherited.append(dict(qid=q,task=row['task'],dataset=ds,model=model,mode=mode,status='ok',arm=arm,origin=origin))
                dest=HERE/f'runs/{ds}/{model}/{mode}/inherited.jsonl';jsonlines(dest,inherited)
                inherited_hashes[str(dest.relative_to(HERE))]=sha(dest)
                completed={r['qid'] for r in inherited};a,b=split_missing(ids,completed)
                plan[ds+'/'+model+'/'+mode]=dict(planned=len(ids),inherited=len(completed),gpu0=a,gpu3=b)
    manifest=dict(created_at=time.time(),stage_code_sha256=code_hashes(),native_freeze_sha256=sha(V5/'manifests/code_freeze.json'),
                  stop_record_sha256=sha(V5/'runs/stop_for_stage_schedule.json'),input_sha256=f['data_sha256'],
                  inherited_sha256=inherited_hashes,plan=plan,baseline_overlap_audit=audit,
                  rule='Historical complete baseline; import every successful v5 arm; no selection by correctness; only missing fixed/agent work.')
    write(HERE/'runs/manifest.json',manifest)
    print(json.dumps({k:{'inherited':v['inherited'],'gpu0':len(v['gpu0']),'gpu3':len(v['gpu3'])} for k,v in plan.items()},indent=2))


if __name__=='__main__':main()
