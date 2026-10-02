"""One unchanged v5 arm per question, two independent replicas, no scoring imports."""
import os
import argparse,fcntl,json,time,traceback
from common_stage import *
from worker import run_arm
from runtime import Runtime


def compare_arm(new,old):
    assert new['model_output']==old['model_output'] and new['review']==old['review'] and new['review_window']==old['review_window']
    assert new['memory']==old['memory'] and len(new['calls'])==len(old['calls'])
    keys=['model_output','generated_tokens','input_tokens','audio_tokens','video_tokens','frames','audio_sha256','video_tensor_sha256','raw_frames_sha256','window']
    for x,y in zip(new['calls'],old['calls']):
        assert all(x.get(k)==y.get(k) for k in keys),{k:(x.get(k),y.get(k)) for k in keys if x.get(k)!=y.get(k)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['omni','videollama'],required=True)
    p.add_argument('--dataset',choices=['joint','avspeaker'],required=True);p.add_argument('--mode',choices=['fixed','agent','check'],required=True)
    p.add_argument('--gpu',choices=['0','3'],required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    lock=open('/tmp/avqa-evidence-agent-gpu'+a.gpu+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=config();rows=readlines(V5/'data'/f'{a.dataset}.jsonl')
    assert all(not set(r)&{'label','answer','correct_answer','correct_prefix','answer_label'} for r in rows)
    durations=json.loads((V5/f'manifests/{a.dataset}_durations.json').read_text())
    if a.mode=='check':
        ids=json.loads((V5/'manifests/pilot.json').read_text())[a.dataset]['qids']
        references={r['qid']:r for r in readlines(V5/f'runs/{a.dataset}/{a.model}/pilot/predictions.jsonl') if r['status']=='ok'}
    else:ids=c['plan'][a.dataset+'/'+a.model+'/'+a.mode]['gpu'+a.gpu]
    rows=[r for r in rows if r['qid'] in set(ids)];assert len(rows)==len(ids)
    dest=HERE/f'runs/{a.dataset}/{a.model}/{a.mode}/gpu{a.gpu}';dest.mkdir(parents=True,exist_ok=True)
    cfg=dict(args=vars(a),qids=ids,stage_manifest_sha256=sha(HERE/'runs/manifest.json'),native_freeze_sha256=c['native_freeze_sha256'])
    if (dest/'config.json').exists():assert json.loads((dest/'config.json').read_text())==cfg
    else:write(dest/'config.json',cfg)
    prior=readlines(dest/'predictions.jsonl');done={r['qid'] for r in prior if r['status']=='ok'}
    assert len(done)==sum(r['status']=='ok' for r in prior)
    t=time.perf_counter();runtime=Runtime(a.dataset,a.model)
    runtime.cache_dir=HERE/f'cache/{a.dataset}_{a.model}_{a.mode}_gpu{a.gpu}';runtime.cache_dir.mkdir(parents=True,exist_ok=True)
    write(dest/f'worker_start_{os.getpid()}.json',dict(time=time.time(),load_seconds=time.perf_counter()-t,pid=os.getpid(),gpu=a.gpu))
    policy=('uniform at most 32 full-clip frames; ' if a.dataset=='joint' else 'native full-clip frame sampling; ')
    policy+=('continuous first 300 seconds of audio' if a.dataset=='joint' else 'continuous full-clip audio') if a.model=='omni' else 'eight sparse 2-second audio excerpts, not continuous speech coverage'
    for row in rows:
        if row['qid'] in done:continue
        start=time.perf_counter();r=dict(qid=row['qid'],task=row['task'],dataset=a.dataset,model=a.model,mode=a.mode,started_at=time.time(),origin=dict(kind='staged_v5_inference',physical_gpu=a.gpu,stage_code_sha256=c['stage_code_sha256'],native_freeze_sha256=c['native_freeze_sha256']))
        try:
            v=durations[row['qid']];duration=float(v['duration_seconds'] if isinstance(v,dict) else v)
            if a.mode=='check':
                arms={}
                for mode in ['fixed','agent']:
                    arms[mode]=run_arm(runtime,row,duration,mode,policy);compare_arm(arms[mode],references[row['qid']]['arms'][mode])
                r['arms']=arms;r['equivalence_passed']=True
            else:r['arm']=run_arm(runtime,row,duration,a.mode,policy)
            r['status']='ok'
        except Exception as e:r.update(status='error',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        r.update(seconds=time.perf_counter()-start,finished_at=time.time())
        with (dest/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
        if r['status']=='ok':done.add(row['qid'])
        write(dest/'progress.json',dict(completed=len(done),total=len(rows),status=r['status'],time=time.time()))
        print(json.dumps({k:r.get(k) for k in ['qid','status','seconds','error']}),flush=True)
        if r['status']!='ok':raise SystemExit(42 if 'out of memory' in r['error'].lower() else 1)
    write(dest/'complete.json',dict(complete=True,n=len(done),time=time.time(),equivalence_passed=a.mode=='check'))


if __name__=='__main__':main()
