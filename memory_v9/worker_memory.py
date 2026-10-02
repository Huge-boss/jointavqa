"""Independent model replicas; inference does not import the scorer or labels."""
import os
os.environ.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',FORCE_QWENVL_VIDEO_READER='decord')
import argparse,fcntl,json,time,traceback
from settings import *
from method import Runtime,run,scout_prompt

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['joint','avspeaker'],required=True)
    p.add_argument('--model',choices=['omni','videollama'],required=True);p.add_argument('--gpu',choices=GPUS,required=True)
    p.add_argument('--phase',choices=['pilot','dev20'],required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    lock=open('/tmp/avqa-evidence-agent-gpu'+a.gpu+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=frozen();rows=readlines(HERE/f'data/{a.dataset}.jsonl')
    ids=c['pilot_qids'][a.dataset][GPUS.index(a.gpu)::3] if a.phase=='pilot' else c['partitions'][a.dataset][a.gpu]
    rows=[r for q in ids for r in rows if r['qid']==q];assert len(rows)==len(ids)
    old={r['qid']:r for r in readlines(HERE/f'data/{a.dataset}_{a.model}_native.jsonl')}
    durations=json.loads((HERE/f'data/{a.dataset}_durations.json').read_text())
    dest=HERE/f'runs/{a.dataset}/{a.model}/{a.phase}/gpu{a.gpu}';dest.mkdir(parents=True,exist_ok=True)
    cfg=dict(args=vars(a),qids=ids,manifest_sha256=sha(HERE/'runs/manifest.json'))
    if (dest/'config.json').exists():assert json.loads((dest/'config.json').read_text())==cfg
    else:write(dest/'config.json',cfg)
    prior=readlines(dest/'predictions.jsonl');done={r['qid'] for r in prior if r['status']=='ok'}
    assert len(done)==sum(r['status']=='ok' for r in prior)
    t=time.perf_counter();runtime=Runtime(a.dataset,a.model)
    runtime.cache_dir=HERE/f'cache/{a.dataset}_{a.model}_{a.phase}_gpu{a.gpu}';runtime.cache_dir.mkdir(parents=True,exist_ok=True)
    write(dest/f'worker_start_{os.getpid()}.json',dict(pid=os.getpid(),time=time.time(),load_seconds=time.perf_counter()-t))
    policy=('uniform at most 32 full-clip frames; ' if a.dataset=='joint' else 'native full-clip frame sampling; ')
    policy+=(('continuous first 300 seconds of audio' if a.dataset=='joint' else 'continuous full-clip audio') if a.model=='omni'
             else 'eight sparse 2-second audio snippets, not continuous speech; '+('limited to the first 300 seconds' if a.dataset=='joint' else 'spread over the clip'))
    for row in rows:
        if row['qid'] in done:continue
        start=time.perf_counter();r=dict(qid=row['qid'],task=row['task'],dataset=a.dataset,model=a.model,physical_gpu=a.gpu,started_at=time.time())
        try:
            d=durations[row['qid']];duration=float(d['duration_seconds'] if isinstance(d,dict) else d)
            r['arm']=run(runtime,row,duration,old[row['qid']],policy)
            if a.phase=='pilot':
                # No score gate. Native cache parity and deterministic observation check only.
                repeat=runtime.call(row,scout_prompt(row,duration,policy),None,'pilot_memory_repeat',208)
                keys=['model_output','generated_tokens','input_tokens','audio_tokens','video_tokens','frames','audio_sha256','video_tensor_sha256','raw_frames_sha256']
                first=r['arm']['calls'][0]
                r['observation_repeat']={k:first.get(k)==repeat.get(k) for k in keys}
                assert all(r['observation_repeat'].values()),r['observation_repeat']
                native=runtime.call(row,row['question_prompt'],None,'native_parity',256)
                ref=old[row['qid']]['media_reference']
                r['native_parity']={k:native.get(k)==ref.get(k) for k in keys}
                assert all(r['native_parity'].values()),r['native_parity']
                assert not r['arm']['calls'][-1]['reached_token_limit'],'Final output truncated'
                r['validation_extra_seconds']=repeat['seconds']+native['seconds']
            r['status']='ok'
        except Exception as e:r.update(status='error',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        r.update(seconds=time.perf_counter()-start,finished_at=time.time())
        with (dest/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
        if r['status']=='ok':done.add(row['qid'])
        write(dest/'progress.json',dict(completed=len(done),total=len(rows),status=r['status'],time=time.time()))
        print(json.dumps({k:r.get(k) for k in ['qid','status','seconds','error']}),flush=True)
        if r['status']!='ok':raise SystemExit(42 if 'out of memory' in r['error'].lower() else 1)
    write(dest/'complete.json',dict(n=len(done),complete=True,technical_gate_passed=a.phase=='pilot',time=time.time()))

if __name__=='__main__':main()
