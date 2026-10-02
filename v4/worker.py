"""Inference-only worker: never loads labels, previous predictions, or scoring code."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('FORCE_QWENVL_VIDEO_READER','decord')
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import time
import traceback
from agent import initial_prompt, final_prompt, parse_memory, window
from support import HERE, ROOT, sha, write, readlines, code_hashes
from runtime import Runtime


def run_arm(runtime, row, duration, mode, policy):
    begin=time.perf_counter();runtime.clear()
    if mode=='baseline':
        call=runtime.call(row,row['question_prompt'],None,'baseline',256)
        return dict(model_output=call['model_output'],calls=[call],seconds=time.perf_counter()-begin,
                    review=False,cache_start='empty_application_media_cache')
    first=runtime.call(row,initial_prompt(row,duration,policy),None,'initial_memory',128)
    memory=parse_memory(first['model_output'],duration)
    # Truncated memory cannot justify early exit, even if its first fields look valid.
    if first['reached_token_limit']:
        memory.update(review=True,reason='truncated_memory')
    calls=[first]; review=mode=='fixed' or memory['review']
    if review:
        center=duration/2 if mode=='fixed' else memory['center']
        interval=window(center,duration)
        last=runtime.call(row,final_prompt(row,duration,interval,first['model_output'][:1000]),interval,'review_final',16)
        calls.append(last);output=last['model_output']
    else:
        interval=None;output=memory['candidate']
    assert output and len(calls)<=2
    return dict(model_output=output,calls=calls,memory=memory,review=review,review_window=interval,
                seconds=time.perf_counter()-begin,cache_start='empty_application_media_cache')


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['avspeaker','joint'],required=True)
    p.add_argument('--model',choices=['omni','videollama'],required=True)
    p.add_argument('--phase',choices=['pilot','dev20'],required=True);p.add_argument('--gpu',choices=['0','3'],required=True)
    a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    lock=open('/tmp/avqa-evidence-agent-gpu'+a.gpu+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=readlines(HERE/'data'/f'{a.dataset}.jsonl')
    assert all(not set(r)&{'label','answer','correct_answer','correct_prefix','answer_label'} for r in rows)
    manifest=json.loads((HERE/'manifests'/f'{a.dataset}_split.json').read_text())
    assert sha(HERE/'data'/f'{a.dataset}.jsonl')==manifest['subset_sha256']
    durations=json.loads((HERE/'manifests'/f'{a.dataset}_durations.json').read_text())
    if a.phase=='pilot':
        ids=json.loads((HERE/'manifests/pilot.json').read_text())[a.dataset]['qids']
        rows=[r for q in ids for r in rows if r['qid']==q]
    frozen=json.loads((HERE/'manifests/code_freeze.json').read_text())
    assert frozen['python_sha256']==code_hashes(), 'Frozen code changed; use a new version'
    assert sha(HERE/'protocol.json')==frozen['protocol_sha256']
    for name,digest in frozen['manifest_sha256'].items():assert sha(HERE/'manifests'/name)==digest,name
    for rel,digest in frozen['external_sha256'].items(): assert sha(ROOT/rel)==digest,rel
    dest=HERE/f'runs/{a.dataset}/{a.model}/{a.phase}';dest.mkdir(parents=True,exist_ok=True)
    config=dict(args=vars(a),qids=[r['qid'] for r in rows],code=frozen,protocol_sha256=sha(HERE/'protocol.json'),
                subset_sha256=manifest['subset_sha256'],packages={n:importlib.metadata.version(n) for n in ['torch','transformers','decord','flash-attn']})
    if (dest/'config.json').exists():assert json.loads((dest/'config.json').read_text())==config
    else:write(dest/'config.json',config)
    records=readlines(dest/'predictions.jsonl');done={r['qid'] for r in records if r['status']=='ok'}
    assert len(done)==sum(r['status']=='ok' for r in records)
    t=time.perf_counter();runtime=Runtime(a.dataset,a.model);load_seconds=time.perf_counter()-t
    write(dest/f'worker_start_{os.getpid()}.json',dict(pid=os.getpid(),load_seconds=load_seconds,time=time.time(),
          note='Model initialization measured separately; all arms use the same already loaded weights. No OS page-cache flush.'))
    policy=('uniform at most 32 full-clip frames; ' if a.dataset=='joint' else 'native full-clip frame sampling; ')
    policy+=('continuous first 300 seconds of audio' if a.dataset=='joint' else 'continuous full-clip audio') if a.model=='omni' else 'eight sparse 2-second audio excerpts, not continuous speech coverage'
    for row in rows:
        if row['qid'] in done:continue
        start=time.perf_counter();r=dict(qid=row['qid'],task=row['task'],dataset=a.dataset,model=a.model,phase=a.phase,started_at=time.time(),arms={})
        try:
            value=durations[row['qid']]
            duration=float(value['duration_seconds'] if isinstance(value,dict) else value)
            order=sorted(['baseline','fixed','agent'],key=lambda m:hashlib.sha256((row['qid']+'|'+m).encode()).hexdigest())
            r.update(duration_seconds=duration,arm_order=order)
            for mode in order:r['arms'][mode]=run_arm(runtime,row,duration,mode,policy)
            if a.phase=='pilot':
                # Explicitly measure a warm-cache baseline and verify exact media/text repeatability.
                runtime.clear()
                cold=runtime.call(row,row['question_prompt'],None,'pilot_cold_repeat',256)
                warm=runtime.call(row,row['question_prompt'],None,'pilot_warm_repeat',256)
                keys=['model_output','generated_tokens','frames','audio_sha256','video_tensor_sha256','raw_frames_sha256','input_tokens','audio_tokens','video_tokens']
                r['baseline_repeat_matches']={k:cold.get(k)==warm.get(k)==r['arms']['baseline']['calls'][0].get(k) for k in keys}
                assert all(r['baseline_repeat_matches'].values()),'Cache changed baseline processing or generation'
                r['pilot_cache_measurement']=dict(cold_seconds=cold['seconds'],warm_seconds=warm['seconds'],warm_hit=warm['cache_hit'],cold_build_seconds=cold['cache_build_seconds'])
                runtime.clear()
                repeat=runtime.call(row,initial_prompt(row,duration,policy),None,'pilot_memory_repeat',128)
                r['memory_repeat_matches']={k:repeat.get(k)==r['arms']['agent']['calls'][0].get(k) for k in keys}
                assert all(r['memory_repeat_matches'].values()),'Memory repeatability failed'
            r['status']='ok'
        except Exception as exc:
            r.update(status='error',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        r.update(seconds=time.perf_counter()-start,finished_at=time.time())
        with (dest/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
        if r['status']=='ok':done.add(row['qid'])
        write(dest/'progress.json',dict(completed=len(done),total=len(rows),last_qid=row['qid'],last_status=r['status'],last_seconds=r['seconds'],updated_at=time.time()))
        print(json.dumps({k:r.get(k) for k in ['qid','status','seconds','error_type']}),flush=True)
        if r['status']=='error':raise SystemExit(42 if 'out of memory' in r['error'].lower() else 1)
    write(dest/'complete.json',dict(complete=True,n=len(done),time=time.time()))


if __name__=='__main__':main()
