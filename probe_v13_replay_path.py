"""Forced interface-path check, never a scored arm or changed development protocol."""
import os
os.environ.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',FORCE_QWENVL_VIDEO_READER='decord')
import argparse,json,pathlib,sys,time,fcntl
BASE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'memory_v13'))
from settings import HERE,frozen,readlines,write,sha
from method import Runtime,isolated_local_prompt,isolated_verify_prompt,note,interval_for
from support import DATA

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model',required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
 assert a.gpu in ['0','1','3'] and os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
 lock=open('/tmp/avqa-evidence-agent-gpu'+a.gpu+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 c=frozen();dest=HERE/f'runs/path_probe/{a.dataset}_{a.model}.json';assert not dest.exists()
 rows={r['qid']:r for r in readlines(HERE/f'data/{a.dataset}.jsonl')}
 durations=json.loads((HERE/f'data/{a.dataset}_durations.json').read_text())
 duration=lambda q:float(durations[q]['duration_seconds'] if isinstance(durations[q],dict) else durations[q])
 q=next(q for q in c['pilot_qids'][a.dataset] if duration(q)>12.001)
 old=next(r for f in (HERE/f'runs/{a.dataset}/{a.model}/pilot').glob('gpu*/predictions.jsonl') for r in readlines(f) if r['qid']==q)
 ledger=old['arm']['calls'][0]['ledger_media'];frames=ledger['frames'];anchor=frames[len(frames)//2]['index']
 # Synthetic event descriptor addresses an implementation path only; not a model observation.
 synthetic=dict(EVIDENCE='Technical path check only.',ANCHOR=anchor,RELATION='AROUND',field_token_caps={})
 window,source=interval_for(synthetic,ledger,duration(q));assert window
 begin=time.perf_counter();runtime=Runtime(a.dataset,a.model);loading=time.perf_counter()-begin
 runtime.cache_dir=HERE/f'cache/path_probe_{a.dataset}_{a.model}_gpu{a.gpu}';runtime.cache_dir.mkdir(parents=True,exist_ok=True)
 runtime.active_duration=duration(q)
 start=time.perf_counter();local=runtime.call(rows[q],isolated_local_prompt(rows[q],window),window,'local_observation',96)
 observation=note(local,sha(DATA[a.dataset]/rows[q]['clip_path']),duration(q),'verification_local')
 if local['reached_token_limit']:observation['text']+=' [PARTIAL: generation token cap reached]'
 verify=runtime.call(rows[q],isolated_verify_prompt(rows[q],[observation]),None,'disagreement_verification',16)
 context=json.dumps(verify['serialized_prompt']);assert 'Available input-frame addresses:' not in context
 assert 'Technical path check only.' not in context and 'Observation proposal_ledger' not in context
 assert local['actual_source_audio']['source_offset']==window[0]
 assert all(window[0]<=x<=window[1]+.001 for span in local['actual_source_audio']['source_intervals'] for x in span)
 write(dest,dict(passed=True,kind='forced technical branch check; not scored or included in development method outputs',qid=q,model=a.model,dataset=a.dataset,gpu=a.gpu,anchor=anchor,window=window,window_source=source,local=local,verify=verify,load_seconds=loading,work_seconds=time.perf_counter()-start,manifest_sha256=sha(HERE/'runs/manifest.json')))
 print(json.dumps(dict(passed=True,dataset=a.dataset,model=a.model,window=window,local_cap=local['reached_token_limit'],verify_cap=verify['reached_token_limit'])))

if __name__=='__main__':main()
