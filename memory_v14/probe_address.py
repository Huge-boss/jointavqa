"""One fixed synthetic address-readout per group; no QA labels or scoring."""
import os
os.environ.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',FORCE_QWENVL_VIDEO_READER='decord')
import argparse,copy,fcntl,json,time,traceback
import numpy as np
from settings import *
from method import Runtime,NativeRuntime,media_ledger
from overlay import digest
from support import DATA

PROMPT=('Read the small white code printed on the black rectangle at the top-left of the video frames. '
        'All supplied frames show the same code. Return only that visible code. If you cannot read it, return UNKNOWN.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['joint','avspeaker'],required=True)
    p.add_argument('--model',choices=['omni','videollama'],required=True)
    p.add_argument('--gpu',choices=GPUS,required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    lock=open('/tmp/avqa-evidence-agent-gpu'+a.gpu+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=frozen();dest=HERE/f'runs/address_probe/{a.dataset}_{a.model}.json'
    assert not dest.exists(),'Never overwrite/retry a completed OCR probe'
    dest.parent.mkdir(parents=True,exist_ok=True)
    report=dict(dataset=a.dataset,model=a.model,gpu=a.gpu,manifest_sha256=sha(HERE/'runs/manifest.json'),
        started_at=time.time(),prompt=PROMPT,qa_labels_loaded=False,qa_score=None,
        limitation='Artificial repeated-frame OCR fixture only; not natural event localization or development-set accuracy.')
    begin=time.perf_counter()
    try:
        qid=c['pilot_qids'][a.dataset][0]
        row=next(r for r in readlines(HERE/f'data/{a.dataset}.jsonl') if r['qid']==qid)
        source=DATA[a.dataset]/row['clip_path'];d=json.loads((HERE/f'data/{a.dataset}_durations.json').read_text())[qid]
        duration=float(d['duration_seconds'] if isinstance(d,dict) else d)
        report.update(qid=qid,source_sha256=sha(source))
        t=time.perf_counter();rt=Runtime(a.dataset,a.model);report['load_seconds']=time.perf_counter()-t
        rt.cache_dir=HERE/f'cache/address_probe_{a.dataset}_{a.model}';rt.cache_dir.mkdir(parents=True,exist_ok=True)
        rt.seed();rt.torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        prepared=rt.prepare(source);ledger=media_ledger(prepared['meta'],source,duration)
        report['prepare_seconds']=time.perf_counter()-t
        marked,meta=rt.mark_prepared(prepared,ledger)
        video=marked['video'] if a.model=='omni' else marked['media']['video']
        pair_index=len(meta['mapping'])//2;pair=meta['mapping'][pair_index]
        assert pair_index>0,'Do not test only P00'
        fixture=video[pair_index*2:pair_index*2+2].repeat((len(video)//2,1,1,1))
        assert fixture.shape==video.shape and fixture.dtype==video.dtype
        fixture_prepared=dict(marked);fixture_prepared['meta']=copy.deepcopy(marked['meta'])
        fixture_prepared['meta']['video_tensor_sha256']=digest(fixture.numpy())
        audio=prepared['audio'] if a.model=='omni' else prepared['media']['audio']
        audio_array=audio if isinstance(audio,np.ndarray) else audio.numpy()
        audio_before=digest(audio_array)
        if a.model=='omni':fixture_prepared['video']=fixture
        else:fixture_prepared['media']=dict(marked['media'],video=fixture)
        report.update(marker=meta,fixture_pair=pair,expected_code=pair['address'],
            actual_source_audio=ledger['audio'],native_meta=prepared['meta'],
            fixture_tensor_sha256=digest(fixture.numpy()),audio_feature_or_waveform_sha256=audio_before)
        # No schema constraint or expected code in perception prompt.
        rt.seed();t=time.perf_counter()
        out=NativeRuntime.generate(rt,{},source,PROMPT,fixture_prepared,16,structured=False)
        report['generation_seconds']=time.perf_counter()-t
        original=prepared['video'] if a.model=='omni' else prepared['media']['video']
        assert digest(original.numpy())==meta['original_tensor_sha256']
        assert digest(audio_array)==audio_before
        report.update(output=out,observed_code=out['model_output'].strip(),
            passed=out['model_output'].strip()==pair['address'] and not out['reached_token_limit'],
            peak_gpu_gib=rt.torch.cuda.max_memory_allocated()/2**30,status='complete')
    except Exception as e:
        report.update(status='error',passed=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    report.update(total_seconds=time.perf_counter()-begin,finished_at=time.time())
    report['nonload_work_seconds']=report['total_seconds']-report.get('load_seconds',0.)
    write(dest,report)
    print(json.dumps({k:report.get(k) for k in ['dataset','model','status','expected_code','observed_code','passed','total_seconds','error']}),flush=True)
    if report['status']=='error':raise SystemExit(1)

if __name__=='__main__':main()
