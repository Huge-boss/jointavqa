"""CPU prototype check using frozen technical-pilot media, without QA labels/models."""
import json
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from overlay import annotate,digest

HERE=Path(__file__).resolve().parent;BASE=HERE.parent;OUT=HERE/'artifacts'
sys.path.insert(0,str(BASE))
from prepare_v11_media_review import sha,DATA


def main():
    import torch
    from decord import VideoReader,cpu
    from transformers import AutoImageProcessor
    torch.set_num_threads(2)
    font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf')
    models=Path('/data/lzq/code/project/acoustic-focs/model')
    proc=AutoImageProcessor.from_pretrained(str(models/'siglip-so400m-patch14-384'),local_files_only=True,use_fast=False)
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'FEASIBILITY.json').exists(),'Inspect existing prototype results before rebuilding'
    begin=time.perf_counter();results=[]
    with tarfile.open(BASE/'memory_v13/runs/complete_results.tar.gz') as tar:
        manifest=json.load(tar.extractfile('runs/manifest.json'))
        for ds in ['joint','avspeaker']:
            rows={r['qid']:r for r in map(json.loads,tar.extractfile(f'data/{ds}.jsonl'))}
            for model in ['omni','videollama']:
                qid=manifest['pilot_qids'][ds][0]
                raw=[json.loads(line) for n in tar.getnames() if n.startswith(f'runs/{ds}/{model}/pilot/gpu') and n.endswith('/predictions.jsonl') for line in tar.extractfile(n)]
                r=next(r for r in raw if r['qid']==qid);call=r['arm']['calls'][0]
                source=DATA[ds]/rows[qid]['clip_path'];assert sha(source)==r['arm']['memories'][0]['clip_sha256']
                frames=call['ledger_media']['frames'];indices=[f['source_frame'] for f in frames];times=[f['source_seconds'] for f in frames]
                vr=VideoReader(str(source),ctx=cpu(0),num_threads=2);rgb=vr.get_batch(indices).asnumpy()
                if model=='videollama':
                    pixels=proc.preprocess([Image.fromarray(f) for f in rgb],return_tensors='pt')['pixel_values'].numpy();encoding='normalized'
                elif ds=='joint':
                    sys.path.insert(0,str(BASE/'compat/joint'));from media import qwen_video
                    pixels=qwen_video(rgb).numpy();encoding='rgb255'
                else:
                    import qwen_omni_utils as q
                    vision=q.process_mm_info.__globals__['process_vision_info']
                    (video,vm),fps=vision.__globals__['fetch_video'](dict(type='video',video=str(source),fps=1.),return_video_sample_fps=True,return_video_metadata=True)
                    assert vm['frames_indices']==indices
                    pixels=video.numpy();encoding='rgb255'
                expected=call.get('video_tensor_sha256')
                if expected:assert digest(pixels)==expected,'Native reconstructed tensor differs; no marker study until diagnosed'
                old=digest(pixels);start=time.perf_counter()
                marked,meta=annotate(pixels,times,font,encoding,proc.image_mean,proc.image_std)
                overhead=time.perf_counter()-start
                assert digest(pixels)==old and marked.shape==pixels.shape
                again,meta2=annotate(pixels,times,font,encoding,proc.image_mean,proc.image_std)
                assert np.array_equal(marked,again) and meta==meta2
                def display(frame):
                    if encoding=='normalized':frame=(frame*np.asarray(proc.image_std)[:,None,None]+np.asarray(proc.image_mean)[:,None,None])*255
                    return Image.fromarray(np.clip(frame.transpose(1,2,0),0,255).astype('uint8'))
                # Keep first four frames at native resolution to inspect address legibility.
                n=min(4,len(marked));h,w=marked.shape[2:];canvas=Image.new('RGB',(2*w,n*(h+24)),'#222222');draw=ImageDraw.Draw(canvas)
                for i in range(n):
                    y=i*(h+24);canvas.paste(display(pixels[i]),(0,y));canvas.paste(display(marked[i]),(w,y));draw.text((4,y+h+4),f'{ds}/{model} input position {i}; original | paired address prototype',fill='white')
                canvas.save(OUT/f'{ds}_{model}.jpg',quality=92)
                results.append(dict(dataset=ds,model=model,qid=qid,native_tensor_hash_checked=bool(expected),
                    source_indices_checked=True,native_reconstruction_sha256=old,annotation_seconds=overhead,**meta))
    report=dict(status='CPU rendering feasibility only; no model readability/localization/accuracy claim',results=results,
        gpu_used=False,model_weights_loaded=False,qa_labels_loaded=False,seconds=time.perf_counter()-begin,
        audio_note='Prototype accepts video only; full runtime audio/token parity must be tested separately before any pilot release.')
    (OUT/'FEASIBILITY.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    (OUT/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(OUT/'feasibility.tar.gz','w:gz') as tar:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.suffix!='.gz':tar.add(p,arcname=p.name)
    print(json.dumps(dict(archive_sha256=sha(OUT/'feasibility.tar.gz'),seconds=report['seconds'],groups=len(results))))

if __name__=='__main__':main()
