"""CPU-only offline evidence material builder; no inference, scorer, or model imports."""
import hashlib
import html
import json
import math
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'offline_v11_media_review'
DATA = {'joint': Path('/data/lzq/data/JointAVBench'), 'avspeaker': Path('/data/lzq/data/AV-SpeakerBench')}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def sheet(video, indices, path, title, fps, offset=0):
    from PIL import Image, ImageDraw
    cols, w, h = 4, 300, 194
    canvas = Image.new('RGB', (cols*w, 28+math.ceil(len(indices)/cols)*h), '#202020')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill='white')
    for j, idx in enumerate(indices):
        frame = Image.fromarray(video[int(idx)].asnumpy())
        frame.thumbnail((w, h-22))
        x, y = (j%cols)*w, 28+(j//cols)*h
        canvas.paste(frame, (x+(w-frame.width)//2,y))
        draw.text((x+5,y+h-19), f'frame {idx} | {offset+idx/fps:.3f}s', fill='white')
    canvas.save(path, quality=88)


def main():
    from decord import VideoReader, cpu
    OUT.mkdir(exist_ok=True)
    (OUT/'sheets').mkdir(exist_ok=True)
    selected = json.loads((ROOT/'memory_v11/runs/completed_audit/DIAGNOSTIC_CASES.json').read_text())
    archive = ROOT/'memory_v11/runs/complete_results.tar.gz'
    expected = json.loads((ROOT/'memory_v11/runs/state.json').read_text())['archive_sha256']
    assert sha(archive) == expected
    raw = {}
    with tarfile.open(archive) as tar:
        for name in tar.getnames():
            if '/dev20/gpu' in name and name.endswith('/predictions.jsonl'):
                for line in tar.extractfile(name):
                    r = json.loads(line)
                    raw[r['dataset'], r['model'], r['qid']] = r
    blind, trace, lookup = [], [], []
    started = time.time()
    for i, c in enumerate(selected):
        caseid = f'C{i+1:02d}'
        ds, model, inp = c['dataset'], c['model'], c['input']
        source = (DATA[ds]/inp['clip_path']).resolve()
        assert source.is_relative_to(DATA[ds]) and source.is_file()
        r = raw[ds,model,inp['qid']]
        arm = r['arm']
        initial = next(x for x in arm['calls'] if x['stage'] == 'initial_memory')
        source_sha = sha(source)
        assert source_sha == arm['memories'][0]['clip_sha256']
        vr = VideoReader(str(source), ctx=cpu(0), num_threads=2)
        fps = float(vr.get_avg_fps())
        vm = initial.get('video_metadata', {})
        indices = initial.get('frame_indices', vm.get('frames_indices'))
        assert indices and len(indices) == initial['frames']
        assert min(indices) >= 0 and max(indices) < len(vr)
        sheet(vr,indices,OUT/'sheets'/f'{caseid}_input.jpg',f'{caseid} | recorded global frame indices | {ds} {model}',fps)
        # Reference survey is explicitly a diagnostic sample, not the evaluated model input.
        survey = sorted(set(round(j*(len(vr)-1)/15) for j in range(16)))
        sheet(vr,survey,OUT/'sheets'/f'{caseid}_survey.jpg',f'{caseid} | 16-frame diagnostic survey (NOT model input)',fps)
        times = [idx/fps for idx in indices]
        blind.append(dict(case_id=caseid,dataset=ds,model=model,input=inp,
            source_path=str(source),source_sha256=source_sha,duration_seconds=len(vr)/fps,
            recorded_global_frame_indices=indices,global_frame_times_seconds=times,
            max_global_frame_gap_seconds=max(b-a for a,b in zip(times,times[1:])) if len(times)>1 else None,
            recorded_audio_coverage=arm['memories'][0]['audio_coverage'],
            local_review_window=arm['review_window'], sheets=[f'sheets/{caseid}_input.jpg',f'sheets/{caseid}_survey.jpg'],
            factual_review_status='pending',audio_semantic_review='not_performed',
            note='Contact sheets show source frames before model resizing. Not proof of all events or audio content.'))
        trace.append(dict(case_id=caseid,memories=arm['memories'],
            calls=[{k:v for k,v in x.items() if k not in ('serialized_prompt','model_output')} for x in arm['calls']]))
        lookup.append(dict(case_id=caseid,category=c['category'],offline_score=c['offline_score']))
        print(caseid, ds, model, 'media SHA and global indices verified', flush=True)
    for name, data in [('BLIND_REVIEW.json',blind),('OBSERVATION_TRACE.json',trace),('OFFLINE_OUTCOMES.json',lookup)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    page = ['<!doctype html><meta charset="utf-8"><title>V11 evidence review</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}section{margin-bottom:70px;max-width:1250px}pre{white-space:pre-wrap}</style>',
        '<h1>V11 evidence review: 48 cases</h1><p>No ground-truth answers or outcome strata appear here. Review source evidence before consulting observation traces or the separate offline outcomes. Audio semantics have not been reviewed. Sparse frames cannot establish speaker identity or speech timing.</p>']
    for b in blind:
        page += [f'<section><h2>{b["case_id"]} · {b["dataset"]} / {b["model"]}</h2>',
                 '<pre>'+html.escape(b['input']['question_prompt'])+'</pre>',
                 '<p>'+html.escape(b['source_path'])+'</p>',
                 f'<p>Duration {b["duration_seconds"]:.3f}s; global frame gap up to {b["max_global_frame_gap_seconds"]:.3f}s. Audio intervals: {b["recorded_audio_coverage"]}. Local review: {b["local_review_window"]}.</p>',
                 f'<img src="sheets/{b["case_id"]}_input.jpg"><details><summary>Diagnostic survey (not evaluated input)</summary><img src="sheets/{b["case_id"]}_survey.jpg"></details></section>']
    (OUT/'REVIEW.html').write_text('\n'.join(page),encoding='utf-8')
    manifest = dict(source_archive_sha256=expected, cases=len(blind), unique_clips=len({b['source_path'] for b in blind}),
        cpu_preparation_seconds=time.time()-started, gpu_used=False,
        limitation='All 48 source hashes and recorded global frame indices checked; no semantic audio or full-motion review claimed.',
        artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ('.jpg','.json','.html')})
    (OUT/'BUILD_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive_out = OUT/'review_materials.tar.gz'
    with tarfile.open(archive_out,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p != archive_out: tar.add(p,arcname=str(p.relative_to(OUT)))
    print('ARCHIVE',sha(archive_out),archive_out.stat().st_size,flush=True)


if __name__ == '__main__':
    main()
