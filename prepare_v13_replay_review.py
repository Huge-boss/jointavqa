"""Offline CPU source/crop evidence builder; no inference or scorer imports."""
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path
from prepare_v11_media_review import sha, sheet, DATA

BASE=Path(__file__).resolve().parent
OUT=BASE/'offline_v13_replay_review'

def main():
    from decord import VideoReader,cpu
    OUT.mkdir(exist_ok=True)
    for d in ['sheets','crops']:(OUT/d).mkdir(exist_ok=True)
    if (OUT/'BUILD_MANIFEST.json').exists():
        raise RuntimeError('Existing material build retained; inspect before any rebuild')
    cases=json.loads((BASE/'memory_v13/runs/completed_audit/REPLAY_DIAGNOSTIC_CASES.json').read_text())
    assert len(cases)==7
    start=time.perf_counter();records=[]
    for i,c in enumerate(cases):
        cid=f'R{i+1:02d}';source=(DATA[c['dataset']]/c['input']['clip_path']).resolve()
        assert source.is_relative_to(DATA[c['dataset']]) and source.is_file()
        assert sha(source)==c['memories'][0]['clip_sha256']
        g,l=c['calls'][:2];frames=g['ledger_media']['frames'];fs=c['scout_fields']
        anchor=next(j for j,f in enumerate(frames) if f['index']==fs['ANCHOR'])
        vr=VideoReader(str(source),ctx=cpu(0),num_threads=2);fps=float(vr.get_avg_fps())
        assert g.get('frame_indices',g.get('video_metadata',{}).get('frames_indices'))==[f['source_frame'] for f in frames]
        neighbors=frames[max(0,anchor-2):min(len(frames),anchor+3)]
        sheet(vr,[f['source_frame'] for f in neighbors],OUT/'sheets'/f'{cid}_anchor.jpg',f'{cid} actual global frames; anchor {fs["ANCHOR"]} {fs["RELATION"]}',fps)
        a,b=c['review_window'];crop=OUT/'crops'/f'{cid}.mov'
        if not crop.exists():
            subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-ss',str(a),'-i',str(source),'-t',str(b-a),'-map','0:v:0','-map','0:a:0','-sn','-dn','-c:v','libx264','-preset','ultrafast','-crf','0','-threads','2','-c:a','pcm_s16le','-map_metadata','-1','-y',str(crop)],check=True,capture_output=True,timeout=240)
        crop_sha=sha(crop);assert crop_sha==l['crop']['sha256']
        local=VideoReader(str(crop),ctx=cpu(0),num_threads=2);lfps=float(local.get_avg_fps())
        indices=l.get('frame_indices',l.get('video_metadata',{}).get('frames_indices'));assert len(indices)==l['frames'] and max(indices)<len(local)
        sheet(local,indices,OUT/'sheets'/f'{cid}_local.jpg',f'{cid} SHA-matched crop; actual local input frames',lfps,offset=a)
        # Generated timestamps are diagnostic hypotheses, never assumed event truth.
        extra=[]
        if i in [0,1]:
            claim=[86.,171.][i];chosen=[round(t*fps) for t in [claim-6,claim-3,claim,claim+3,claim+6]]
            sheet(vr,chosen,OUT/'sheets'/f'{cid}_claimed_time.jpg',f'{cid} diagnostic at unverified generated time; NOT model input',fps)
            extra=[dict(kind='unverified_generated_timestamp_diagnostic',seconds=claim,indices=chosen)]
        question='\n'.join(s for s in c['input']['question_prompt'].splitlines() if not re.match(r'^\s*[A-D][.\):]',s))
        records.append(dict(case_id=cid,qid=c['input']['qid'],dataset=c['dataset'],model=c['model'],question=question,
            source_path=str(source),source_sha256=sha(source),duration_seconds=len(vr)/fps,
            anchor=frames[anchor],relation=fs['RELATION'],global_neighbor_frames=neighbors,
            global_audio=g['ledger_media']['audio'],requested_window=[a,b],actual_crop_duration_seconds=len(local)/lfps,
            recorded_crop_sha256=l['crop']['sha256'],recreated_crop_sha256=crop_sha,
            local_frames=indices,local_source_times_seconds=[a+j/lfps for j in indices],
            local_audio=l['actual_source_audio'],diagnostic_extra=extra,
            semantic_audio='not_reviewed',visual_review='pending',
            note='No correct label, candidate or outcome in this material. Existing outcome context prevents independent blind-review claims.'))
        print(cid,'source/crop SHA and input frame mapping verified',flush=True)
    (OUT/'REVIEW_INPUTS.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files=[OUT/'REVIEW_INPUTS.json']+sorted((OUT/'sheets').glob('*.jpg'))
    manifest=dict(cases=len(records),source_crop_checks='7/7 matched',gpu_used=False,cpu_preparation_seconds=time.perf_counter()-start,
        semantic_audio_reviewed=0,artifacts={str(p.relative_to(OUT)):sha(p) for p in files},
        source_packet_sha256=sha(BASE/'memory_v13/runs/completed_audit/REPLAY_DIAGNOSTIC_CASES.json'))
    (OUT/'BUILD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=OUT/'review_materials.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in files+[OUT/'BUILD_MANIFEST.json']:t.add(p,arcname=str(p.relative_to(OUT)))
    print(json.dumps(dict(archive_sha256=sha(archive),cpu_seconds=manifest['cpu_preparation_seconds'],files=len(files))))

if __name__=='__main__':main()
