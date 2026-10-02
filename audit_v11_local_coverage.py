"""Offline CPU replay provenance and interval audit. No labels or inference."""
import json
import subprocess
import tarfile
import time
from pathlib import Path
from prepare_v11_media_review import sha, sheet

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'offline_v11_media_review'
OUT=SRC/'local_coverage_audit'

def union(spans, duration):
    merged=[]
    for a,b in sorted((max(0.,float(a)),min(duration,float(b))) for a,b in spans):
        if b<=a:continue
        if merged and a<=merged[-1][1]+1e-7:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return merged

def seconds(spans):return sum(b-a for a,b in spans)

def main():
    from decord import VideoReader,cpu
    OUT.mkdir(exist_ok=True)
    (OUT/'sheets').mkdir(exist_ok=True)
    (OUT/'crops').mkdir(exist_ok=True)
    cases=json.loads((SRC/'BLIND_REVIEW.json').read_text(encoding='utf-8'))
    traces={r['case_id']:r for r in json.loads((SRC/'OBSERVATION_TRACE.json').read_text(encoding='utf-8'))}
    start=time.time(); records=[]; source_hashes={}
    for case in cases:
        cid=case['case_id']; duration=case['duration_seconds']; source=Path(case['source_path'])
        if source not in source_hashes:source_hashes[source]=sha(source)
        assert source_hashes[source]==case['source_sha256']
        trace=traces[cid]; global_spans=union(case['recorded_audio_coverage'],duration)
        local_mem=next((m for m in trace['memories'] if m['id']=='verification_local'),None)
        call=next((c for c in trace['calls'] if c['stage']=='local_observation'),None)
        local_spans=union(local_mem['audio_coverage'],duration) if local_mem else []
        total_spans=union(global_spans+local_spans,duration)
        initial=next(c for c in trace['calls'] if c['stage']=='initial_memory')
        r=dict(case_id=cid,dataset=case['dataset'],model=case['model'],duration_seconds=duration,
            global_audio_union=global_spans,global_audio_unique_seconds=seconds(global_spans),
            global_audio_fraction=seconds(global_spans)/duration,
            local_audio_union=local_spans,combined_audio_union=total_spans,
            local_unique_new_audio_seconds=max(0.,seconds(total_spans)-seconds(global_spans)),
            focus_field=initial.get('memory_fields',{}).get('FOCUS'),
            local_review_window=case['local_review_window'],
            audio_semantics='not_reviewed',original_sha256=case['source_sha256'])
        if call:
            a,b=call['window'];crop=OUT/'crops'/f'{cid}.mov'
            if not crop.exists():
                subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-ss',str(a),'-i',str(source),
                    '-t',str(b-a),'-map','0:v:0','-map','0:a:0','-sn','-dn','-c:v','libx264','-preset','ultrafast',
                    '-crf','0','-threads','2','-c:a','pcm_s16le','-map_metadata','-1','-y',str(crop)],
                    check=True,capture_output=True,timeout=240)
            actual=sha(crop);expected=call['crop']['sha256']
            r.update(recreated_crop_sha256=actual,recorded_crop_sha256=expected,crop_byte_equal=actual==expected)
            if actual==expected:
                vr=VideoReader(str(crop),ctx=cpu(0),num_threads=2);fps=float(vr.get_avg_fps())
                vm=call.get('video_metadata',{})
                indices=call.get('frame_indices',vm.get('frames_indices'))
                assert len(indices)==call['frames'] and max(indices)<len(vr)
                dst=OUT/'sheets'/f'{cid}_local_input.jpg'
                sheet(vr,indices,dst,f'{cid} | SHA-matched replay crop; recorded local frames',fps,offset=a)
                r.update(local_sheet=str(dst.relative_to(OUT)),local_frame_times=[a+i/fps for i in indices])
        records.append(r)
        print(cid,'crop_match',r.get('crop_byte_equal'),'new_audio',round(r['local_unique_new_audio_seconds'],3),flush=True)
    # Prospective diagnostic windows from prior visible hand/mouth event and actual
    # recorded replay; never selected from labels or candidate answers.
    case=cases[5];vr=VideoReader(case['source_path'],ctx=cpu(0),num_threads=2);fps=float(vr.get_avg_fps())
    dense=[]
    for name,a,b in [('hand_event',163.24,179.24),('actual_replay',409.46,421.46)]:
        indices=sorted(set(min(len(vr)-1,round((a+j*.25)*fps)) for j in range(round((b-a)*4)+1)))
        for page,begin in enumerate(range(0,len(indices),32)):
            sub=indices[begin:begin+32];dst=OUT/'sheets'/f'C06_{name}_{page+1}.jpg'
            sheet(vr,sub,dst,f'C06 | diagnostic 4fps {name}; NOT evaluated input',fps)
            dense.append(dict(path=str(dst.relative_to(OUT)),frame_indices=sub,times=[i/fps for i in sub]))
    report=dict(cases=records,diagnostic_dense_sheets=dense,
        source_inputs_sha256={n:sha(SRC/n) for n in ['BLIND_REVIEW.json','OBSERVATION_TRACE.json']},
        gpu_used=False,labels_loaded=False,semantic_audio_reviewed=0,
        crop_matches=sum(r.get('crop_byte_equal',False) for r in records),
        crop_recreated=sum('crop_byte_equal' in r for r in records),
        cpu_seconds=time.time()-start,
        limitation='Byte equality establishes crop provenance, not correct temporal localization, audio interpretation, or answer correctness.')
    (OUT/'COVERAGE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files=[OUT/'COVERAGE_AUDIT.json']+sorted((OUT/'sheets').glob('*.jpg'))
    manifest={str(p.relative_to(OUT)):sha(p) for p in files}
    (OUT/'ARTIFACT_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=OUT/'local_review_materials.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files+[OUT/'ARTIFACT_SHA256.json']:tar.add(p,arcname=str(p.relative_to(OUT)))
    print(json.dumps({'crop_matches':report['crop_matches'],'crop_recreated':report['crop_recreated'],
        'cpu_seconds':report['cpu_seconds'],'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size}),flush=True)

if __name__=='__main__':main()
