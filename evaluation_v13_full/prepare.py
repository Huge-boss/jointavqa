"""Freeze full inputs and reuse sources without loading QA labels or scorers."""
import json,time,collections
from settings import *

REFKEYS=['model_output','generated_tokens','frames','input_tokens','audio_tokens','video_tokens','audio_sha256','video_tensor_sha256','raw_frames_sha256']

def main():
    assert not (HERE/'runs/manifest.json').exists(),'Never refreeze'
    v13=HERE.parent/'memory_v13';oldman=json.loads((v13/'runs/manifest.json').read_text())
    assert oldman['code']=={p.name:sha(p) for p in v13.glob('*.py')}
    assert oldman['protocol']==sha(v13/'protocol.json')
    for name in ['method.py','ledger_grammar.py']:assert sha(HERE/name)==sha(v13/name)
    deps=dict(oldman['dependencies'])
    for p in v13.glob('*.py'):deps[str(p.relative_to(ROOT))]=sha(p)
    deps[str((v13/'protocol.json').relative_to(ROOT))]=sha(v13/'protocol.json')
    deps[str((v13/'runs/manifest.json').relative_to(ROOT))]=sha(v13/'runs/manifest.json')
    assets={};sources={};partitions={};splits={};audit={}
    def source(p):sources[str(p.relative_to(ROOT))]=sha(p)
    def asset(p):assets[str(p.relative_to(HERE))]=sha(p)
    for ds,n,devn in [('joint',2853,571),('avspeaker',3212,642)]:
        p=ROOT/('reproduction/frozen_seed42/inputs.jsonl' if ds=='joint' else 'reproduction/avspeaker_v1/inputs.jsonl')
        split=json.loads((V5/f'manifests/{ds}_split.json').read_text());assert sha(p)==split['source_sha256'];source(p)
        rows=readlines(p);ids=[r['qid'] for r in rows];byid={r['qid']:r for r in rows}
        assert len(ids)==len(set(ids))==n
        assert all(not set(r)&{'label','answer','correct_answer','correct_prefix','answer_label'} for r in rows)
        dev=readlines(v13/f'data/{ds}.jsonl');dev_ids=[r['qid'] for r in dev]
        assert len(dev_ids)==devn and set(dev_ids)==set(split['qids'])
        assert all(byid[r['qid']]==r for r in dev)
        remaining=[q for q in ids if q not in set(dev_ids)]
        assert remaining==split['remaining_qids'] and len(remaining)==n-devn
        splits[ds]=dict(dev20=dev_ids,remaining80=remaining,full=ids)
        partitions[ds]={g:remaining[i::3] for i,g in enumerate(GPUS)}
        jsonl(HERE/f'data/{ds}.jsonl',rows);asset(HERE/f'data/{ds}.jsonl')
        durations={};media=None
        if ds=='avspeaker':
            p=ROOT/'reproduction/avspeaker_v1/media_probe.json';source(p);media=json.loads(p.read_text())
            for r in rows:
                item=media[r['clip_path']];v=next(s for s in item['streams'] if s['codec_type']=='video')
                durations[r['qid']]=float(v.get('duration',item['format']['duration']))
        for model in ['omni','videollama']:
            paths=([ROOT/f'results/custom_v1_20261002/{model}/full/gpu{g}/predictions.jsonl' for g in [0,3]] if ds=='joint' else [ROOT/f'results/avspeaker_v1_20261002/{model}/full/predictions.jsonl'])
            native={}
            for p in paths:
                source(p);digest=sha(p)
                for line,r in enumerate(readlines(p),1):
                    assert r['status']=='ok' and r['qid'] not in native
                    q=r['qid'];row=byid[q];assert r['question_prompt']==row['question_prompt']
                    opt='prefix2text' if ds=='joint' else 'options';assert r[opt]==row[opt]
                    assert 0<r['frames']<=128 and r['audio_tokens']>0 and r['video_tokens']>0
                    if ds=='joint':
                        d=float(r['duration_seconds'])
                        if q in durations:assert abs(d-durations[q])<1e-6
                        durations[q]=d
                    native[q]=dict(qid=q,model_output=r['model_output'],seconds=r['elapsed_seconds'] if ds=='joint' else r['seconds'],
                        media_reference={k:r.get(k) for k in REFKEYS},source=dict(path=str(p.relative_to(ROOT)),sha256=digest,line=line,origin='historical_native_full'))
            assert set(native)==set(ids)
            oldnative={r['qid']:r for r in readlines(v13/f'data/{ds}_{model}_native.jsonl')}
            for q,r in oldnative.items():
                assert native[q]['model_output']==r['model_output'] and native[q]['seconds']==r['seconds']
                assert native[q]['media_reference']==r['media_reference']
            p=HERE/f'data/{ds}_{model}_native.jsonl';jsonl(p,[native[q] for q in ids]);asset(p)
            inherited=[]
            for p in sorted((v13/f'runs/{ds}/{model}/dev20').glob('gpu*/predictions.jsonl')):
                source(p)
                for r in readlines(p):
                    assert r['status']=='ok' and r['qid'] in set(dev_ids)
                    inherited.append(dict(r,reuse_origin=dict(path=str(p.relative_to(ROOT)),sha256=sha(p))))
            assert len(inherited)==len({r['qid'] for r in inherited})==devn
            p=HERE/f'data/{ds}_{model}_v13_dev20.jsonl';jsonl(p,inherited);asset(p)
        olddur=json.loads((v13/f'data/{ds}_durations.json').read_text())
        for q,d in olddur.items():assert abs(durations[q]-float(d['duration_seconds'] if isinstance(d,dict) else d))<1e-6,(ds,q)
        p=HERE/f'data/{ds}_durations.json';write(p,durations);asset(p)
        video=lambda r:r.get('video_name',r.get('video_id',r['clip_path']))
        dv={video(byid[q]) for q in dev_ids};rv={video(byid[q]) for q in remaining}
        audit[ds]=dict(full=n,reused_dev20=devn,new_remaining80=n-devn,tasks=dict(collections.Counter(r['task'] for r in rows)),shared_video_clusters_between_splits=len(dv&rv))
    prior_audit=v13/'runs/completed_audit/AUDIT.json';source(prior_audit)
    history=dict(source=str(prior_audit.relative_to(ROOT)),source_sha256=sha(prior_audit),
        v13_formal_wall_seconds=3767.315656,v13_formal_load_seconds=166.297375,
        v13_pilot_wall_seconds=185.248179,v13_pilot_question_work_seconds=274.874905,v13_pilot_load_seconds=163.902674,
        v13_extra_four_path_probe_work_seconds=39.518128,v13_extra_four_path_probe_load_seconds=53.119872,
        note='Reused historical technical validation and dev20 setup, not new full-evaluation execution. Work sums and elapsed times are not interchangeable. Earlier failed method development is additional history.')
    p=HERE/'data/setup_history.json';write(p,history);asset(p)
    write(HERE/'runs/manifest.json',dict(created_at=time.time(),code=hashes(),protocol=sha(HERE/'protocol.json'),dependencies=deps,assets=assets,sources=sources,partitions=partitions,splits=splits,preflight=audit,algorithm='V13 byte-identical method and grammar; scheduling/data scope only'))
    frozen();write(HERE/'runs/preflight.json',audit);print(json.dumps(audit))

if __name__=='__main__':main()
