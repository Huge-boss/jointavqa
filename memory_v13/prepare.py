"""Freeze inputs and whitelist native cache fields. No labels are loaded."""
import json,time
from settings import *

def main():
    assert not (HERE/'runs/manifest.json').exists(), 'Never refreeze an existing version'
    native=json.loads((V5/'manifests/code_freeze.json').read_text())
    deps={str((V5/p).relative_to(ROOT)):h for p,h in native['python_sha256'].items()}
    deps.update(native['external_sha256'])
    deps[str((HERE.parent/'stage_roundoff_v2/roundoff_fix.py').relative_to(ROOT))]=sha(HERE.parent/'stage_roundoff_v2/roundoff_fix.py')
    assets={};sources={};pilots={};partitions={}
    original_pilots=json.loads((V5/'manifests/pilot.json').read_text())
    for ds in ['joint','avspeaker']:
        rows=readlines(V5/f'data/{ds}.jsonl');ids=[r['qid'] for r in rows]
        assert len(ids)==len(set(ids))==(571 if ds=='joint' else 642)
        assert all(not (set(r)&{'label','answer','correct_answer','correct_prefix','answer_label'}) for r in rows)
        jsonl(HERE/f'data/{ds}.jsonl',rows)
        assets[f'data/{ds}.jsonl']=sha(HERE/f'data/{ds}.jsonl')
        dp=V5/f'manifests/{ds}_durations.json'
        write(HERE/f'data/{ds}_durations.json',json.loads(dp.read_text()))
        assets[f'data/{ds}_durations.json']=sha(HERE/f'data/{ds}_durations.json')
        selected=list(original_pilots[ds]['qids']);tasks={r['task'] for r in rows if r['qid'] in selected}
        for r in rows:
            if len(selected)==6:break
            if r['qid'] not in selected and r['task'] not in tasks:
                selected.append(r['qid']);tasks.add(r['task'])
        assert len(selected)==6
        pilots[ds]=selected
        partitions[ds]={g:ids[i::3] for i,g in enumerate(GPUS)}
        for model in ['omni','videollama']:
            p=HERE.parent/f'stage_roundoff_v2/runs/{ds}/{model}/baseline/inherited.jsonl'
            digest=sha(p);sources[str(p.relative_to(ROOT))]=digest
            old={r['qid']:r for r in readlines(p) if r['status']=='ok'}
            assert set(old)==set(ids)
            sanitized=[]
            for q in ids:
                a=old[q]['arm'];call=a['calls'][0]
                sanitized.append(dict(qid=q,model_output=a['model_output'],seconds=a['seconds'],
                    media_reference={k:call.get(k) for k in ['model_output','generated_tokens','frames','input_tokens','audio_tokens','video_tokens','audio_sha256','video_tensor_sha256','raw_frames_sha256']},
                    source=dict(path=str(p.relative_to(ROOT)),sha256=digest,origin=old[q]['origin'])))
            path=HERE/f'data/{ds}_{model}_native.jsonl';jsonl(path,sanitized)
            assets[str(path.relative_to(HERE))]=sha(path)
    write(HERE/'runs/manifest.json',dict(created_at=time.time(),code=hashes(),protocol=sha(HERE/'protocol.json'),
          dependencies=deps,assets=assets,native_sources=sources,pilot_qids=pilots,partitions=partitions,
          note='Same development20; no held-out inputs or labels loaded. Immutable native outputs are a reused model stage, not factual memory.'))
    frozen();print('Frozen: 571+642 inputs, two native caches each, GPUs 0/1/3')

if __name__=='__main__':main()
