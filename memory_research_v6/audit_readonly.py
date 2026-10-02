"""Post-hoc stage decomposition. CPU only; never imported by inference."""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent
sys.path.insert(0,str(EXPERIMENT/'stage_roundoff_v2'))
from common_stage import ROOT,V5
from evaluate import answer


def snapshot(path,sources):
    raw=path.read_bytes()
    sources[str(path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    assert not raw or raw.endswith(b'\n'), 'Retry read after current JSONL append completes'
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def main():
    out=HERE/'runs';out.mkdir(exist_ok=True)
    sources={};results={};inventory={}
    for ds in ['joint','avspeaker']:
        rows=snapshot(V5/f'data/{ds}.jsonl',sources);inputs={r['qid']:r for r in rows}
        durations=json.loads((V5/f'manifests/{ds}_durations.json').read_text())
        seconds=[float(v['duration_seconds'] if isinstance(v,dict) else v) for v in (durations[r['qid']] for r in rows)]
        counts=collections.Counter(r['clip_path'] for r in rows)
        inventory[ds]=dict(questions=len(rows),distinct_paths=len(counts),paths_with_multiple_questions=sum(n>1 for n in counts.values()),
                           median_duration_seconds=sorted(seconds)[len(seconds)//2],at_most_12_seconds=sum(s<=12 for s in seconds),
                           caveat='Path count is not an exact-content deduplication audit; distinct source-video clusters cannot be treated as identical released clips.')
        labels_path=ROOT/('reproduction/avspeaker_v1/labels.json' if ds=='avspeaker' else 'reproduction/frozen_seed42/labels.json')
        labels_raw=labels_path.read_bytes();labels=json.loads(labels_raw)
        sources[str(labels_path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(labels_raw).hexdigest(),bytes=len(labels_raw))
        for model in ['omni','videollama']:
            arms={}
            for mode in ['baseline','agent']:
                root=EXPERIMENT/f'stage_roundoff_v2/runs/{ds}/{model}/{mode}'
                arms[mode]={}
                for f in [root/'inherited.jsonl']+sorted(root.glob('gpu*/predictions.jsonl')):
                    for r in snapshot(f,sources):
                        if r['status']!='ok':continue
                        assert r['qid'] not in arms[mode] and r['qid'] in inputs
                        arms[mode][r['qid']]=r['arm']
            d=collections.Counter(n=len(arms['agent']));bins=collections.Counter();category={}
            for q,a in arms['agent'].items():
                y=labels[q] if ds=='avspeaker' else labels[q]['correct_prefix']
                m=a['memory'];first=m['candidate'];final=answer(ds,a['model_output'],inputs[q])[0]
                base=answer(ds,arms['baseline'][q]['model_output'],inputs[q])[0]
                event=dict(baseline_correct=int(base==y),initial_correct=int(first==y),final_correct=int(final==y),
                    review_count=int(a['review']),review_wrong_to_right=int(first!=y and final==y),review_right_to_wrong=int(first==y and final!=y),
                    initial_wrong_to_right_vs_baseline=int(base!=y and first==y),initial_right_to_wrong_vs_baseline=int(base==y and first!=y),
                    evidence_limited=int(m.get('evidence_limited',False)))
                d.update(event);bins[str(m.get('selected_bin'))]+=1
                cat=category.setdefault(inputs[q]['task'],collections.Counter());cat.update(event);cat['n']+=1
            assert d['initial_correct']-d['baseline_correct']==d['initial_wrong_to_right_vs_baseline']-d['initial_right_to_wrong_vs_baseline']
            assert d['final_correct']-d['initial_correct']==d['review_wrong_to_right']-d['review_right_to_wrong']
            assert sum(bins.values())==d['n'] and sum(v['n'] for v in category.values())==d['n']
            results[ds+'/'+model]=dict(complete=d['n']==len(inputs),planned=len(inputs),stats=dict(d),seek=dict(bins),by_task={k:dict(v) for k,v in category.items()})
    report=dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),inventory=inventory,results=results,sources=sources,
       interpretation='Post-hoc accounting, not randomized causal attribution. Initial vs baseline changes prompt and constrained decoding together; review changes input and context. Partial cohorts must use matched baseline qids. No new inference or training.',
       dataset_role='Previously inspected development20%, not an untouched validation set. Labels loaded only by this offline analysis.')
    (out/'stage_decomposition.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(inventory=inventory,results={k:{a:v[a] for a in ['complete','planned','stats','seek']} for k,v in results.items()}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
