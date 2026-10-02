"""Independent frozen scorer, paired comparisons, and cold application-cache timing."""
import argparse
import collections
import importlib.util
import itertools
import json
import math
import random
import statistics
from support import HERE, ROOT, write, readlines
from joint_parser import parse as joint_parse

spec=importlib.util.spec_from_file_location('avscore',HERE.parent/'compat/avspeaker/common.py')
av=importlib.util.module_from_spec(spec);spec.loader.exec_module(av)
MODES=['baseline','fixed','agent']


def answer(ds,text,row):
    if ds=='joint':
        a=joint_parse(text,row['prefix2text'])['answer'];return a,a
    return av.upstream_parse(text),av.strict_parse(text)


def old_records(ds,model):
    paths=([ROOT/f'results/custom_v1_20261002/{model}/full/gpu{g}/predictions.jsonl' for g in [0,3]] if ds=='joint'
           else [ROOT/f'results/avspeaker_v1_20261002/{model}/full/predictions.jsonl'])
    return {r['qid']:r for p in paths for r in readlines(p) if r['status']=='ok'}


def pilot_gate(ds,model):
    records=readlines(HERE/f'runs/{ds}/{model}/pilot/predictions.jsonl')
    old=old_records(ds,model);checks=[]
    for r in records:
        assert r['status']=='ok'
        baseline=r['arms']['baseline']['calls'][0];orig=old[r['qid']]
        keys=['frames','audio_sha256','video_tensor_sha256','raw_frames_sha256','input_tokens','audio_tokens','video_tokens']
        parity={k:baseline.get(k)==orig.get(k) for k in keys}
        assert all(parity.values()),('native_baseline_parity',r['qid'],parity)
        assert baseline['model_output']==orig['model_output'],('baseline_text_changed',r['qid'])
        assert all(r['baseline_repeat_matches'].values()) and all(r['memory_repeat_matches'].values())
        finals={mode:r['arms'][mode]['calls'][-1]['reached_token_limit'] for mode in MODES}
        assert not any(finals.values()),('final_output_truncation',r['qid'])
        assert r['arms']['fixed']['calls'][0]['model_output']==r['arms']['agent']['calls'][0]['model_output'],'identical initial prompts must repeat'
        assert all(answer(ds,r['arms'][m]['model_output'],{'prefix2text':{}} if ds=='joint' else {})[0] is not None for m in ['fixed','agent']), 'Pilot final output unreadable'
        checks.append(dict(qid=r['qid'],baseline_media_parity=parity,baseline_text_identical=True,
                           memory_schema_valid=r['arms']['agent']['memory']['schema_valid'],agent_review=r['arms']['agent']['review'],
                           cache_repeat=r['baseline_repeat_matches'],memory_repeat=r['memory_repeat_matches'],final_limits=finals))
    assert len(checks)==2 and any(x['memory_schema_valid'] for x in checks),'Memory schema unusable in pilot; diagnose independently'
    write(HERE/f'runs/{ds}/{model}/pilot/gate.json',dict(passed=True,checks=checks,criterion='input/token/text parity, cache equivalence, repeatability, tool/schema execution; no accuracy threshold'))


def percentile(values,q):
    if not values:return None
    vals=sorted(values);index=(len(vals)-1)*q;lo=math.floor(index);hi=math.ceil(index)
    return vals[lo]+(vals[hi]-vals[lo])*(index-lo)


def paired(rows,a,b):
    gain=sum(r[a] and not r[b] for r in rows);loss=sum(r[b] and not r[a] for r in rows)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['cluster']].append(int(r[a])-int(r[b]))
    clusters=[(sum(v),len(v)) for v in groups.values()]
    rng=random.Random(42);draws=[]
    for _ in range(2000):
        sample=rng.choices(clusters,k=len(clusters));draws.append(100*sum(s for s,n in sample)/sum(n for s,n in sample))
    return dict(n=len(rows),wrong_to_right=gain,right_to_wrong=loss,net_correct=gain-loss,
                delta_pp=100*(gain-loss)/len(rows),cluster_bootstrap_95pct_pp=[percentile(draws,.025),percentile(draws,.975)],clusters=len(clusters))


def main():
    table=[];details={};pairs={};setup={}
    for ds in ['avspeaker','joint']:
        rows=readlines(HERE/'data'/f'{ds}.jsonl')
        lp=ROOT/('reproduction/frozen_seed42/labels.json' if ds=='joint' else 'reproduction/avspeaker_v1/labels.json')
        if not lp.exists():lp=ROOT/'results/avspeaker_v1_20261002/evidence_snapshot/reproduction/avspeaker_v1/labels.json'
        labels=json.loads(lp.read_text(encoding='utf-8'))
        for model in ['omni','videollama']:
            dest=HERE/f'runs/{ds}/{model}/dev20';records=readlines(dest/'predictions.jsonl')
            ok={r['qid']:r for r in records if r['status']=='ok'}
            assert len(ok)==sum(r['status']=='ok' for r in records),'duplicate successful qid'
            full=len(ok)==len(rows);parsed=[];key=ds+'/'+model
            starts=[json.loads(p.read_text()) for p in dest.glob('worker_start_*.json')]
            setup[key]=dict(worker_initializations=starts,total_model_load_seconds=sum(s['load_seconds'] for s in starts),
                            recorded_question_wall_seconds=sum(r['seconds'] for r in records),
                            failure_attempt_seconds=sum(r['seconds'] for r in records if r['status']!='ok'))
            for row in rows:
                q=row['qid'];y=labels[q]['correct_prefix'] if ds=='joint' else labels[q]
                item=dict(qid=q,task=row['task'],label=y,complete=q in ok,
                          cluster=row.get('video_name',row['clip_path'].rsplit('/',1)[-1].rsplit('_',2)[0]))
                for mode in MODES:
                    text=ok[q]['arms'][mode]['model_output'] if q in ok else None
                    a,s=answer(ds,text,row) if text else (None,None)
                    item[mode]=a;item[mode+'_strict']=s;item[mode+'_correct']=a==y;item[mode+'_strict_correct']=s==y
                parsed.append(item)
            observed=[r for r in parsed if r['complete']]
            taskrows=[]
            for mode in MODES:
                arms=[ok[r['qid']]['arms'][mode] for r in observed]
                calls=[c for arm in arms for c in arm['calls']]
                times=[arm['seconds'] for arm in arms]
                correct=sum(r[mode+'_correct'] for r in observed);n=len(observed)
                nparsed=sum(r[mode] is not None for r in observed)
                strictparsed=sum(r[mode+'_strict'] is not None for r in observed)
                strictcorrect=sum(r[mode+'_strict_correct'] for r in observed)
                table.append(dict(dataset=ds,model=model,method=mode,complete=full,planned_n=len(rows),observed_n=n,
                    correct=correct,accuracy_pct=100*correct/len(rows) if full else None,
                    partial_observed_accuracy_pct=100*correct/n if n else None,strict_full_denominator_pct=100*correct/len(rows),
                    unparsed=n-nparsed,parsed_only_pct=100*correct/nparsed if nparsed else None,
                    secondary_strict_correct=strictcorrect,secondary_strict_unparsed=n-strictparsed,
                    secondary_strict_full_denominator_pct=100*strictcorrect/len(rows),
                    missing=len(rows)-n,historical_error_attempts=sum(r['status']=='error' for r in records),
                    cold_application_mean_seconds=statistics.mean(times) if n else None,cold_application_p50_seconds=percentile(times,.5),
                    cold_application_p95_seconds=percentile(times,.95),cold_application_total_seconds=sum(times),
                    standalone_total_with_shared_model_load_seconds=sum(times)+setup[key]['total_model_load_seconds'],
                    mean_calls=len(calls)/n if n else None,max_calls=max([len(a['calls']) for a in arms],default=0),
                    reviews=sum(a.get('review',False) for a in arms),invalid_memory=sum(not a.get('memory',{}).get('schema_valid',True) for a in arms),
                    tool_fallbacks=sum(a.get('review',False) and a.get('memory',{}).get('tool_fallback',False) for a in arms) if mode=='agent' else 0,
                    evidence_token_cap_reached=sum(a.get('memory',{}).get('evidence_limited',False) for a in arms),
                    crop_seconds=sum(c['crop_seconds'] for c in calls),cache_build_seconds=sum(c['cache_build_seconds'] for c in calls),
                    processor_and_generation_seconds=sum(c['processor_and_generation_seconds'] for c in calls),
                    cache_hits=sum(c['cache_hit'] for c in calls),mean_input_tokens=sum(c['input_tokens'] for c in calls)/n if n else None,
                    mean_generated_tokens=sum(c['generated_tokens'] for c in calls)/n if n else None,
                    mean_frames=sum(c['frames'] for c in calls)/n if n else None,
                    mean_encoded_audio_seconds=sum(c.get('audio_seconds_encoded',sum(b-a for a,b in c.get('native_audio_clip_timepoints',[]))) for c in calls)/n if n else None,
                    final_truncations=sum(a['calls'][-1]['reached_token_limit'] for a in arms),
                    intermediate_truncations=sum(c['reached_token_limit'] for a in arms for c in a['calls'][:-1])))
                for task in sorted({r['task'] for r in rows}):
                    sub=[r for r in parsed if r['task']==task]
                    taskrows.append(dict(method=mode,task=task,denominator=len(sub),observed=sum(r['complete'] for r in sub),
                                         correct=sum(r[mode+'_correct'] for r in sub),accuracy_pct=100*sum(r[mode+'_correct'] for r in sub)/len(sub) if full else None))
            details[key]=taskrows
            pairs[key]=dict(complete=full,observed=len(observed),planned=len(rows))
            if full:
                for a,b in [('agent','baseline'),('fixed','baseline'),('agent','fixed')]:
                    pairs[key][a+'_vs_'+b]=paired(parsed,a+'_correct',b+'_correct')
                    pairs[key][a+'_vs_'+b+'_secondary_strict']=paired(parsed,a+'_strict_correct',b+'_strict_correct')
            write(HERE/f'runs/reports/{ds}_{model}_paired_questions.json',parsed)
    write(HERE/'runs/reports/TOTAL_TABLE.json',table);write(HERE/'runs/reports/TASK_DETAILS.json',details)
    write(HERE/'runs/reports/PAIRED_SUMMARY.json',pairs);write(HERE/'runs/reports/SETUP_TIMES.json',setup)
    lines=['V5 three-arm fixed20% development experiment. Baseline is re-run, not historical replay.',
           'Each arm starts with empty application media cache. Timings include crop/cache build/processor/generation; OS cache is uncontrolled.',
           'All arms share loaded weights; model initialization is listed separately. Standalone totals add that common cost, and must not be summed across arms.',
           'Fixed: two calls and one midpoint12s review. Agent: one or two calls with bounded working memory; no cross-question answer memory.',
           'Partial results are descriptive only. No effect claims until complete paired evaluation. No automatic80% or full run.', '']
    for r in table:lines.append(f"{r['dataset']} {r['model']} {r['method']} {r['observed_n']}/{r['planned_n']} accuracy={r['accuracy_pct']} cold_mean={r['cold_application_mean_seconds']} p95={r['cold_application_p95_seconds']} calls={r['mean_calls']}")
    (HERE/'runs/reports/REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return all(r['complete'] for r in table)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot-gate',nargs=2);a=p.parse_args()
    if a.pilot_gate:pilot_gate(*a.pilot_gate)
    else:main()
