"""Independent scoring and provenance-aware timing; never imported by inference."""
import json,time,statistics,collections
from common_stage import *
from evaluate import answer,paired,percentile


def main():
    table=[];pairtable={};tasks={};all_complete=True
    c=config()
    for rel,h in c['inherited_sha256'].items():assert sha(HERE/rel)==h
    for ds in ['avspeaker','joint']:
        inputs=readlines(V5/'data'/f'{ds}.jsonl');n=len(inputs)
        labels=json.loads((ROOT/('reproduction/avspeaker_v1/labels.json' if ds=='avspeaker' else 'reproduction/frozen_seed42/labels.json')).read_text())
        for model in ['omni','videollama']:
            records={};scored={};taskrows=[];pairrows=[]
            for mode in ['baseline','fixed','agent']:
                ok,errors=arm_records(ds,model,mode);assert set(ok)<={r['qid'] for r in inputs}
                records[mode]=ok;scored[mode]={}
                for row in inputs:
                    q=row['qid'];y=labels[q] if ds=='avspeaker' else labels[q]['correct_prefix']
                    a,s=answer(ds,ok[q]['arm']['model_output'],row) if q in ok else (None,None)
                    scored[mode][q]=dict(answer=a,strict=s,correct=a==y,strict_correct=s==y)
                arms=[r['arm'] for r in ok.values()];times=[a['seconds'] for a in arms];calls=[call for a in arms for call in a['calls']]
                complete=len(ok)==n;all_complete=all_complete and complete
                correct=sum(scored[mode][q]['correct'] for q in ok);unparsed=sum(scored[mode][q]['answer'] is None for q in ok)
                origins={}
                for kind in sorted({r['origin']['kind'] for r in ok.values()}):
                    rr=[r for r in ok.values() if r['origin']['kind']==kind]
                    origins[kind]=dict(n=len(rr),recorded_work_seconds=sum(r['arm']['seconds'] for r in rr))
                dest=HERE/f'runs/{ds}/{model}/{mode}'
                starts=[json.loads(p.read_text()) for p in dest.glob('gpu*/worker_start_*.json')]
                d=dict(dataset=ds,model=model,method=mode,planned_n=n,observed_n=len(ok),complete=complete,correct=correct,
                       accuracy_pct=100*correct/n if complete else None,partial_observed_pct=100*correct/len(ok) if ok else None,
                       missing=n-len(ok),unparsed=unparsed,parsed_only_pct=100*correct/(len(ok)-unparsed) if len(ok)>unparsed else None,
                       secondary_strict_correct=sum(scored[mode][q]['strict_correct'] for q in ok),
                       mean_seconds=statistics.mean(times) if times else None,p95_seconds=percentile(times,.95),
                       whole_question_work_seconds=sum(times),mean_calls=len(calls)/len(ok) if ok else None,
                       reviews=sum(a['review'] for a in arms),errors=len(errors),historical_errors=sum(1 for e in readlines(HERE/'runs/historical_errors.jsonl') if (e['dataset'],e['model'],e['mode'])==(ds,model,mode)),final_truncations=sum(a['calls'][-1].get('reached_token_limit',False) for a in arms),
                       cache_build_seconds=sum(call['cache_build_seconds'] for call in calls) if calls and all('cache_build_seconds' in call for call in calls) else None,
                       crop_seconds=sum(call.get('crop_seconds',0) for call in calls) if mode!='baseline' else None,
                       origin_timing=origins,new_model_load_seconds_sum=sum(s['load_seconds'] for s in starts),worker_initializations=starts,
                       timing_note='Whole question time includes preprocessing; historical baseline lacks separate cache/build and startup breakdown. Mean/P95 pool explicitly listed source epochs; sum work is not two-GPU wall time.')
                table.append(d)
                for task in sorted({r['task'] for r in inputs}):
                    qs=[r['qid'] for r in inputs if r['task']==task]
                    taskrows.append(dict(method=mode,task=task,denominator=len(qs),observed=sum(q in ok for q in qs),correct=sum(scored[mode][q]['correct'] for q in qs),accuracy_pct=100*sum(scored[mode][q]['correct'] for q in qs)/len(qs) if complete else None))
            for row in inputs:
                q=row['qid'];r=dict(qid=q,task=row['task'],cluster=row.get('video_name',row['clip_path'].rsplit('/',1)[-1].rsplit('_',2)[0]),complete=all(q in records[m] for m in records))
                for m in records:r[m]=scored[m][q]['answer'];r[m+'_correct']=scored[m][q]['correct'];r[m+'_strict_correct']=scored[m][q]['strict_correct']
                pairrows.append(r)
            full=all(r['complete'] for r in pairrows);p=dict(complete=full,paired_n=sum(r['complete'] for r in pairrows),planned=n)
            if full:
                for a,b in [('agent','baseline'),('fixed','baseline'),('agent','fixed')]:
                    p[a+'_vs_'+b]=paired(pairrows,a+'_correct',b+'_correct')
                    p[a+'_vs_'+b+'_strict']=paired(pairrows,a+'_strict_correct',b+'_strict_correct')
            pairtable[ds+'/'+model]=p;tasks[ds+'/'+model]=taskrows
            write(HERE/f'runs/reports/{ds}_{model}_paired_questions.json',pairrows)
    write(HERE/'runs/reports/TOTAL_TABLE.json',table);write(HERE/'runs/reports/PAIRED_SUMMARY.json',pairtable);write(HERE/'runs/reports/TASK_DETAILS.json',tasks)
    stages=readlines(HERE.parent/'stage_v1/runs/stage_history.jsonl')+readlines(HERE/'runs/stage_history.jsonl')
    write(HERE/'runs/reports/EXECUTION_COSTS.json',dict(stages=stages,prior_v5_launch=json.loads((V5/'runs/full_launch.json').read_text()),prior_v5_stop=json.loads((V5/'runs/stop_for_stage_schedule.json').read_text()),
          note='Preserve prior computation cost. Stage walls include fresh model loads, cache construction, crop and GPU execution. Historical baseline score/timing imported, no new baseline calls. Check stages are additional validation cost.'))
    lines=['双模型三组20%对照：按模型、按方法双卡分题续跑','',
      '方法仍是冻结v5；独立版本仅修复片段上限浮点误判，实际窗口和ffmpeg参数不变。历史baseline全部复用；已完成固定/Agent逐题保留，并记录来源，缺失题才补跑。',
      'Omni固定→Omni Agent→VideoLLaMA固定→VideoLLaMA Agent。每阶段GPU0/3各载入完整模型，处理互不重叠题目。',
      '平均/P95包含每题预处理和缓存构建。历史baseline与续跑计时来源不同，详见origin_timing。模型加载单列；两卡墙钟见EXECUTION_COSTS，不把两卡工作秒数相加当墙钟。',
      '历史baseline缺少单独缓存/加载细分，未知项为null；不将既有baseline结果当作零成本，也不声称精确的同轮冷启动加速比。',
      '这是反复开发用20%，不作为独立盲测；只有完整对照才显示准确率和配对差值。','']
    for r in table:
        lines.append(f"{r['dataset']} {r['model']} {r['method']}: {r['observed_n']}/{r['planned_n']} 正确{r['correct']} 准确率{r['accuracy_pct']} 平均秒{r['mean_seconds']} P95秒{r['p95_seconds']} 调用{r['mean_calls']} 未解析{r['unparsed']} 错误{r['errors']}")
    for k,v in pairtable.items():
        if v['complete']:
            for comparison in ['agent_vs_baseline','fixed_vs_baseline','agent_vs_fixed']:
                p=v[comparison];lines.append(f"{k} {comparison}: 改对{p['wrong_to_right']} 改错{p['right_to_wrong']} 净{p['net_correct']:+d} 差值{p['delta_pp']:+.4f}pp 区间{p['cluster_bootstrap_95pct_pp']}")
    (HERE/'runs/reports/REPORT_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return all_complete


if __name__=='__main__':main()
