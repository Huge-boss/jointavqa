"""Independent offline scoring. Never imported by inference."""
import collections,json,statistics,time
from settings import *
from evaluate import answer,paired,percentile

def main():
    frozen();table=[];pairs={};details={};complete=True
    for ds in ['joint','avspeaker']:
        rows=readlines(HERE/f'data/{ds}.jsonl');n=len(rows)
        labels=json.loads((ROOT/('reproduction/frozen_seed42/labels.json' if ds=='joint' else 'reproduction/avspeaker_v1/labels.json')).read_text())
        for model in ['omni','videollama']:
            dest=HERE/f'runs/{ds}/{model}/dev20'
            raw=[r for p in sorted(dest.glob('gpu*/predictions.jsonl')) for r in readlines(p)]
            ok={r['qid']:r for r in raw if r['status']=='ok'}
            assert len(ok)==sum(r['status']=='ok' for r in raw) and set(ok)<={r['qid'] for r in rows}
            old={r['qid']:r for r in readlines(HERE/f'data/{ds}_{model}_native.jsonl')}
            full=len(ok)==n;complete=complete and full;scored=[]
            for row in rows:
                q=row['qid'];label=labels[q]['correct_prefix'] if ds=='joint' else labels[q]
                a,s=answer(ds,ok[q]['arm']['model_output'],row) if q in ok else (None,None)
                b,bs=answer(ds,old[q]['model_output'],row)
                scored.append(dict(qid=q,task=row['task'],label=label,observed=q in ok,
                    cluster=row.get('video_name',row.get('video_id',row['clip_path'])),
                    agent=a,baseline=b,agent_correct=a==label,baseline_correct=b==label,
                    agent_strict=s,baseline_strict=bs,agent_strict_correct=s==label,baseline_strict_correct=bs==label))
            observed=[r for r in scored if r['observed']];m=len(observed)
            arms=[r['arm'] for r in ok.values()];calls=[c for a in arms for c in a['calls']]
            times=[a['seconds'] for a in arms];newtimes=[a['new_work_seconds'] for a in arms]
            correct=sum(r['agent_correct'] for r in observed);unparsed=sum(r['agent'] is None for r in observed)
            starts=[json.loads(p.read_text()) for p in dest.glob('gpu*/worker_start_*.json')]
            t=dict(dataset=ds,model=model,method='source_memory_v7',planned=n,observed=m,complete=full,
                correct=correct,accuracy_pct=100*correct/n if full else None,partial_accuracy_pct=100*correct/m if m else None,
                baseline_correct=sum(r['baseline_correct'] for r in scored),baseline_accuracy_pct=100*sum(r['baseline_correct'] for r in scored)/n,
                baseline_matched_correct=sum(r['baseline_correct'] for r in observed),missing=n-m,unparsed=unparsed,
                parsed_only_accuracy_pct=100*correct/(m-unparsed) if m>unparsed else None,
                strict_correct=sum(r['agent_strict_correct'] for r in observed),strict_unparsed=sum(r['agent_strict'] is None for r in observed),
                errors=sum(r['status']!='ok' for r in raw),mean_seconds=statistics.mean(times) if times else None,p95_seconds=percentile(times,.95),
                incremental_mean_seconds=statistics.mean(newtimes) if newtimes else None,incremental_work_seconds=sum(newtimes),
                logical_total_work_seconds=sum(times),historical_native_work_seconds=sum(a['baseline_cost_seconds'] for a in arms),
                model_load_seconds=sum(s['load_seconds'] for s in starts),executed_calls=len(calls),
                mean_logical_calls=statistics.mean([a['logical_calls'] for a in arms]) if arms else None,
                reviews=sum(a['review'] for a in arms),decisions=dict(collections.Counter(a['decision'] for a in arms)),
                bad_scout_schema=sum(not a['scout_schema_valid'] for a in arms),
                final_call_truncations=sum(a['calls'][-1]['reached_token_limit'] for a in arms),
                intermediate_truncations=sum(c['reached_token_limit'] for a in arms for c in a['calls'][:-1]),
                preparation_seconds=sum(c['cache_build_seconds'] for c in calls),crop_seconds=sum(c['crop_seconds'] for c in calls),
                generated_tokens=sum(c['generated_tokens'] for c in calls),input_tokens=sum(c.get('input_tokens',0) for c in calls),
                input_token_note='VideoLLaMA adapter may not expose full expanded multimodal input count; raw metadata preserved; absent counts not comparable to Omni.',
                timing_note='Historical baseline cost plus measured incremental cold application-cache work. Different source epochs; not an exact same-run speed benchmark. Model initialization and pilot validation are additional.')
            table.append(t);key=ds+'/'+model
            pairs[key]=dict(complete=full,observed=m,planned=n,comparison=paired(observed,'agent_correct','baseline_correct') if observed else None,
                secondary_strict=paired(observed,'agent_strict_correct','baseline_strict_correct') if observed else None)
            details[key]=[]
            for task in sorted({r['task'] for r in rows}):
                sub=[r for r in scored if r['task']==task];obs=[r for r in sub if r['observed']]
                details[key].append(dict(task=task,denominator=len(sub),observed=len(obs),
                    correct=sum(r['agent_correct'] for r in obs),baseline_matched_correct=sum(r['baseline_correct'] for r in obs),
                    gain=sum(r['agent_correct'] and not r['baseline_correct'] for r in obs),loss=sum(r['baseline_correct'] and not r['agent_correct'] for r in obs)))
            write(HERE/f'runs/reports/{ds}_{model}_paired_questions.json',scored)
    write(HERE/'runs/reports/TOTAL_TABLE.json',table);write(HERE/'runs/reports/PAIRED_SUMMARY.json',pairs)
    write(HERE/'runs/reports/TASK_DETAILS.json',details)
    lines=['来源约束的音画证据记忆：完整方案 V7，固定20%开发集',
        '先完整方案评估，不做组件消融。开发集已多次使用，不能宣称独立测试或论文创新已验证。',
        '全局音画始终保留；局部观察有来源区间和音频覆盖；回答改变须独立分歧复核。',
        '基线复用已完成同题原始输出，不是零耗时。所有新增缓存构建/裁剪/调用均计时；另列模型加载和pilot成本。',
        '没有改变随机种子/选项/标签或删除失败题。生成观察仍可能幻觉；输入预算增加，不能把收益全部归因记忆。',
        '本轮GPU0/1/3，每卡一个完整模型，按模型分阶段。剩余80%未运行。','']
    for r in table:
        p=pairs[r['dataset']+'/'+r['model']]['comparison']
        lines.append(f"{r['dataset']} {r['model']}: {r['observed']}/{r['planned']} 完成={r['complete']} 准确率={r['accuracy_pct']} baseline={r['baseline_accuracy_pct']:.4f}% 未解析={r['unparsed']} 错误={r['errors']} 总成本均值={r['mean_seconds']}s P95={r['p95_seconds']}s")
        if p:lines.append(f"同已完成题目改对={p['wrong_to_right']} 改错={p['right_to_wrong']} 净={p['net_correct']}；未完成时仅为阶段统计。")
    (HERE/'runs/reports/REPORT_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return complete

if __name__=='__main__':main()
