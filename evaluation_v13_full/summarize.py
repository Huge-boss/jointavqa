"""Offline only: frozen v5 scoring of dev20, complement80 and full dataset."""
import collections,json,statistics,time
from settings import *
from evaluate import answer,paired,percentile

def main():
    c=frozen();table=[];pairs={};details={};complete=True
    for ds in ['joint','avspeaker']:
        rows=readlines(HERE/f'data/{ds}.jsonl')
        lp=ROOT/('reproduction/frozen_seed42/labels.json' if ds=='joint' else 'reproduction/avspeaker_v1/labels.json')
        labels=json.loads(lp.read_text());assert set(labels)=={r['qid'] for r in rows}
        for model in ['omni','videollama']:
            inherited=readlines(HERE/f'data/{ds}_{model}_v13_dev20.jsonl')
            dest=HERE/f'runs/{ds}/{model}/remaining80'
            raw=[r for p in sorted(dest.glob('gpu*/predictions.jsonl')) for r in readlines(p)]
            fresh=[r for r in raw if r['status']=='ok'];assert len(fresh)==len({r['qid'] for r in fresh})
            assert {r['qid'] for r in fresh}<=set(c['splits'][ds]['remaining80'])
            ok={r['qid']:r for r in inherited+fresh};assert len(ok)==len(inherited)+len(fresh)
            native={r['qid']:r for r in readlines(HERE/f'data/{ds}_{model}_native.jsonl')}
            scored=[]
            for row in rows:
                q=row['qid'];y=labels[q]['correct_prefix'] if ds=='joint' else labels[q]
                a,s=answer(ds,ok[q]['arm']['model_output'],row) if q in ok else (None,None)
                b,bs=answer(ds,native[q]['model_output'],row)
                scored.append(dict(qid=q,task=row['task'],label=y,observed=q in ok,cluster=row.get('video_name',row.get('video_id',row['clip_path'])),
                    agent=a,baseline=b,agent_strict=s,baseline_strict=bs,agent_correct=a==y,baseline_correct=b==y,
                    agent_strict_correct=s==y,baseline_strict_correct=bs==y,
                    source_partition='dev20_reused' if q in set(c['splits'][ds]['dev20']) else 'remaining80_new'))
            starts=[json.loads(p.read_text()) for p in dest.glob('gpu*/worker_start_*.json')]
            for scope in ['dev20','remaining80','full']:
                ids=set(c['splits'][ds][scope]);sub=[r for r in scored if r['qid'] in ids];n=len(sub)
                observed=[r for r in sub if r['observed']];m=len(observed);full=m==n
                if scope=='full':complete=complete and full
                arms=[ok[r['qid']]['arm'] for r in observed];calls=[x for a in arms for x in a['calls']]
                times=[a['seconds'] for a in arms];newtimes=[a['new_work_seconds'] for a in arms]
                correct=sum(r['agent_correct'] for r in sub);basecorrect=sum(r['baseline_correct'] for r in sub)
                key=f'{ds}/{model}/{scope}'
                t=dict(dataset=ds,model=model,scope=scope,method='frozen_event_ledger_v13',planned=n,observed=m,complete=full,
                    correct=correct,accuracy_pct=100*correct/n if full else None,baseline_correct=basecorrect,baseline_accuracy_pct=100*basecorrect/n,
                    missing=n-m,errors=sum(r['status']!='ok' and r['qid'] in ids for r in raw),unparsed=sum(r['agent'] is None for r in observed),
                    strict_correct=sum(r['agent_strict_correct'] for r in sub),baseline_strict_correct=sum(r['baseline_strict_correct'] for r in sub),
                    strict_unparsed=sum(r['agent_strict'] is None for r in observed),
                    mean_seconds=statistics.mean(times) if times else None,p95_seconds=percentile(times,.95),
                    incremental_mean_seconds=statistics.mean(newtimes) if newtimes else None,incremental_p95_seconds=percentile(newtimes,.95),
                    incremental_work_seconds=sum(newtimes),historical_native_work_seconds=sum(a['baseline_cost_seconds'] for a in arms),logical_total_work_seconds=sum(times),
                    new_eval_model_load_seconds=sum(s['load_seconds'] for s in starts) if scope!='dev20' else 0,
                    executed_calls=len(calls),mean_logical_calls=statistics.mean(a['logical_calls'] for a in arms) if arms else None,
                    reviews=sum(a['review'] for a in arms),decisions=dict(collections.Counter(a['decision'] for a in arms)),
                    final_call_truncations=sum(a['calls'][-1]['reached_token_limit'] for a in arms),intermediate_truncations=sum(x['reached_token_limit'] for a in arms for x in a['calls'][:-1]),
                    field_caps=dict(collections.Counter(k for a in arms for k,v in a['scout_fields'].get('field_token_caps',{}).items() if v)),
                    preparation_seconds=sum(x['cache_build_seconds'] for x in calls),crop_seconds=sum(x['crop_seconds'] for x in calls),generated_tokens=sum(x['generated_tokens'] for x in calls),
                    labels_sha256=sha(lp),timing_note='Dev20 and native are historical epochs; new complement includes cold application cache/CPU/IO. No same-epoch speedup claim; initialization separately.')
                table.append(t)
                pairs[key]=dict(complete=full,planned=n,observed=m,comparison=paired(sub,'agent_correct','baseline_correct') if full else None,secondary_strict=paired(sub,'agent_strict_correct','baseline_strict_correct') if full else None)
                details[key]=[]
                for task in sorted({r['task'] for r in sub}):
                    ss=[r for r in sub if r['task']==task]
                    details[key].append(dict(task=task,denominator=len(ss),observed=sum(r['observed'] for r in ss),correct=sum(r['agent_correct'] for r in ss),baseline_correct=sum(r['baseline_correct'] for r in ss),gain=sum(r['agent_correct'] and not r['baseline_correct'] for r in ss if r['observed']),loss=sum(r['baseline_correct'] and not r['agent_correct'] for r in ss if r['observed'])))
            write(HERE/f'runs/reports/{ds}_{model}_paired_questions.json',scored)
    write(HERE/'runs/reports/TOTAL_TABLE.json',table);write(HERE/'runs/reports/PAIRED_SUMMARY.json',pairs);write(HERE/'runs/reports/TASK_DETAILS.json',details)
    aggregates={}
    for scope in ['dev20','remaining80','full']:
        rr=[r for r in table if r['scope']==scope];done=all(r['complete'] for r in rr)
        aggregates[scope]=dict(complete=done,planned=sum(r['planned'] for r in rr),observed=sum(r['observed'] for r in rr),
            agent_correct=sum(r['correct'] for r in rr) if done else None,baseline_correct=sum(r['baseline_correct'] for r in rr),
            macro_delta_pp=statistics.mean(r['accuracy_pct']-r['baseline_accuracy_pct'] for r in rr) if done else None,
            per_model_net={model:sum(r['correct']-r['baseline_correct'] for r in rr if r['model']==model) for model in ['omni','videollama']} if done else None)
    write(HERE/'runs/reports/AGGREGATE.json',aggregates)
    lines=['冻结V13全数据集评测：复用开发20%和所有native baseline，仅新增剩余80%的方法预测。',
        '分别报告dev20 / remaining80 / full，未完成范围不公布准确率或配对CI。固定完整分母；缺失/错误/未解析计错并单列。',
        '剩余80%此前做过baseline审计，题目划分可能共享视频，不称完全独立未见测试集。全量含反复开发20%，不能直接当无偏泛化证明。',
        '算法与V13字节一致，未运行V14、baseline生成或消融。核验文本不等于事实；额外局部媒体/计算预算公开计入。',
        '历史native、复用dev20和新增80%来自不同时段，成本只作记账；旧V13的pilot/4强制probe及加载见setup_history.json，不重复计为本轮工作。','']
    for r in table:
        lines.append(f"{r['dataset']}/{r['model']}/{r['scope']}: {r['observed']}/{r['planned']} complete={r['complete']} accuracy={r['accuracy_pct']} native={r['baseline_accuracy_pct']:.6f}% mean={r['mean_seconds']} P95={r['p95_seconds']} errors={r['errors']} missing={r['missing']}")
    (HERE/'runs/reports/REPORT_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return complete

if __name__=='__main__':main()
