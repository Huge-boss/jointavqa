"""Offline complete-round audit. Never generates predictions or changes frozen files."""
import collections
import hashlib
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RUNS = BASE / 'evaluation_v13_full/runs'
sys.path.insert(0, str(BASE / 'v5'))
from evaluate import answer, paired, percentile


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6), (a, b)


def unique(rs):
    d = {r['qid']: r for r in rs}
    assert len(d) == len(rs)
    return d


def remote_receipt():
    """Read original sources on the evaluation host; no inference or source edits."""
    sys.path.insert(0, str(BASE/'evaluation_v13_full'))
    from settings import frozen
    m = frozen()
    def loadlines(p):
        return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    cached = {}; checked = 0
    for ds in ['joint','avspeaker']:
        data = BASE/'evaluation_v13_full/data'
        inputs = loadlines(data/f'{ds}.jsonl'); imap = unique(inputs)
        original = ROOT/('reproduction/frozen_seed42/inputs.jsonl' if ds == 'joint' else 'reproduction/avspeaker_v1/inputs.jsonl')
        assert inputs == loadlines(original)
        split = json.loads((BASE/f'v5/manifests/{ds}_split.json').read_text())
        assert sha(original.read_bytes()) == split['source_sha256']
        assert m['splits'][ds]['remaining80'] == split['remaining_qids']
        assert set(m['splits'][ds]['dev20']) == set(split['qids'])
        assert all(not set(r) & {'label','answer','correct_answer','correct_prefix','answer_label'} for r in inputs)
        for model in ['omni','videollama']:
            for n in loadlines(data/f'{ds}_{model}_native.jsonl'):
                source=n['source']; p=source['path']
                if p not in cached:
                    assert sha((ROOT/p).read_bytes()) == source['sha256'] == m['sources'][p]
                    cached[p]=loadlines(ROOT/p)
                raw=cached[p][source['line']-1]; row=imap[n['qid']]
                assert raw['qid'] == n['qid'] and raw['status'] == 'ok'
                assert raw['model_output'] == n['model_output']
                assert raw['elapsed_seconds'] == n['seconds'] if ds == 'joint' else raw['seconds'] == n['seconds']
                assert all(raw.get(k) == v for k,v in n['media_reference'].items())
                assert raw['question_prompt'] == row['question_prompt']
                option='prefix2text' if ds == 'joint' else 'options'
                assert raw[option] == row[option]
                checked += 1
    receipt=dict(status='passed',manifest_sha256=sha((RUNS/'manifest.json').read_bytes()),dependencies=len(m['dependencies']),sources=len(m['sources']),assets=len(m['assets']),native_source_records_verified=checked,original_split_and_inputs_identical=True,scope='End-of-run frozen dependency/source hashes and full native output/token/media/cost/question/options provenance; no model calls.')
    (RUNS/'completed_audit').mkdir(exist_ok=True)
    dump(RUNS/'completed_audit/REMOTE_RECEIPT.json',receipt)
    print(json.dumps(receipt))


def main():
    digest = sha((RUNS/'complete_results.tar.gz').read_bytes())
    state = json.loads((RUNS/'state.json').read_text(encoding='utf-8'))
    assert state['status'] == 'complete' and state['archive_sha256'] == digest
    assert digest == '1f28604e4b8fa77e1be76ee0d861cc24c292c5f8021302ba50041fe2eeb59983'
    out = RUNS/'completed_audit'
    out.mkdir(exist_ok=True)
    trajectories, order_notes, exceptions = {}, {}, []
    with tarfile.open(RUNS/'complete_results.tar.gz') as tar, tarfile.open(BASE/'memory_v13/runs/complete_results.tar.gz') as oldtar:
        def read(n):
            return json.load(tar.extractfile(n))
        def rows(n):
            return [json.loads(s) for s in tar.extractfile(n) if s.strip()]
        mb = tar.extractfile('runs/manifest.json').read()
        assert sha(mb) == '118067fc02501a33d53d020ec4870e087a6cd242d2a6480a1dc2837f6db15944'
        m = json.loads(mb)
        for n, h in m['code'].items():
            assert sha(tar.extractfile('source/'+n).read()) == h
            assert sha((BASE/'evaluation_v13_full'/n).read_bytes()) == h
        assert sha(tar.extractfile('source/protocol.json').read()) == m['protocol']
        for n, h in m['assets'].items():
            assert sha(tar.extractfile(n).read()) == h
        for n in ['method.py', 'ledger_grammar.py']:
            assert tar.extractfile('source/'+n).read() == oldtar.extractfile('source/'+n).read()
        for n in ['v5/evaluate.py', 'v5/joint_parser.py', 'compat/avspeaker/common.py']:
            assert sha((BASE/n).read_bytes()) == m['dependencies']['experiments/av_evidence_agent_v1/'+n]
        table = read('runs/reports/TOTAL_TABLE.json')
        details = read('runs/reports/TASK_DETAILS.json')
        comparisons = read('runs/reports/PAIRED_SUMMARY.json')
        aggregate = read('runs/reports/AGGREGATE.json')
        oldpaired = json.load(oldtar.extractfile('runs/reports/PAIRED_SUMMARY.json'))
        assert len(table) == len(details) == len(comparisons) == 12
        assert all(t['complete'] for t in table)
        totals = collections.Counter()
        for ds, expected in [('joint', 2853), ('avspeaker', 3212)]:
            inputs = rows(f'data/{ds}.jsonl'); imap = unique(inputs)
            split = {k:set(v) for k,v in m['splits'][ds].items()}
            assert len(imap) == expected and set(imap) == split['full']
            assert not split['dev20'] & split['remaining80']
            assert split['dev20'] | split['remaining80'] == split['full']
            lp = ROOT/('reproduction/frozen_seed42/labels.json' if ds == 'joint' else 'reproduction/avspeaker_v1/labels.json')
            if not lp.exists():
                lp = ROOT/'results/avspeaker_v1_20261002/evidence_snapshot/reproduction/avspeaker_v1/labels.json'
            labels = json.loads(lp.read_text(encoding='utf-8'))
            assert set(labels) == set(imap)
            for model in ['omni','videollama']:
                key = ds+'/'+model
                native = unique(rows(f'data/{ds}_{model}_native.jsonl'))
                inherited = rows(f'data/{ds}_{model}_v13_dev20.jsonl')
                fresh = [r for g in ['0','1','3'] for r in rows(f'runs/{ds}/{model}/remaining80/gpu{g}/predictions.jsonl')]
                assert set(unique(inherited)) == split['dev20']
                assert set(unique(fresh)) == split['remaining80']
                raw = unique(inherited+fresh)
                assert set(raw) == set(native) == set(imap)
                originals = unique([json.loads(s) for n in oldtar.getnames() if n.startswith(f'runs/{ds}/{model}/dev20/gpu') and n.endswith('/predictions.jsonl') for s in oldtar.extractfile(n) if s.strip()])
                for r in inherited:
                    assert {k:v for k,v in r.items() if k != 'reuse_origin'} == originals[r['qid']]
                for g in ['0','1','3']:
                    gr = rows(f'runs/{ds}/{model}/remaining80/gpu{g}/predictions.jsonl')
                    assert {r['qid'] for r in gr} == set(m['partitions'][ds][g])
                    assert all(str(r['physical_gpu']) == g for r in gr)
                scores = read(f'runs/reports/{ds}_{model}_paired_questions.json')
                assert [r['qid'] for r in scores] == [r['qid'] for r in inputs]
                smap = unique(scores)
                counts = collections.Counter(); routes = collections.Counter()
                for q, r in raw.items():
                    assert r['status'] == 'ok' and str(r['physical_gpu']) in ['0','1','3']
                    a = r['arm']; calls = a['calls']; fs = a['scout_fields']; s = smap[q]
                    y = labels[q]['correct_prefix'] if ds == 'joint' else labels[q]
                    pred, strict = answer(ds,a['model_output'],imap[q])
                    b, bs = answer(ds,native[q]['model_output'],imap[q])
                    assert (pred,strict,b,bs) == (s['agent'],s['agent_strict'],s['baseline'],s['baseline_strict'])
                    assert s['label'] == y and s['task'] == imap[q]['task'] and s['observed']
                    assert s['cluster'] == imap[q].get('video_name',imap[q].get('video_id',imap[q]['clip_path']))
                    for field,val in [('agent_correct',pred==y),('agent_strict_correct',strict==y),('baseline_correct',b==y),('baseline_strict_correct',bs==y)]:
                        assert s[field] == val
                    assert pred == strict and b == bs
                    assert s['source_partition'] == ('dev20_reused' if q in split['dev20'] else 'remaining80_new')
                    assert 1 <= len(calls) <= 3 and a['logical_calls'] == 1+len(calls) and a['executed_calls'] == len(calls)
                    assert sum(c['generated_tokens'] for c in calls) <= 240
                    assert calls[0]['generated_tokens'] <= 128
                    assert all(c['generated_tokens'] <= (96 if c['window'] is not None else 16) for c in calls[1:])
                    assert sum(c['window'] is None for c in calls) <= 2
                    assert all(c['audio_tokens'] > 0 and c['video_tokens'] > 0 for c in calls)
                    assert a['cache_start'] == 'empty_application_cache_per_question'
                    assert a['verifier_context_policy'] == 'raw_global_and_fresh_local_only_no_candidate_conditioned_ledger'
                    assert a['new_work_seconds'] >= 0
                    close(a['baseline_cost_seconds'],native[q]['seconds'])
                    close(a['seconds'],a['new_work_seconds']+a['baseline_cost_seconds'])
                    if a['decision'] == 'agreement':
                        assert len(calls) == 1 and not a['review']
                    if a['decision'] == 'verified_revision':
                        assert a['confirmation_answer'] == a['proposal_answer'] == pred and not calls[-1]['reached_token_limit']
                    else:
                        assert a['model_output'] == native[q]['model_output']
                    if a['review']:
                        assert len(calls) == 3 and not fs['field_token_caps']['EVIDENCE']
                        assert fs['ANCHOR'] != 'NONE' and fs['RELATION'] != 'GLOBAL'
                        assert 0 < a['review_window'][1]-a['review_window'][0] <= 12.000001
                        assert a['verification_memory_ids'] == ['verification_local']
                    else:
                        assert a['verification_memory_ids'] == []
                    counts['proposal_correct'] += a['proposal_answer'] == y
                    counts['saved_vs_proposal'] += a['proposal_answer'] != y and pred == y
                    counts['hurt_vs_proposal'] += a['proposal_answer'] == y and pred != y
                    counts['agreement_wrong'] += a['decision'] == 'agreement' and pred != y
                    counts['local_gain'] += a['review'] and pred == y and b != y
                    counts['local_loss'] += a['review'] and pred != y and b == y
                    if a['proposal_answer'] != a['native_answer']:
                        routes[a['window_source']] += 1
                    if pred is None or any(c['reached_token_limit'] for c in calls):
                        exceptions.append(dict(dataset=ds,model=model,qid=q,unparsed=pred is None,decision=a['decision'],final_output=a['model_output'],capped_call_indices=[i for i,c in enumerate(calls) if c['reached_token_limit']]))
                trajectories[key] = dict(counts=dict(counts),disagreement_routes=dict(routes))
                starts = [read(n) for n in tar.getnames() if n.startswith(f'runs/{ds}/{model}/remaining80/gpu') and '/worker_start_' in n and n.endswith('.json')]
                assert len(starts) == 3
                for scope in ['dev20','remaining80','full']:
                    sk = key+'/'+scope; ss = [s for s in scores if s['qid'] in split[scope]]
                    arms = [raw[s['qid']]['arm'] for s in ss]; calls = [c for a in arms for c in a['calls']]
                    t = next(t for t in table if (t['dataset'],t['model'],t['scope']) == (ds,model,scope))
                    assert t['planned'] == t['observed'] == len(ss)
                    assert t['missing'] == t['errors'] == 0
                    for k, value in dict(correct=sum(s['agent_correct'] for s in ss),baseline_correct=sum(s['baseline_correct'] for s in ss),strict_correct=sum(s['agent_strict_correct'] for s in ss),baseline_strict_correct=sum(s['baseline_strict_correct'] for s in ss),unparsed=sum(s['agent'] is None for s in ss),strict_unparsed=sum(s['agent_strict'] is None for s in ss),executed_calls=len(calls),reviews=sum(a['review'] for a in arms),final_call_truncations=sum(a['calls'][-1]['reached_token_limit'] for a in arms),intermediate_truncations=sum(c['reached_token_limit'] for a in arms for c in a['calls'][:-1]),generated_tokens=sum(c['generated_tokens'] for c in calls)).items():
                        assert t[k] == value,(sk,k)
                    assert t['labels_sha256'] == sha(lp.read_bytes())
                    assert t['decisions'] == dict(collections.Counter(a['decision'] for a in arms))
                    assert t['field_caps'] == dict(collections.Counter(k for a in arms for k,v in a['scout_fields']['field_token_caps'].items() if v))
                    times = [a['seconds'] for a in arms]; new = [a['new_work_seconds'] for a in arms]
                    for k, value in dict(accuracy_pct=100*t['correct']/len(ss),baseline_accuracy_pct=100*t['baseline_correct']/len(ss),mean_seconds=statistics.mean(times),p95_seconds=percentile(times,.95),incremental_mean_seconds=statistics.mean(new),incremental_p95_seconds=percentile(new,.95),incremental_work_seconds=sum(new),historical_native_work_seconds=sum(a['baseline_cost_seconds'] for a in arms),logical_total_work_seconds=sum(times),mean_logical_calls=statistics.mean(a['logical_calls'] for a in arms),preparation_seconds=sum(c['cache_build_seconds'] for c in calls),crop_seconds=sum(c['crop_seconds'] for c in calls),new_eval_model_load_seconds=sum(s['load_seconds'] for s in starts) if scope != 'dev20' else 0).items():
                        close(t[k],value)
                    assert paired(ss,'agent_correct','baseline_correct') == comparisons[sk]['comparison']
                    assert comparisons[sk]['comparison'] == comparisons[sk]['secondary_strict']
                    assert len(details[sk]) == (15 if ds == 'joint' else 12)
                    for task in details[sk]:
                        tt = [s for s in ss if s['task'] == task['task']]
                        assert task == dict(task=task['task'],denominator=len(tt),observed=len(tt),correct=sum(s['agent_correct'] for s in tt),baseline_correct=sum(s['baseline_correct'] for s in tt),gain=sum(s['agent_correct'] and not s['baseline_correct'] for s in tt),loss=sum(s['baseline_correct'] and not s['agent_correct'] for s in tt))
                    assert sum(t['denominator'] for t in details[sk]) == len(ss)
                op = json.load(oldtar.extractfile(f'runs/reports/{ds}_{model}_paired_questions.json'))
                original_order = [smap[s['qid']] for s in op]
                for s in op:
                    assert all(s[k] == smap[s['qid']][k] for k in ['cluster','label','agent','baseline','agent_correct','baseline_correct'])
                oldci = paired(original_order,'agent_correct','baseline_correct')
                assert oldci == oldpaired[key]['comparison']
                order_notes[key] = dict(original_dev_order=oldci,current_full_filtered_order=comparisons[key+'/dev20']['comparison'],explanation='Same qids/clusters/scores/seed; finite bootstrap differs with cluster insertion order. Frozen evaluator unchanged; neither interval selected for advantage.')
                totals['full'] += len(raw); totals['new'] += len(fresh); totals['reused'] += len(inherited)
        assert dict(totals) == dict(full=12130,new=9704,reused=2426)
        cost = {}
        for scope in ['dev20','remaining80','full']:
            tt = [r for r in table if r['scope'] == scope]
            ag = aggregate[scope]
            assert ag['complete'] and ag['planned'] == ag['observed'] == sum(r['planned'] for r in tt)
            assert ag['agent_correct'] == sum(r['correct'] for r in tt) and ag['baseline_correct'] == sum(r['baseline_correct'] for r in tt)
            close(ag['macro_delta_pp'],statistics.mean(r['accuracy_pct']-r['baseline_accuracy_pct'] for r in tt))
            assert ag['per_model_net'] == {model:sum(r['correct']-r['baseline_correct'] for r in tt if r['model']==model) for model in ['omni','videollama']}
            cost[scope] = {k:sum(r[k] for r in tt) for k in ['incremental_work_seconds','historical_native_work_seconds','logical_total_work_seconds','executed_calls','reviews','final_call_truncations','intermediate_truncations','unparsed','new_eval_model_load_seconds']}
            cost[scope]['evidence_field_caps'] = sum(r['field_caps'].get('EVIDENCE',0) for r in tt)
            cost[scope]['wrong_to_right'] = sum(comparisons[r['dataset']+'/'+r['model']+'/'+scope]['comparison']['wrong_to_right'] for r in tt)
            cost[scope]['right_to_wrong'] = sum(comparisons[r['dataset']+'/'+r['model']+'/'+scope]['comparison']['right_to_wrong'] for r in tt)
        stages = rows('runs/stages.jsonl')
        assert len(stages) == 4 and len({(s['dataset'],s['model']) for s in stages}) == 4
        for s in stages:
            close(s['wall_seconds'],s['finished_at']-s['started_at'])
        setup = read('data/setup_history.json')
        audit = dict(status='passed',archive_sha256=digest,manifest_sha256=sha(mb),coverage=dict(totals),validation='All raw outputs rescored with frozen primary/strict parser against label hashes; unique qids, partition/GPU allocation, old raw reuse, source/protocol/assets, 12 report rows, 54 per-scope task groups, bootstrap, routing/call/token/cache/cost bounds checked. Remote dependency/source checks are in REMOTE_RECEIPT.json.',aggregate=aggregate,cost=cost,formal_wall_seconds=sum(s['wall_seconds'] for s in stages),first_stage_to_last_end_seconds=max(s['finished_at'] for s in stages)-min(s['started_at'] for s in stages),stages=stages,historical_setup=setup,table=table,comparisons=comparisons,tasks=details,trajectories=trajectories,exceptions=exceptions,ci_order_notes=order_notes)
    dump(out/'AUDIT.json',audit)
    dump(out/'CI_ORDER_NOTES.json',order_notes)
    dump(out/'EXCEPTIONS.json',exceptions)
    lines = ['冻结V13全数据集评测：完整审计与结论（2026-10-03）',
        '12130/12130输出完整，其中9704条本轮补跑、2426条原开发结果逐字复用；baseline全部复用，无新增baseline或pilot生成。四阶段与collector均已退出。',
        '独立离线复核通过：唯一qid、分区互斥/并集、原始主分与strict逐题重评分、任务分母、代码/协议/资产hash、调用/路由/缓存/成本预算。冻结方法与memory_v13逐字节一致。',
        '完整归档SHA256：'+digest, '']
    for scope,title in [('remaining80','新增80%'),('dev20','复用开发20%'),('full','全量')]:
        ag=aggregate[scope];co=cost[scope]
        lines.append(f"{title}：{ag['observed']}题，native {ag['baseline_correct']} → V13 {ag['agent_correct']}，净{ag['agent_correct']-ag['baseline_correct']:+d}；改对/改错{co['wrong_to_right']}/{co['right_to_wrong']}；四组等权{ag['macro_delta_pp']:+.6f}pp；Omni净{ag['per_model_net']['omni']:+d}、Video净{ag['per_model_net']['videollama']:+d}。")
        for r in table:
            if r['scope'] != scope:continue
            g=comparisons[r['dataset']+'/'+r['model']+'/'+scope]['comparison']
            lines.append(f"  {r['dataset']}/{r['model']}: {r['baseline_correct']}/{r['planned']} ({r['baseline_accuracy_pct']:.6f}%) → {r['correct']}/{r['planned']} ({r['accuracy_pct']:.6f}%)；改对/改错{g['wrong_to_right']}/{g['right_to_wrong']}；差值{g['delta_pp']:+.6f}pp，视频聚类95%CI {g['cluster_bootstrap_95pct_pp']}。")
            lines.append(f"  新增工作均值{r['incremental_mean_seconds']:.6f}s；含历史native均值/P95 {r['mean_seconds']:.6f}/{r['p95_seconds']:.6f}s；逻辑调用均值{r['mean_logical_calls']:.6f}；局部回看{r['reviews']}。")
        lines.append('')
    co=cost['remaining80'];full=cost['full']
    lines += ['结论：当前方法带来小幅净收益，但新增80%四组区间全部跨零，两个Video组各仅净多1题；没有得到稳定跨数据集、跨模型提升的证据。开发20%净34，新80%净19，不能用含开发题的全量净53掩盖新增范围的有限收益。全量最小增益为AV Video +0.093400pp；新增范围最小为AV Video +0.038911pp。',
        '全量Joint Omni区间高于零，其余三组跨零；全量包含多轮开发选择，新增范围也与开发共享99个Joint/335个AV视频cluster、且此前已审baseline，不是完全独立未见保留集。区间未校正多轮方法选择，不当无偏泛化证明。',
        f"本轮新增逐题工作{co['incremental_work_seconds']:.6f}s；对应历史native {co['historical_native_work_seconds']:.6f}s。本轮正式阶段墙钟合计{audit['formal_wall_seconds']:.6f}s（{audit['formal_wall_seconds']/60:.6f}分钟）；从首阶段开始到末阶段结束{audit['first_stage_to_last_end_seconds']:.6f}s。加载{co['new_eval_model_load_seconds']:.6f}s已包含阶段墙钟，不重复相加。",
        f"合并历史开发后，全量方法新增工作{full['incremental_work_seconds']:.6f}s，历史native {full['historical_native_work_seconds']:.6f}s，合计{full['logical_total_work_seconds']:.6f}s，是历史native工作之和的{full['logical_total_work_seconds']/full['historical_native_work_seconds']:.4f}倍；这是跨时段成本记账，不是同轮速度比。工作包含CPU/IO、读取、裁剪、准备与生成，不是纯GPU时间。",
        '历史设置成本单列（并非本轮再执行）：'+json.dumps(setup,ensure_ascii=False),
        f"缺失/运行错误均0；未解析{full['unparsed']}，原文保留并计错。新调用末次触限{full['final_call_truncations']}、中间触限{full['intermediate_truncations']}，EVIDENCE字段触48token上限{full['evidence_field_caps']}；不能声称全无截断。详情EXCEPTIONS.json。",
        '唯一未解析来自Joint Omni保留native含解释的原文；冻结评分没有提取其解释中的C，按原规则计错。另一个问题末次生成触限，不与该未解析混为一谈。',
        '开发集CI与旧汇总的细小差异只来自全量过滤后的行/cluster插入顺序。四组恢复旧开发行序均精确复现历史CI；qid、标签、答案、seed均不变，冻结汇总未改，两种结果和原因均保留。',
        f"全量局部回看{full['reviews']}次；其路径改对/改错{sum(t['counts']['local_gain'] for t in trajectories.values())}/{sum(t['counts']['local_loss'] for t in trajectories.values())}。这是路径描述，不是消融因果；未做独立音频听证，模型解释不视为事实。候选格式/提示、核验和局部额外媒体同时存在，不能把净收益纯归事件记忆。",
        'V13全局保持native采样：Joint最多32帧、Omni前300秒连续音频、Video前300秒8×2秒稀疏音频；AV Omni原生1fps全音频，Video8帧/8×2秒稀疏音频。局部可能增加native未输入音频，非等媒体预算或纯记忆；请求12秒不保证编码容器严格12秒，可能有一帧量化差。候选A-D语法约束及提示是相对旧方法的改变。每题最多3次新增调用、2次新增全局前向，128+96+16≤240新token；无V14可见标记。',
        '基于这轮收益与成本，不建议直接宣称方法稳定有效或继续盲目全量调参。本轮已完成，保留全部结果与失败，不自动启动新版本、消融、baseline或重复本轮。原开发排行榜保留，全量记录单独保存。', '', '全任务明细（分母完整保留）：']
    for key,ts in details.items():
        lines.append(key+': '+'; '.join(f"{t['task']} {t['correct']}/{t['denominator']} native{t['baseline_correct']} 改对/改错{t['gain']}/{t['loss']}" for t in ts))
    (out/'ANALYSIS_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    record = dict(status='complete_audited',archive_sha256=digest,manifest_sha256=sha(mb),aggregate=aggregate,cost=cost,table=table,comparisons=comparisons,formal_wall_seconds=audit['formal_wall_seconds'],historical_setup=setup,conclusion='Small descriptive net gains; all four remaining80 cluster intervals cross zero. No tuning/ablation authorized. Preserve development leaderboard separately.')
    dump(BASE/'V13_FULL_EVALUATION.json',record)
    (BASE/'V13_FULL_EVALUATION_ZH.txt').write_text('\n'.join(lines[:lines.index('全任务明细（分母完整保留）：')])+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',coverage=dict(totals),aggregate=aggregate,cost=cost,formal_wall_seconds=audit['formal_wall_seconds']),ensure_ascii=True))


if __name__ == '__main__':
    remote_receipt() if '--remote-receipt' in sys.argv else main()
