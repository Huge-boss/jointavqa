"""Offline completed-round audit. Labels never enter inference or routing."""
import collections
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / 'memory_v11/runs'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    digest = sha((RUNS / 'complete_results.tar.gz').read_bytes())
    assert digest == json.loads((RUNS / 'state.json').read_text())['archive_sha256']
    out = RUNS / 'completed_audit'
    out.mkdir(exist_ok=True)
    previous = json.loads((BASE / 'DEVELOPMENT_LEADERBOARD.json').read_text(encoding='utf-8'))
    oldtable = {(r['dataset'], r['model']): r for r in previous['v10_complete_table']}
    trajectories, comparisons, packet, groups = {}, {}, [], []
    with tarfile.open(RUNS / 'complete_results.tar.gz') as tar, tarfile.open(BASE / 'memory_v10/runs/complete_results.tar.gz') as oldtar:
        def read(name):
            return json.load(tar.extractfile(name))
        manifest = read('runs/manifest.json')
        for name, expected in manifest['code'].items():
            assert sha(tar.extractfile('source/' + name).read()) == expected
        assert sha(tar.extractfile('source/protocol.json').read()) == manifest['protocol']
        table = read('runs/reports/TOTAL_TABLE.json')
        details = read('runs/reports/TASK_DETAILS.json')
        paired = read('runs/reports/PAIRED_SUMMARY.json')
        assert len(table) == 4 and all(r['complete'] for r in table)
        for row in table:
            ds, model = row['dataset'], row['model']
            key = ds + '/' + model
            pairs = read(f'runs/reports/{ds}_{model}_paired_questions.json')
            scores = {r['qid']: r for r in pairs}
            oldscores = {r['qid']: r for r in json.load(oldtar.extractfile(f'runs/reports/{ds}_{model}_paired_questions.json'))}
            inputs = {r['qid']: r for r in map(json.loads, tar.extractfile(f'data/{ds}.jsonl'))}
            assert tar.extractfile(f'data/{ds}.jsonl').read() == oldtar.extractfile(f'data/{ds}.jsonl').read()
            raw = [json.loads(line) for name in tar.getnames()
                   if name.startswith(f'runs/{ds}/{model}/dev20/gpu') and name.endswith('/predictions.jsonl')
                   for line in tar.extractfile(name) if line.strip()]
            assert len(raw) == len(scores) == len(inputs) == row['planned']
            assert len({r['qid'] for r in raw}) == len(raw)
            assert {r['qid'] for r in raw} == set(scores) == set(inputs) == set(oldscores)
            assert all(r['status'] == 'ok' for r in raw)
            assert sum(s['agent_correct'] for s in pairs) == row['correct']
            assert sum(s['baseline_correct'] for s in pairs) == row['baseline_correct']
            assert len(details[key]) == (15 if ds == 'joint' else 12)
            for task in details[key]:
                subset = [s for s in pairs if s['task'] == task['task']]
                assert len(subset) == task['denominator'] == task['observed']
                assert sum(s['agent_correct'] for s in subset) == task['correct']
                assert sum(s['baseline_correct'] for s in subset) == task['baseline_matched_correct']
            assert sum(t['denominator'] for t in details[key]) == row['planned']
            count, categories = collections.Counter(), collections.defaultdict(list)
            for r in raw:
                a, s = r['arm'], scores[r['qid']]
                label, old, final = s['label'], s['baseline'], s['agent']
                proposal = a['proposal_answer']
                assert old == oldscores[r['qid']]['baseline']
                count['proposal_correct'] += proposal == label
                count['final_correct'] += final == label
                count['baseline_correct'] += old == label
                count['disagreements'] += proposal != old
                count['accepted'] += a['decision'] == 'verified_revision'
                count['verifier_saved_vs_proposal'] += proposal != label and final == label
                count['verifier_hurt_vs_proposal'] += proposal == label and final != label
                if a['decision'] == 'agreement':
                    count['agreement_wrong'] += final != label
                category = ('harmful_revision' if old == label and final != label else
                            'beneficial_revision' if old != label and final == label else
                            'missed_correct_proposal' if proposal == label and final != label else
                            'agreement_wrong' if a['decision'] == 'agreement' and final != label else None)
                if category:
                    categories[category].append(dict(dataset=ds, model=model, category=category,
                        input=inputs[r['qid']], offline_score=s, decision=a['decision'],
                        proposal=proposal, confirmation=a['confirmation_answer'],
                        review_window=a['review_window'], memories=a['memories']))
            # Fixed diagnostic strata and qid SHA ordering; never used for selecting prediction answers.
            for category, cases in categories.items():
                packet.extend(sorted(cases, key=lambda c: sha(c['input']['qid'].encode()))[:3])
            trajectories[key] = dict(count)
            c = paired[key]['comparison']
            groups.append(dict(dataset=ds, model=model, **c))
            oldrow = oldtable[ds, model]
            comparisons[key] = dict(correct_v10=oldrow['correct'], correct_v11=row['correct'],
                net_vs_v10=row['correct'] - oldrow['correct'],
                gain_vs_v10=sum(s['agent_correct'] and not oldscores[s['qid']]['agent_correct'] for s in pairs),
                loss_vs_v10=sum(not s['agent_correct'] and oldscores[s['qid']]['agent_correct'] for s in pairs),
                new_work_reduction_pct=100*(1-row['incremental_work_seconds']/oldrow['incremental_work_seconds']))
        stages = list(map(json.loads, tar.extractfile('runs/stages.jsonl')))
    aggregate = dict(total_questions=sum(r['planned'] for r in table),
        baseline_correct=sum(r['baseline_correct'] for r in table), method_correct=sum(r['correct'] for r in table),
        wrong_to_right=sum(g['wrong_to_right'] for g in groups), right_to_wrong=sum(g['right_to_wrong'] for g in groups),
        macro_delta_pp=statistics.mean(g['delta_pp'] for g in groups),
        new_work_seconds=sum(r['incremental_work_seconds'] for r in table),
        historical_native_work_seconds=sum(r['historical_native_work_seconds'] for r in table),
        model_load_seconds=sum(r['model_load_seconds'] for r in table),
        formal_wall_seconds=sum(s['wall_seconds'] for s in stages if s['phase'] == 'dev20'),
        pilot_wall_seconds=sum(s['wall_seconds'] for s in stages if s['phase'] == 'pilot'))
    audit = dict(archive_sha256=digest, validation='2426 unique planned outputs; frozen code/protocol and unchanged v10 inputs; all task counts/correctness checked',
                 aggregate=aggregate, trajectories=trajectories, comparisons_v10=comparisons, groups=groups, table=table, tasks=details)
    (out/'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'DIAGNOSTIC_CASES.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['V11完整开发轮审计：同一20%，两模型×两数据集',
        '2426/2426完成，错误/缺失/未解析0；输入集合、唯一qid、冻结代码/协议、任务数和逐项正确计数均核对。归档已本地SHA校验。',
        '四组原评分与strict分数一致。末次调用触限1次，中间18次；工具字段另有截短，不能声称全部文本无截短。', '']
    for r,g in zip(table,groups):
        key = r['dataset']+'/'+r['model']
        lines += [f"{key}: native {r['baseline_correct']}/{r['planned']} ({r['baseline_accuracy_pct']:.4f}%) → V11 {r['correct']} ({r['accuracy_pct']:.4f}%)，改对/改错 {g['wrong_to_right']}/{g['right_to_wrong']}，增益 {g['delta_pp']:+.4f}pp，聚类95%CI {g['cluster_bootstrap_95pct_pp']}",
            f"新增均值{r['incremental_mean_seconds']:.4f}s，含历史native均值/P95 {r['mean_seconds']:.4f}/{r['p95_seconds']:.4f}s，逻辑调用{r['mean_logical_calls']:.4f}，回看{r['reviews']}题。相对V10净题数{comparisons[key]['net_vs_v10']:+d}。",
            '分项：'+ '; '.join(f"{t['task']} {t['correct']}/{t['denominator']} 净{t['gain']-t['loss']:+d}" for t in details[key])]
    saved = sum(c['verifier_saved_vs_proposal'] for c in trajectories.values())
    hurt = sum(c['verifier_hurt_vs_proposal'] for c in trajectories.values())
    wrong = sum(c.get('agreement_wrong',0) for c in trajectories.values())
    new_reduction = 100*(1-aggregate['new_work_seconds']/previous['v10_aggregate']['new_work_seconds'])
    lines += ['', f"总计native1058→V11{aggregate['method_correct']}，改对{aggregate['wrong_to_right']}/改错{aggregate['right_to_wrong']}，净+29；四组等权增益{aggregate['macro_delta_pp']:.5f}pp。Omni净+23，Video净+6。V10为1062，V11多25题，但Joint Video比V10少2题。",
        '四组CI都跨零，20%已经反复用于开发，当前小幅正增益不能当作独立泛化证据，也不是大幅提升。',
        f"成本：正式阶段墙钟{aggregate['formal_wall_seconds']/60:.2f}分钟，pilot另{aggregate['pilot_wall_seconds']/60:.2f}分钟；新增逐题工作合计{aggregate['new_work_seconds']/3600:.4f}小时，历史native另{aggregate['historical_native_work_seconds']/3600:.4f}小时，模型加载和人工预检另计。相比V10新增工作少{new_reduction:.2f}%，仅为跨轮描述性比较，仍明显高于直接baseline成本；工作秒含CPU准备，不是纯GPU小时。",
        f"路径：核验相对提案挽回{saved}题、损失{hurt}题。native与提案一致但最终错误的有{wrong}题，当前路由不会触发这些题的局部核验。此为离线诊断，不能把正确标签作为部署路由。",
        '两次完整记忆候选尚未证明稳定收益。V11减少改错并降低成本的方向值得保留，但不能直接据此叠加更多调用。当前不启动无证据的V12整轮，也不运行80%或消融。',
        '下一步证据计划：DIAGNOSTIC_CASES.json按固定四类（改错、改对、正确提案被拒、一致但错）和qid SHA取每组每类最多3题，共最多48题。人工逐段核对原始音画、实际全局/局部覆盖、人物绑定和记忆是否有事实支撑；结果未知须标未知。样本用于定性机制，不估计全体发生率，不据单题设计答案规则。',
        '优先区分：关键证据未采到、已有证据感知错误、笔记与原始片段冲突、核验拒绝正确提案、答案一致而证据不足。先完成可复查证据记录再决定下一完整方案；模型的自生成解释不能作为事实裁判。',
        '保留统一方法完整排行榜，不按组/题拼最佳；相同模型上下文隔离不等于独立模型。局部回看可能增加原生未输入音频，不能把收益全部归因记忆。']
    (out/'ANALYSIS_ZH.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    previous.update(v11_complete_table=table, v11_aggregate=aggregate,
                    next_candidate='Not launched; grounded media diagnostic evidence required before another full candidate')
    (BASE/'DEVELOPMENT_LEADERBOARD.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(aggregate)); print(json.dumps(trajectories)); print('diagnostic_cases',len(packet))


if __name__ == '__main__':
    main()
