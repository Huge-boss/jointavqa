"""Offline completed-round audit; never imported by inference workers."""
import collections
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / 'memory_v10/runs'


def main():
    archive = RUNS / 'complete_results.tar.gz'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == json.loads((RUNS / 'state.json').read_text(encoding='utf-8'))['archive_sha256']
    out = RUNS / 'completed_audit'; out.mkdir(exist_ok=True)
    trajectories = {}; all_pairs = []; groups = []
    with tarfile.open(archive) as tar:
        table = json.load(tar.extractfile('runs/reports/TOTAL_TABLE.json'))
        assert len(table) == 4 and all(r['complete'] for r in table)
        details = json.load(tar.extractfile('runs/reports/TASK_DETAILS.json'))
        pairs_summary = json.load(tar.extractfile('runs/reports/PAIRED_SUMMARY.json'))
        for row in table:
            ds, model = row['dataset'], row['model']; key = ds + '/' + model
            pairs = json.load(tar.extractfile(f'runs/reports/{ds}_{model}_paired_questions.json'))
            scores = {r['qid']: r for r in pairs}
            inputs = [json.loads(l) for l in tar.extractfile(f'data/{ds}.jsonl') if l.strip()]
            raw = [json.loads(l) for n in tar.getnames()
                   if n.startswith(f'runs/{ds}/{model}/dev20/gpu') and n.endswith('/predictions.jsonl')
                   for l in tar.extractfile(n) if l.strip()]
            assert len(raw) == row['planned'] == len(scores)
            assert len({r['qid'] for r in raw}) == len(raw)
            assert {r['qid'] for r in raw} == {r['qid'] for r in inputs} == set(scores)
            assert all(r['status'] == 'ok' for r in raw)
            assert sum(t['denominator'] for t in details[key]) == row['planned']
            count = collections.Counter()
            for r in raw:
                a, s = r['arm'], scores[r['qid']]; label = s['label']
                proposal, final, old = a['proposal_answer'], s['agent'], s['baseline']
                count['proposal_correct'] += proposal == label
                count['final_correct'] += final == label
                count['baseline_correct'] += old == label
                if proposal != old:
                    count['disagreements'] += 1
                    count['accepted'] += a['decision'] == 'verified_revision'
                count['verifier_saved_vs_proposal'] += proposal != label and final == label
                count['verifier_hurt_vs_proposal'] += proposal == label and final != label
            trajectories[key] = dict(count)
            groups.append(dict(dataset=ds, model=model, **pairs_summary[key]['comparison']))
            all_pairs.extend(pairs)
        stages = [json.loads(l) for l in tar.extractfile('runs/stages.jsonl') if l.strip()]
    aggregate = dict(
        total_questions=len(all_pairs), baseline_correct=sum(r['baseline_correct'] for r in all_pairs),
        method_correct=sum(r['agent_correct'] for r in all_pairs),
        wrong_to_right=sum(r['agent_correct'] and not r['baseline_correct'] for r in all_pairs),
        right_to_wrong=sum(r['baseline_correct'] and not r['agent_correct'] for r in all_pairs),
        macro_delta_pp=statistics.mean(g['delta_pp'] for g in groups),
        new_work_seconds=sum(r['incremental_work_seconds'] for r in table),
        native_historical_work_seconds=sum(r['historical_native_work_seconds'] for r in table),
        model_load_seconds=sum(r['model_load_seconds'] for r in table),
        formal_wall_seconds=sum(r['wall_seconds'] for r in stages if r['phase'] == 'dev20'),
        pilot_wall_seconds=sum(r['wall_seconds'] for r in stages if r['phase'] == 'pilot'))
    audit = dict(archive_sha256=digest, validation='All 2426 unique planned outputs, no missing/runtime errors; task sums checked',
                 aggregate=aggregate, trajectories=trajectories, groups=groups, table=table, tasks=details)
    (out/'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['V10完整方案：两模型×两数据集，同一20%开发集',
             '全2426次正式预测已完成；唯一qid、输入集合、任务分母、归档SHA已核验。错误、缺失、未解析均0。',
             '该20%多次用于开发，不能作为独立测试集。剩余80%未运行。', '']
    for r,g in zip(table,groups):
        lines.append(f"{r['dataset']} / {r['model']}: native {r['baseline_correct']}/{r['planned']}={r['baseline_accuracy_pct']:.4f}%; V10 {r['correct']}/{r['planned']}={r['accuracy_pct']:.4f}%; delta {g['delta_pp']:+.4f}pp; 改对/改错 {g['wrong_to_right']}/{g['right_to_wrong']}; 聚类95%CI {g['cluster_bootstrap_95pct_pp']}")
        lines.append(f"新增均值 {r['incremental_mean_seconds']:.4f}s/题；含历史native均值/P95 {r['mean_seconds']:.4f}/{r['p95_seconds']:.4f}s；逻辑调用均值 {r['mean_logical_calls']:.4f}；末次/中间触限 {r['final_call_truncations']}/{r['intermediate_truncations']}。")
    lines += ['', '总体判断：native1058/2426，V10为1062/2426，改对229、改错225，净+4。四组等权平均增益+0.17513pp；Omni净+8，Video净-4。全部单组CI跨零，不能认定稳定提升，更不是大幅提升。',
              f"正式阶段墙钟合计{aggregate['formal_wall_seconds']/60:.2f}分钟，pilot另{aggregate['pilot_wall_seconds']/60:.2f}分钟。新增逐题工作量总和{aggregate['new_work_seconds']/3600:.3f}小时，历史native{aggregate['native_historical_work_seconds']/3600:.3f}小时；不能将三卡并行墙钟与单题工作量混淆。历史native非同轮速度基准。",
              '路径核查：提案总正确1048，核验后1062，净挽回14；781次分歧接受667次。核验本身有贡献，但两步共享同一生成笔记和候选提示，不能将其一致当作独立证据。示例中存在全局与局部人物描述冲突；未逐帧人工核定，不断言哪条描述是真实。',
              '成本瓶颈：每题全局观察+全局提案，大量题再局部回看/复核。225次改错基本抵消229次改对。格式全通过并不意味着事实准确。',
              '下一完整候选V11：把提案与核验记忆分隔，先提案，只有分歧才局部取证；局部观察不接收旧笔记或候选，核验只读全局原始输入与新局部观察。窗口仍由同模型的FOCUS/实际覆盖决定，不声称统计独立。完整四组统一检验，不按类别挑用方法。',
              'V10已有方法收益很小而成本增加。V11目标是减少共享笔记干扰与不必要回看，准确率和速度均待验证，不预先承诺创新。',
              '所有历史版本、技术失败和原始输出保留；本报告不改变任何主评分或推理文件。']
    (out/'ANALYSIS_ZH.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    old = json.loads((BASE/'stage_roundoff_v2/runs/reports/TOTAL_TABLE.json').read_text(encoding='utf-8'))
    leaderboard = dict(note='All complete development methods retained; no per-question oracle. Historical v5 costs preserved in source report.',
                       native_and_v5_complete_table=old, v10_complete_table=table, v10_aggregate=aggregate,
                       next_candidate='split_evidence_v11, not yet scored')
    (BASE/'DEVELOPMENT_LEADERBOARD.json').write_text(json.dumps(leaderboard,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(aggregate,ensure_ascii=False));print(json.dumps(trajectories,ensure_ascii=False))


if __name__ == '__main__':
    main()
