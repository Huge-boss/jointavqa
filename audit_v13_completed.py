"""Offline audit only. Scored labels never enter model execution or routing."""
import collections
import contextlib
import hashlib
import json
import statistics
import sys
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / 'memory_v13/runs'
sys.path.insert(0, str(BASE / 'v5'))
from evaluate import answer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    digest = sha((RUNS/'complete_results.tar.gz').read_bytes())
    state = json.loads((RUNS/'state.json').read_text())
    assert state['status'] == 'complete' and digest == state['archive_sha256']
    previous = json.loads((BASE/'DEVELOPMENT_LEADERBOARD.json').read_text(encoding='utf-8'))
    out = RUNS/'completed_audit'
    out.mkdir(exist_ok=True)
    trajectories, comparisons, groups, all_pairs, replay_cases = {}, {}, [], [], []
    with contextlib.ExitStack() as stack:
        archives = {v: stack.enter_context(tarfile.open(BASE/f'memory_{v}/runs/complete_results.tar.gz')) for v in ['v10','v11','v13']}
        tar = archives['v13']
        def read(name):
            return json.load(tar.extractfile(name))
        def rows(name):
            return [json.loads(line) for line in tar.extractfile(name) if line.strip()]
        manifest_bytes = tar.extractfile('runs/manifest.json').read()
        assert sha(manifest_bytes) == 'b4b99cea989ad628aff075ce622a9a58ef319e6b37c1c8f0f7a997a76a5ed787'
        manifest = json.loads(manifest_bytes)
        for name, expected in manifest['code'].items():
            assert sha(tar.extractfile('source/'+name).read()) == expected
            assert sha((BASE/'memory_v13'/name).read_bytes()) == expected
        assert sha(tar.extractfile('source/protocol.json').read()) == manifest['protocol']
        for name, expected in manifest['assets'].items():
            assert sha(tar.extractfile(name).read()) == expected
        # Ensure the independent scoring implementation is the frozen version.
        for name in ['evaluate.py','joint_parser.py']:
            assert sha((BASE/'v5'/name).read_bytes()) == manifest['dependencies']['experiments/av_evidence_agent_v1/v5/'+name]
        table = read('runs/reports/TOTAL_TABLE.json')
        details = read('runs/reports/TASK_DETAILS.json')
        paired = read('runs/reports/PAIRED_SUMMARY.json')
        assert len(table) == 4 and all(r['complete'] for r in table)
        for row in table:
            ds, model = row['dataset'], row['model']
            key = ds+'/'+model
            pairs = read(f'runs/reports/{ds}_{model}_paired_questions.json')
            scores = {r['qid']: r for r in pairs}
            inputs = {r['qid']: r for r in rows(f'data/{ds}.jsonl')}
            native = {r['qid']: r for r in rows(f'data/{ds}_{model}_native.jsonl')}
            raw = [r for name in tar.getnames() if name.startswith(f'runs/{ds}/{model}/dev20/gpu') and name.endswith('/predictions.jsonl') for r in rows(name)]
            assert len(raw) == len(scores) == len(inputs) == row['planned']
            assert len({r['qid'] for r in raw}) == len(raw)
            assert {r['qid'] for r in raw} == set(scores) == set(inputs) == set(native)
            assert all(r['status'] == 'ok' and str(r['physical_gpu']) in ['0','1','3'] for r in raw)
            assert sum(s['agent_correct'] for s in pairs) == row['correct']
            assert sum(s['baseline_correct'] for s in pairs) == row['baseline_correct']
            assert len(details[key]) == (15 if ds == 'joint' else 12)
            for task in details[key]:
                subset = [s for s in pairs if s['task'] == task['task']]
                assert len(subset) == task['denominator'] == task['observed']
                assert sum(s['agent_correct'] for s in subset) == task['correct']
                assert sum(s['baseline_correct'] for s in subset) == task['baseline_matched_correct']
            assert sum(t['denominator'] for t in details[key]) == row['planned']
            count, routes, anchors, relations = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
            for r in raw:
                q = r['qid']; a = r['arm']; s = scores[q]; fs = a['scout_fields']; calls = a['calls']
                final, strict = answer(ds, a['model_output'], inputs[q])
                old, oldstrict = answer(ds, native[q]['model_output'], inputs[q])
                assert (final, strict, old, oldstrict) == (s['agent'], s['agent_strict'], s['baseline'], s['baseline_strict'])
                label = s['label']; proposal = a['proposal_answer']
                assert s['agent_correct'] == (final == label) and s['baseline_correct'] == (old == label)
                assert final is not None and old is not None and final == strict and old == oldstrict
                assert 1 <= len(calls) <= 3 and a['logical_calls'] == 1+len(calls)
                assert sum(c['generated_tokens'] for c in calls) <= 240
                assert all(c['audio_tokens'] > 0 and c['video_tokens'] > 0 for c in calls)
                assert sum(c['window'] is None for c in calls) <= 2
                count['proposal_correct'] += proposal == label
                count['verifier_saved_vs_proposal'] += proposal != label and final == label
                count['verifier_hurt_vs_proposal'] += proposal == label and final != label
                count['agreement_wrong'] += a['decision'] == 'agreement' and final != label
                count['disagreements'] += proposal != a['native_answer']
                count['accepted'] += a['decision'] == 'verified_revision'
                count['field_caps'] += bool(fs.get('field_token_caps', {}).get('EVIDENCE'))
                count['total_token_caps'] += sum(c['reached_token_limit'] for c in calls)
                anchors[fs['ANCHOR']] += 1; relations[fs['RELATION']] += 1
                if a['decision'] == 'agreement':
                    assert len(calls) == 1 and final == old and not a['review']
                elif a['decision'] == 'verified_revision':
                    assert a['confirmation_answer'] == proposal and final == proposal and not calls[-1]['reached_token_limit']
                else:
                    assert final == old
                if proposal != a['native_answer']:
                    routes[a['window_source']] += 1
                if a['review']:
                    assert len(calls) == 3 and not fs['field_token_caps']['EVIDENCE']
                    assert fs['ANCHOR'] != 'NONE' and fs['RELATION'] != 'GLOBAL'
                    assert a['review_window'][1]-a['review_window'][0] <= 12.000001
                    assert a['verification_memory_ids'] == ['verification_local']
                    replay_cases.append(dict(dataset=ds, model=model, input=inputs[q], offline_score=s,
                        scout_fields=fs, review_window=a['review_window'], decision=a['decision'],
                        raw_file=q, calls=calls, memories=a['memories']))
                count['reviews'] += a['review']
            assert count['field_caps'] == row['tool_field_caps'].get('EVIDENCE', 0)
            assert count['reviews'] == row['reviews']
            assert abs(sum(r['arm']['new_work_seconds'] for r in raw)-row['incremental_work_seconds']) < 1e-6
            trajectories[key] = dict(counts=dict(count), disagreement_routes=dict(routes), anchors=dict(anchors), relations=dict(relations))
            groups.append(dict(dataset=ds, model=model, **paired[key]['comparison']))
            comparisons[key] = {}
            for version in ['v10','v11']:
                oldtar = archives[version]
                assert tar.extractfile(f'data/{ds}.jsonl').read() == oldtar.extractfile(f'data/{ds}.jsonl').read()
                oldscores = {r['qid']: r for r in json.load(oldtar.extractfile(f'runs/reports/{ds}_{model}_paired_questions.json'))}
                assert set(scores) == set(oldscores)
                assert all(s['baseline'] == oldscores[s['qid']]['baseline'] and s['label'] == oldscores[s['qid']]['label'] for s in pairs)
                oldrow = next(r for r in previous[version+'_complete_table'] if (r['dataset'], r['model']) == (ds, model))
                comparisons[key][version] = dict(previous_correct=oldrow['correct'], current_correct=row['correct'],
                    net=row['correct']-oldrow['correct'], gain=sum(s['agent_correct'] and not oldscores[s['qid']]['agent_correct'] for s in pairs),
                    loss=sum(not s['agent_correct'] and oldscores[s['qid']]['agent_correct'] for s in pairs),
                    new_work_reduction_pct=100*(1-row['incremental_work_seconds']/oldrow['incremental_work_seconds']))
            all_pairs.extend(dict(dataset=ds, model=model, **s) for s in pairs)
        stages = rows('runs/stages.jsonl')
        pilots = [r for name in tar.getnames() if '/pilot/gpu' in name and name.endswith('/predictions.jsonl') and name.startswith(('runs/joint/','runs/avspeaker/')) for r in rows(name)]
        assert len(pilots) == 24 and all(r['status'] == 'ok' for r in pilots)
        pilot_load = sum(read(n)['load_seconds'] for n in tar.getnames() if n.startswith(('runs/joint/','runs/avspeaker/')) and '/pilot/gpu' in n and '/worker_start_' in n)
        probes = [read(f'runs/path_probe/{ds}_{model}.json') for ds in ['joint','avspeaker'] for model in ['omni','videollama']]
        assert all(p['passed'] and p['manifest_sha256'] == sha(manifest_bytes) for p in probes)
    aggregate = dict(total_questions=sum(r['planned'] for r in table), baseline_correct=sum(r['baseline_correct'] for r in table),
        method_correct=sum(r['correct'] for r in table), wrong_to_right=sum(g['wrong_to_right'] for g in groups),
        right_to_wrong=sum(g['right_to_wrong'] for g in groups), macro_delta_pp=statistics.mean(g['delta_pp'] for g in groups),
        new_work_seconds=sum(r['incremental_work_seconds'] for r in table), historical_native_work_seconds=sum(r['historical_native_work_seconds'] for r in table),
        model_load_seconds=sum(r['model_load_seconds'] for r in table), formal_wall_seconds=sum(s['wall_seconds'] for s in stages if s['phase']=='dev20'),
        pilot_wall_seconds=sum(s['wall_seconds'] for s in stages if s['phase']=='pilot'),
        pilot_measured_question_seconds=sum(r['seconds'] for r in pilots), pilot_load_seconds=pilot_load,
        forced_probe_work_seconds=sum(p['work_seconds'] for p in probes), forced_probe_load_seconds=sum(p['load_seconds'] for p in probes),
        executed_calls=sum(r['executed_calls'] for r in table), reviews=sum(r['reviews'] for r in table),
        field_caps=sum(r['tool_field_caps'].get('EVIDENCE', 0) for r in table), total_token_caps=sum(r['final_call_truncations']+r['intermediate_truncations'] for r in table))
    audit = dict(archive_sha256=digest, manifest_sha256=sha(manifest_bytes),
        validation='2426 unique planned outputs rescored from raw by frozen primary/strict parser; code/protocol/assets hashes, unchanged V10/V11 inputs, GPU allocation, route and call bounds, all task counts verified',
        aggregate=aggregate, trajectories=trajectories, comparisons=comparisons, groups=groups, table=table, tasks=details)
    dump(out/'AUDIT.json', audit)
    dump(out/'REPLAY_DIAGNOSTIC_CASES.json', replay_cases)
    dump(out/'ALL_PAIRED_QUESTIONS.json', all_pairs)
    dump(out/'REPLAY_TRACE_REVIEW.json', dict(scope='All seven natural replay traces; text and routing audit only, not semantic media verification',
        count=len(replay_cases), accepted=sum(r['decision']=='verified_revision' for r in replay_cases),
        gain=sum(r['offline_score']['agent_correct'] and not r['offline_score']['baseline_correct'] for r in replay_cases),
        loss=sum(not r['offline_score']['agent_correct'] and r['offline_score']['baseline_correct'] for r in replay_cases),
        unknown_local=sum(r['calls'][1]['model_output'].strip().upper().startswith('UNKNOWN') for r in replay_cases),
        textual_time_anchor_conflicts=[dict(qid='-dToWCRBMPg_task11_1', generated_time_seconds=86, actual_replay=[48.48,60.48]),
                                     dict(qid='TKGeXyYiLBo_task11_0', generated_time_seconds=171, actual_replay=[578.68,590.68])],
        limit='Generated times and event descriptions are not verified truth; these conflicts establish internal inconsistency, not which interval contains the true event. No inference or label changes.'))
    previous.update(v13_complete_table=table, v13_aggregate=aggregate,
        next_candidate='V14 not launched. Audit all seven natural replay traces and complete-round anchor/coverage routing evidence before deciding a new unified candidate; no component ablation or held-out evaluation.')
    dump(BASE/'DEVELOPMENT_LEADERBOARD.json', previous)
    lines = ['V13完整开发轮审计：2426/2426，同一20%，两模型×两数据集',
        '原始输出已用冻结主评分/strict逐题复核：缺失、运行错误、未解析均0；主分与strict一致。全部输入、唯一qid、任务分母、代码/协议/数据hash核验通过。',
        '归档SHA256：'+digest, '']
    for r,g in zip(table, groups):
        key=r['dataset']+'/'+r['model']; c=comparisons[key]
        lines += [f"{key}: native {r['baseline_correct']}/{r['planned']} ({r['baseline_accuracy_pct']:.5f}%) → V13 {r['correct']} ({r['accuracy_pct']:.5f}%)；改对/改错 {g['wrong_to_right']}/{g['right_to_wrong']}；增益{g['delta_pp']:+.5f}pp；聚类95%CI {g['cluster_bootstrap_95pct_pp']}",
            f"对V11净{c['v11']['net']:+d}题（改对{c['v11']['gain']}/改错{c['v11']['loss']}）；对V10净{c['v10']['net']:+d}题。新增均值{r['incremental_mean_seconds']:.5f}s；含历史native均值/P95 {r['mean_seconds']:.5f}/{r['p95_seconds']:.5f}s；逻辑调用{r['mean_logical_calls']:.5f}；回看{r['reviews']}。",
            '分项：'+'; '.join(f"{t['task']} {t['correct']}/{t['denominator']} 净{t['gain']-t['loss']:+d}" for t in details[key])]
    saved=sum(t['counts']['verifier_saved_vs_proposal'] for t in trajectories.values());hurt=sum(t['counts']['verifier_hurt_vs_proposal'] for t in trajectories.values())
    wrong=sum(t['counts']['agreement_wrong'] for t in trajectories.values())
    reduction=100*(1-aggregate['new_work_seconds']/previous['v11_aggregate']['new_work_seconds'])
    lines += ['', f"总计native {aggregate['baseline_correct']} → V13 {aggregate['method_correct']}；改对{aggregate['wrong_to_right']}/改错{aggregate['right_to_wrong']}，净34；四组等权增益{aggregate['macro_delta_pp']:.5f}pp。Omni净26，Video净8。V13比V11多5题，但AV Omni少1题、AV Video少2题；不能称各组都更好。",
        '两个Omni组区间高于零，两个Video组跨零；所有区间都来自反复使用的开发集，未修正多轮方法选择。当前是有限开发收益，不是独立泛化或大幅提升证明。',
        f"正式阶段墙钟合计{aggregate['formal_wall_seconds']/60:.5f}分钟；新增逐题工作{aggregate['new_work_seconds']:.6f}s（含CPU/IO/读取/裁剪/缓存准备），历史native另{aggregate['historical_native_work_seconds']:.6f}s；正式加载{aggregate['model_load_seconds']:.6f}s已包含阶段墙钟，不重复加到墙钟。新增工作比V11少{reduction:.2f}%，仅跨轮描述性比较，仍高于baseline。",
        f"技术成本单列：24题pilot阶段墙钟{aggregate['pilot_wall_seconds']:.6f}s，实测逐题工作{aggregate['pilot_measured_question_seconds']:.6f}s，加载{pilot_load:.6f}s；4条不计分强制路径检查工作{aggregate['forced_probe_work_seconds']:.6f}s、加载{aggregate['forced_probe_load_seconds']:.6f}s。各秒数口径不同，不相加冒充墙钟；此前V12等失败预检属于额外研发历史成本，未混入V13正式延迟。",
        f"正式末次/中间总token触限合计{aggregate['total_token_caps']}；EVIDENCE字段触48token上限{aggregate['field_caps']}条。字段截短仍存在，不可声称文本完全无截断。强制路径检查另有一次局部触限，记录保留。",
        f"路径：提案→最终复核挽回{saved}、损失{hurt}；一致但错{wrong}题。自然局部回看仅7次（Joint Omni3、AV Omni4、两个Video均0）。其余主要是候选提示/格式与全局核验，不能将收益归因为局部事件记忆、证据定位或额外媒体。",
        '全部7条回看文本已离线核对：6次保留native、1次接受并改对，未出现改错；其余净33题发生在没有局部回看的路径，不能据此做模块因果结论。3条局部输出UNKNOWN。两条全局描述的自报时间（86秒/171秒）与实际帧锚点裁剪（48.48—60.48秒/578.68—590.68秒）不一致；只证明生成时间和帧地址内部冲突，尚未确定真实事件在哪里。详见REPLAY_TRACE_REVIEW.json。',
        '候选直接正确数1019低于native1058，核验后1092。特别是Joint Video候选230、AV Video候选214均低于native259/238，说明当前候选提示未直接改善基础感知；全局核验和保留机制非常关键，但没有消融不能作独立贡献定量。',
        '候选A-D语法约束、合并观察与提案、缩短输出及更严格定位规则同时改变了完整方法。当前按用户要求未做消融，不能隔离各模块贡献。回看7题及其正确性是描述性路径统计，非因果证据；模型生成解释也不是真实事件证明。',
        '下一步不为占卡直接启动V14。保留V13作为当前较低成本的完整候选，先对全部7条自然回看做源帧—锚点—局部事件连接审计，并用整轮路由分布区分NONE/GLOBAL、字段截短、短片和可定位事件。REPLAY_DIAGNOSTIC_CASES.json保留7条原始调用用于离线审阅，不产生新预测或按标签调整路由。',
        '须先证实一个跨题、与标签无关的可修复定位/覆盖机制，写明新的统一流程和首次成本上限，才建立新目录/分支并进行标签盲技术pilot。否则停止盲目堆模块，报告收益边界。当前不运行80%或组件消融，不修改旧冻结代码或标签，不按题/组拼最佳。',
        '现有48条视觉/29条局部材料不重复生成；语义音频审听仍0/48，不把同模型转述当独立听证。最终方法选择还需未来冻结后的保留集评估，当前20%只称开发集。']
    (out/'ANALYSIS_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(aggregate)); print(json.dumps(trajectories)); print(json.dumps(comparisons))


if __name__ == '__main__':
    main()
