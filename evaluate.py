"""Independent scorer. Never imported by inference; no score-driven selection."""
import collections
import importlib.util
import json
import random
from pathlib import Path
from joint_parser import parse as joint_parse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('avscore', HERE / 'compat/avspeaker/common.py')
av = importlib.util.module_from_spec(spec); spec.loader.exec_module(av)


def readlines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'); t.replace(p)


def old_records(dataset, model):
    if dataset == 'joint':
        paths = [ROOT / f'results/custom_v1_20261002/{model}/full/gpu{gpu}/predictions.jsonl' for gpu in [0, 3]]
    else:
        paths = [ROOT / f'results/avspeaker_v1_20261002/{model}/full/predictions.jsonl']
    out = {}
    for p in paths:
        for r in readlines(p):
            if r['status'] == 'ok':
                assert r['qid'] not in out
                out[r['qid']] = r
    return out


def answer(dataset, text, row):
    if dataset == 'joint':
        a = joint_parse(text, row['prefix2text'])['answer']
        return a, a
    return av.upstream_parse(text), av.strict_parse(text)


def pilot_gate(dataset, model):
    records = readlines(HERE / f'runs/{dataset}/{model}/pilot/predictions.jsonl')
    old = old_records(dataset, model)
    checks = []
    for r in records:
        assert r['status'] == 'ok'
        direct = r['direct_baseline_check']; original = old[r['qid']]
        keys = ['frames', 'audio_sha256', 'video_tensor_sha256', 'raw_frames_sha256', 'input_tokens', 'audio_tokens', 'video_tokens']
        media = {k: direct.get(k) == original.get(k) for k in keys}
        assert all(media.values()), ('baseline_media_or_tokens_differ', r['qid'], media)
        checks.append({'qid': r['qid'], 'media_parity': media, 'direct_text_identical': direct['model_output'] == original['model_output'],
                       'repeat_matches': r['repeat_matches'], 'agent_valid_tool_requests': sum(not d['fallback'] for d in r['arms']['agent']['decisions']),
                       'final_token_limit': {a: r['arms'][a]['calls'][-1]['reached_token_limit'] for a in ['agent', 'fixed']}})
    assert len(checks) >= 2 and sum(c['agent_valid_tool_requests'] for c in checks) > 0, 'No usable tool request in pilot'
    assert all(not any(c['final_token_limit'].values()) for c in checks), 'Final answer capacity failed'
    write(HERE / f'runs/{dataset}/{model}/pilot/gate.json', {'passed': True, 'criteria': 'execution, baseline input parity, repeatability, bounded usable tools; no accuracy threshold', 'checks': checks})


def paired(rs, a, b):
    n = len(rs)
    gain = sum(r[a] and not r[b] for r in rs)
    loss = sum(r[b] and not r[a] for r in rs)
    result = {'n': n, 'wrong_to_right': gain, 'right_to_wrong': loss, 'net_correct': gain - loss,
              'delta_pp': 100 * (gain - loss) / n if n else None}
    if n:
        groups = collections.defaultdict(list)
        for r in rs:
            groups[r['cluster']].append(int(r[a]) - int(r[b]))
        clusters = [(sum(v), len(v)) for v in groups.values()]
        rng = random.Random(42); draws = []
        for _ in range(2000):
            sampled = rng.choices(clusters, k=len(clusters))
            draws.append(100 * sum(x[0] for x in sampled) / sum(x[1] for x in sampled))
        draws.sort()
        result.update(cluster_bootstrap_95pct_pp=[draws[49], draws[1949]], clusters=len(clusters), bootstrap_repeats=2000)
    return result


def main():
    table = []; details = {}; pairs = {}
    for dataset in ['avspeaker', 'joint']:
        rows = readlines(HERE / 'data' / (dataset + '.jsonl'))
        # Only this scorer loads answer labels, after predictions have been persisted.
        lp = ROOT / ('reproduction/frozen_seed42/labels.json' if dataset == 'joint' else 'reproduction/avspeaker_v1/labels.json')
        if not lp.exists():
            lp = ROOT / 'results/avspeaker_v1_20261002/evidence_snapshot/reproduction/avspeaker_v1/labels.json'
        labels = json.loads(lp.read_text(encoding='utf-8'))
        for model in ['omni', 'videollama']:
            old = old_records(dataset, model)
            records = readlines(HERE / f'runs/{dataset}/{model}/dev20/predictions.jsonl')
            latest = {r['qid']: r for r in records}
            matched = []; task_rows = []; parsed_artifact = []
            for row in rows:
                q = row['qid']; y = labels[q]['correct_prefix'] if dataset == 'joint' else labels[q]
                base, _ = answer(dataset, old[q]['model_output'], row)
                rr = latest.get(q)
                p = {'qid': q, 'task': row['task'], 'baseline': base, 'label': y,
                     'cluster': row.get('video_name', row['clip_path'].rsplit('/', 1)[-1].rsplit('_', 2)[0])}
                for mode in ['agent', 'fixed']:
                    arm = rr.get('arms', {}).get(mode) if rr and rr['status'] == 'ok' else None
                    ans, strict = answer(dataset, arm['model_output'], row) if arm else (None, None)
                    p[mode] = ans; p[mode + '_strict'] = strict
                    p[mode + '_correct'] = ans == y
                p['baseline_correct'] = base == y
                p['complete'] = bool(rr and rr['status'] == 'ok')
                parsed_artifact.append(p)
                if p['complete']:
                    matched.append(p)
            complete = len(matched) == len(rows)
            for mode in ['baseline', 'fixed', 'agent']:
                selected = parsed_artifact if mode == 'baseline' else matched
                correct = sum(r[mode] == r['label'] for r in selected)
                parsed = sum(r[mode] is not None for r in selected)
                cost = []
                if mode != 'baseline':
                    cost = [latest[r['qid']]['arms'][mode] for r in matched]
                actual = [call for arm in cost for call in arm['calls']]
                entry = {'dataset': dataset, 'model': model, 'method': mode, 'complete': mode == 'baseline' or complete,
                         'planned_n': len(rows), 'observed_n': len(selected), 'correct': correct,
                         'accuracy_observed_pct': 100 * correct / len(selected) if selected else None,
                         'accuracy_full_denominator_pct': 100 * correct / len(rows), 'unparsed_observed': len(selected) - parsed,
                         'parsed_only_pct': 100 * correct / parsed if parsed else None,
                         'historical_error_attempts': sum(r['status'] == 'error' for r in records),
                         'mean_logical_calls': 1 if mode == 'baseline' else 4,
                         'mean_model_seconds': sum(c['seconds'] for c in actual) / len(cost) if cost else None,
                         'mean_input_tokens': sum(c.get('input_tokens', 0) for c in actual) / len(cost) if cost else None,
                         'mean_frames': sum(c.get('frames', 0) for c in actual) / len(cost) if cost else None,
                         'mean_supplied_audio_seconds': sum(c.get('audio_seconds_supplied', 0) for c in actual) / len(cost) if cost else None,
                         'mean_generated_tokens': sum(c['generated_tokens'] for c in actual) / len(cost) if cost else None,
                         'intermediate_token_limits': sum(c['reached_token_limit'] for c in actual if not c['stage'].endswith('_final')),
                         'final_token_limits': sum(c['reached_token_limit'] for c in actual if c['stage'].endswith('_final')),
                         'tool_fallbacks': sum(d['fallback'] for arm in cost for d in arm['decisions'])}
                if mode == 'baseline':
                    entry['mean_model_seconds'] = sum(old[r['qid']].get('elapsed_seconds', old[r['qid']].get('seconds', 0)) for r in rows) / len(rows)
                table.append(entry)
                for task in sorted({r['task'] for r in rows}):
                    sub = [r for r in selected if r['task'] == task]
                    task_rows.append({'method': mode, 'task': task, 'planned_n': sum(r['task'] == task for r in rows), 'observed_n': len(sub),
                                      'correct': sum(r[mode] == r['label'] for r in sub),
                                      'accuracy_observed_pct': 100 * sum(r[mode] == r['label'] for r in sub) / len(sub) if sub else None})
            key = dataset + '/' + model
            details[key] = task_rows
            # Partial summaries deliberately omit effect confidence intervals.
            pairs[key] = {'complete': complete, 'observed': len(matched), 'planned': len(rows),
                          'agent_vs_baseline': paired(matched, 'agent_correct', 'baseline_correct') if complete else None,
                          'agent_vs_fixed': paired(matched, 'agent_correct', 'fixed_correct') if complete else None,
                          'fixed_vs_baseline': paired(matched, 'fixed_correct', 'baseline_correct') if complete else None}
            write(HERE / 'runs/reports' / (dataset + '_' + model + '_paired_questions.json'), parsed_artifact)
    write(HERE / 'runs/reports/TOTAL_TABLE.json', table)
    write(HERE / 'runs/reports/TASK_DETAILS.json', details)
    write(HERE / 'runs/reports/PAIRED_SUMMARY.json', pairs)
    lines = ['AV evidence agent v1 — fixed 20% development comparison',
             'Baseline is replayed from unchanged historical raw outputs, using the same frozen scorer as both new arms.',
             'Agent/fixed: 4 logical calls per question; shared global call computed once. Actual tokens/time differ.',
             'Partial observed percentages are provisional; no claims of improvement until a complete paired subset.',
             'This is development data, not an untouched test split. Remaining questions can share source videos.', '']
    for r in table:
        value = r['accuracy_observed_pct']
        lines.append(f"{r['dataset']:10} {r['model']:10} {r['method']:8} {r['observed_n']:4}/{r['planned_n']} correct={r['correct']:4} accuracy={value if value is not None else 'pending'} complete={r['complete']}")
    if all(r['complete'] for r in table):
        for key, p in pairs.items():
            lines.append(key + ': ' + json.dumps(p, ensure_ascii=False))
    (HERE / 'runs/reports/REPORT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return all(r['complete'] for r in table)


if __name__ == '__main__':
    main()
