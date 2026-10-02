"""Freeze stratified development subsets without reading labels or predictions."""
import collections
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def choose(rows, dataset):
    groups = collections.defaultdict(list)
    for r in rows:
        assert not set(r) & {'answer', 'label', 'correct_answer', 'correct_prefix', 'answer_label'}
        groups[r['task']].append(r)
    target = int(len(rows) * .2 + .5)
    counts = {t: math.floor(len(rs) * .2) for t, rs in groups.items()}
    order = sorted(groups, key=lambda t: (-(len(groups[t]) * .2 - counts[t]), t))
    for t in order[:target - sum(counts.values())]:
        counts[t] += 1
    selected = []
    for t in sorted(groups):
        ranked = sorted(groups[t], key=lambda r: hashlib.sha256(
            f'av-agent-dev-v1|42|{dataset}|{r["qid"]}'.encode()).hexdigest())
        selected.extend(ranked[:counts[t]])
    # Round-robin tasks so early progress contains all categories, never score-ordered.
    ordered = []
    buckets = {t: [r for r in selected if r['task'] == t] for t in sorted(groups)}
    for i in range(max(counts.values())):
        ordered.extend(buckets[t][i] for t in buckets if i < len(buckets[t]))
    assert len(ordered) == target == len({r['qid'] for r in ordered})
    return ordered, counts


def main():
    av = ROOT / 'reproduction/avspeaker_v1/inputs.jsonl'
    if not av.exists():
        av = ROOT / 'results/avspeaker_v1_20261002/evidence_snapshot/reproduction/avspeaker_v1/inputs.jsonl'
    sources = {'joint': ROOT / 'reproduction/frozen_seed42/inputs.jsonl', 'avspeaker': av}
    expected = {'joint': '3af376e083b92501154aa9aed9d5be1466ce9a7b6e12f498a04e9cd60b7cee50',
                'avspeaker': 'c6f824433ebb0874c33dc579a656157ad11d1b84ec2998d9b59e37d940356d30'}
    for dataset, p in sources.items():
        assert sha(p) == expected[dataset]
        rows = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
        selected, counts = choose(rows, dataset)
        d = HERE / 'data'; d.mkdir(exist_ok=True)
        out = d / (dataset + '.jsonl')
        content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected)
        if out.exists():
            assert out.read_text(encoding='utf-8') == content
        else:
            out.write_text(content, encoding='utf-8', newline='\n')
        write(HERE / 'manifests' / (dataset + '_split.json'), {
            'dataset': dataset, 'total': len(rows), 'selected': len(selected),
            'selection': 'largest-remainder 20% per task, SHA256 fixed seed42 rank, no labels/correctness',
            'role': 'development comparison subset; repeated experiments are not untouched testing',
            'source_sha256': sha(p), 'subset_sha256': sha(out), 'by_task': counts,
            'qids': [r['qid'] for r in selected],
            'remaining_qids': [r['qid'] for r in rows if r['qid'] not in {x['qid'] for x in selected}],
            'remaining_is_unseen': False,
            'note': 'Prior baseline/audit inspected the benchmark. Same-video related questions may span the split.'})
        print(dataset, len(selected), counts)


if __name__ == '__main__':
    main()
