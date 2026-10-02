"""CPU-only protocol freeze before any inference; no labels/predictions read."""
import concurrent.futures
import json
import time
from pathlib import Path
from runtime import HERE, ROOT, DATA, probe, sha


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def main():
    assert not list((HERE / 'runs').rglob('predictions.jsonl')), 'Already started: do not refreeze'
    pilots = {}
    for dataset in ['joint', 'avspeaker']:
        rows = [json.loads(s) for s in (HERE / 'data' / (dataset + '.jsonl')).read_text().splitlines()]
        def inspect(r):
            return r['qid'], probe(DATA[dataset] / r['clip_path'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            durations = dict(pool.map(inspect, rows))
        ordered = sorted(durations, key=lambda q: (durations[q], q))
        pilots[dataset] = {'qids': [ordered[0], ordered[-1]], 'criterion': 'shortest and longest selected input, no labels',
                           'durations': {q: durations[q] for q in [ordered[0], ordered[-1]]}}
        write(HERE / 'manifests' / (dataset + '_durations.json'), durations)
        print(dataset, pilots[dataset], flush=True)
    write(HERE / 'manifests/pilot.json', pilots)
    vendor = ROOT / 'reproduction/custom_v1/vendor/VideoLLaMA2-official'
    files = [p for p in vendor.rglob('*.py') if '.git' not in p.parts]
    source = [p for p in HERE.rglob('*.py') if not set(p.relative_to(HERE).parts) & {'runs', 'cache', '.git'}]
    write(HERE / 'manifests/code_freeze.json', {'time': time.time(), 'python_sha256': {p.relative_to(HERE).as_posix(): sha(p) for p in sorted(source)},
                                            'external_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)},
                                            'protocol_sha256': sha(HERE / 'protocol.json')})


if __name__ == '__main__':
    main()
