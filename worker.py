"""GPU worker: inference never opens labels, old predictions or audit cases."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import traceback
from agent import global_prompt, run_arm
from runtime import HERE, DATA, Runtime, probe, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def code_hashes():
    return {p.relative_to(HERE).as_posix(): sha(p) for p in sorted(HERE.rglob('*.py')) if not set(p.relative_to(HERE).parts) & {'runs', 'cache', '.git'}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['omni', 'videollama'], required=True)
    ap.add_argument('--dataset', choices=['joint', 'avspeaker'], required=True)
    ap.add_argument('--phase', choices=['pilot', 'dev20'], required=True)
    ap.add_argument('--gpu', choices=['0', '3'], required=True)
    args = ap.parse_args()
    assert args.gpu == os.environ['CUDA_VISIBLE_DEVICES']
    lock = open('/tmp/avqa-evidence-agent-gpu' + args.gpu + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = [json.loads(s) for s in (HERE / 'data' / (args.dataset + '.jsonl')).read_text().splitlines()]
    manifest = json.loads((HERE / 'manifests' / (args.dataset + '_split.json')).read_text())
    assert sha(HERE / 'data' / (args.dataset + '.jsonl')) == manifest['subset_sha256']
    assert all(not set(r) & {'answer', 'label', 'correct_answer', 'correct_prefix', 'answer_label'} for r in rows)
    if args.phase == 'pilot':
        ids = json.loads((HERE / 'manifests' / 'pilot.json').read_text())[args.dataset]['qids']
        rows = [r for q in ids for r in rows if r['qid'] == q]
    dest = HERE / 'runs' / args.dataset / args.model / args.phase
    dest.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((HERE / 'manifests' / 'code_freeze.json').read_text())
    assert frozen['python_sha256'] == code_hashes(), 'Code changed after freeze; new version required'
    for rel, digest in frozen['external_sha256'].items():
        assert sha(HERE.parents[1] / rel) == digest, 'Native dependency changed: ' + rel
    config = {'args': vars(args), 'protocol_sha256': sha(HERE / 'protocol.json'), 'code': frozen,
              'qids': [r['qid'] for r in rows], 'subset_sha256': manifest['subset_sha256'],
              'packages': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'decord', 'flash-attn']}}
    cp = dest / 'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config
    else:
        write(cp, config)
    records = [json.loads(s) for s in (dest / 'predictions.jsonl').read_text().splitlines()] if (dest / 'predictions.jsonl').exists() else []
    done = {r['qid'] for r in records if r['status'] == 'ok'}
    assert len(done) == sum(r['status'] == 'ok' for r in records), 'Duplicate successful qid'
    cache = HERE / 'cache' / (args.dataset + '_' + args.model)
    cache.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(args.dataset, args.model, cache)
    audio_policy = ('at most 32 sparse frames across the full clip; ' if args.dataset == 'joint' else 'native video sampling; ')
    audio_policy += ('continuous audio from the first 300 seconds' if args.dataset == 'joint' else 'continuous full-clip audio') if args.model == 'omni' else 'eight 2-second audio excerpts; this is sparse audio, not a continuous transcript'
    for row in rows:
        if row['qid'] in done:
            continue
        start = time.time()
        result = {'qid': row['qid'], 'task': row['task'], 'dataset': args.dataset, 'model': args.model,
                  'phase': args.phase, 'started_at': start, 'gpu': args.gpu}
        try:
            duration = probe(DATA[args.dataset] / row['clip_path'])
            infer = lambda prompt, interval, stage: runtime.call(row, prompt, interval, stage)
            global_result = infer(global_prompt(row, duration, audio_policy), None, 'global_plan')
            if args.phase == 'pilot':
                # Baseline parity and repetition are execution gates, not accuracy gates.
                result['direct_baseline_check'] = infer(row['question_prompt'], None, 'direct_baseline_check')
                repeat = infer(global_prompt(row, duration, audio_policy), None, 'global_repeat')
                keys = ['model_output', 'generated_tokens', 'audio_sha256', 'video_tensor_sha256', 'raw_frames_sha256', 'input_tokens']
                result['repeat_matches'] = {k: global_result.get(k) == repeat.get(k) for k in keys}
                assert all(result['repeat_matches'].values()), 'Same-seed global repeat mismatch'
            # Swap arm order deterministically to avoid systematic timing/order advantages.
            modes = ['agent', 'fixed'] if int(hashlib.sha256(row['qid'].encode()).hexdigest(), 16) % 2 else ['fixed', 'agent']
            result['arms'] = {m: run_arm(row, duration, global_result, m, infer) for m in modes}
            result.update(status='ok', duration_seconds=duration, actual_new_calls=7)
            done.add(row['qid'])
        except Exception as exc:
            result.update(status='error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        result.update(seconds=time.time() - start, finished_at=time.time())
        with (dest / 'predictions.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())
        write(dest / 'progress.json', {'completed': len(done), 'total': len(rows), 'last_qid': row['qid'],
                                     'last_status': result['status'], 'updated_at': time.time(), 'last_seconds': result['seconds']})
        print(json.dumps({k: result.get(k) for k in ['qid', 'status', 'seconds', 'error_type']}), flush=True)
        if result['status'] != 'ok':
            raise SystemExit(42 if 'out of memory' in result['error'].lower() else 1)
    write(dest / 'complete.json', {'complete': True, 'n': len(done), 'time': time.time()})


if __name__ == '__main__':
    main()
