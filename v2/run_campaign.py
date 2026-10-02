"""Two independent GPU queues. No training, scores never influence dispatch."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from evaluate import pilot_gate, write

HERE = Path(__file__).resolve().parent
ENVS = {'omni': '/root/anaconda3/envs/Joint-avqa/bin/python', 'videollama': '/root/anaconda3/envs/Joint-avqa-video/bin/python'}


def lane(model):
    gpu = '0' if model == 'omni' else '3'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    d = HERE / 'runs'; d.mkdir(exist_ok=True)
    for phase in ['pilot', 'dev20']:
        for dataset in ['avspeaker', 'joint']:
            dest = d / dataset / model / phase
            if (dest / 'complete.json').exists():
                if phase == 'pilot':
                    pilot_gate(dataset, model)
                continue
            for attempt in range(1, 4):
                write(d / (model + '_state.json'), {'model': model, 'gpu': gpu, 'dataset': dataset, 'phase': phase, 'attempt': attempt, 'status': 'running', 'time': time.time()})
                log = d / 'logs' / f'{model}_{dataset}_{phase}_attempt{attempt}_{int(time.time())}.log'
                log.parent.mkdir(exist_ok=True)
                with log.open('w') as f:
                    p = subprocess.Popen([ENVS[model], str(HERE / 'worker.py'), '--model', model, '--dataset', dataset, '--phase', phase, '--gpu', gpu], env=env, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
                    write(d / (model + '_worker.json'), {'pid': p.pid, 'model': model, 'dataset': dataset, 'phase': phase, 'log': str(log), 'time': time.time()})
                    code = p.wait()
                if code == 0:
                    if phase == 'pilot':
                        pilot_gate(dataset, model)
                    break
                if code != 42 or attempt == 3:
                    raise RuntimeError(f'{model}/{dataset}/{phase} failed: code={code}; inspect {log}; no score-driven retry')
    write(d / (model + '_state.json'), {'model': model, 'status': 'complete', 'time': time.time()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lane', choices=list(ENVS)); a = ap.parse_args()
    if a.lane:
        try:
            lane(a.lane)
        except Exception as exc:
            write(HERE / 'runs' / (a.lane + '_state.json'), {'model': a.lane, 'status': 'needs_diagnosis', 'error': str(exc), 'time': time.time()})
            raise
        return
    d = HERE / 'runs'; d.mkdir(exist_ok=True)
    lock = (d / 'campaign.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Protect other workloads. Never terminate GPU processes to make room.
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    used = {x.split(',')[0].strip(): int(x.split(',')[1]) for x in raw.splitlines()}
    assert all(used[g] < 1000 for g in ['0', '3']), 'Authorized GPUs are busy; do not duplicate jobs'
    children = []
    for model in ENVS:
        log = (d / (model + '_scheduler.log')).open('a')
        children.append(subprocess.Popen([sys.executable, str(HERE / 'run_campaign.py'), '--lane', model], stdout=log, stderr=subprocess.STDOUT, cwd=HERE))
    last_summary = 0
    while any(p.poll() is None for p in children):
        write(d / 'state.json', {'status': 'running', 'pid': os.getpid(), 'lanes': [{'pid': p.pid, 'exit_code': p.poll()} for p in children], 'time': time.time()})
        if time.time() - last_summary > 300:
            subprocess.run([sys.executable, str(HERE / 'evaluate.py')], check=True)
            last_summary = time.time()
        time.sleep(30)
    ok = all(p.returncode == 0 for p in children)
    subprocess.run([sys.executable, str(HERE / 'evaluate.py')], check=True)
    write(d / 'state.json', {'status': 'complete' if ok else 'needs_diagnosis', 'time': time.time(), 'exit_codes': [p.returncode for p in children]})
    if ok:
        with tarfile.open(d / 'complete_results.tar.gz', 'w:gz') as tar:
            for p in sorted(d.rglob('*')):
                if p.is_file() and p.suffix not in ['.gz', '.lock']:
                    tar.add(p, arcname=p.relative_to(d))


if __name__ == '__main__':
    main()
