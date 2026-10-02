"""Two fixed GPU lanes, pilot gates before dev20, no score-driven decisions."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import tarfile
import time
from support import HERE,write

ENVS={'omni':'/root/anaconda3/envs/Joint-avqa/bin/python','videollama':'/root/anaconda3/envs/Joint-avqa-video/bin/python'}


def lane(model,pilot_only):
    gpu='0' if model=='omni' else '3'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    d=HERE/'runs'
    for phase in (['pilot'] if pilot_only else ['pilot','dev20']):
        for ds in ['avspeaker','joint']:
            dest=d/f'{ds}/{model}/{phase}'
            if (dest/'complete.json').exists():
                if phase=='pilot':subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--pilot-gate',ds,model],check=True)
                continue
            # Persist attempts across scheduler restarts; do not reset OOM allowance.
            prior=list((d/'logs').glob(f'{model}_{ds}_{phase}_attempt*.log'))
            for attempt in range(len(prior)+1,4):
                write(d/f'{model}_state.json',dict(model=model,gpu=gpu,dataset=ds,phase=phase,attempt=attempt,status='running',time=time.time()))
                log=d/f'logs/{model}_{ds}_{phase}_attempt{attempt}.log';log.parent.mkdir(exist_ok=True)
                with log.open('x') as f:
                    p=subprocess.Popen([ENVS[model],str(HERE/'worker.py'),'--model',model,'--dataset',ds,'--phase',phase,'--gpu',gpu],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=HERE)
                    write(d/f'{model}_worker.json',dict(pid=p.pid,model=model,dataset=ds,phase=phase,log=str(log),time=time.time()))
                    code=p.wait()
                if code==0:
                    if phase=='pilot':subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--pilot-gate',ds,model],check=True)
                    break
                if code!=42 or attempt==3:raise RuntimeError(f'{model}/{ds}/{phase} exit={code}; inspect {log}')
            else:raise RuntimeError(f'{model}/{ds}/{phase}: OOM retry allowance exhausted')
    write(d/f'{model}_state.json',dict(model=model,status='pilot_complete' if pilot_only else 'complete',time=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=list(ENVS));p.add_argument('--pilot-only',action='store_true');a=p.parse_args()
    d=HERE/'runs';d.mkdir(exist_ok=True)
    if a.lane:
        try:lane(a.lane,a.pilot_only)
        except Exception as exc:
            write(d/f'{a.lane}_state.json',dict(status='needs_diagnosis',error=str(exc),time=time.time()));raise
        return
    lock=(d/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={s.split(',')[0].strip():int(s.split(',')[1]) for s in raw.splitlines()}
    assert all(used[g]<1000 for g in ['0','3']),'Authorized GPUs busy; no duplicate launch'
    children=[]
    for model in ENVS:
        cmd=[sys.executable,str(HERE/'run_campaign.py'),'--lane',model]+(['--pilot-only'] if a.pilot_only else [])
        f=(d/f'{model}_scheduler.log').open('a');children.append(subprocess.Popen(cmd,cwd=HERE,stdout=f,stderr=subprocess.STDOUT))
    last=0
    while any(c.poll() is None for c in children):
        write(d/'state.json',dict(status='running',pilot_only=a.pilot_only,pid=os.getpid(),lanes=[dict(pid=c.pid,exit_code=c.poll()) for c in children],time=time.time()))
        if not a.pilot_only and time.time()-last>300:
            subprocess.run([sys.executable,str(HERE/'evaluate.py')],check=True);last=time.time()
        time.sleep(10)
    ok=all(c.returncode==0 for c in children)
    if not a.pilot_only:subprocess.run([sys.executable,str(HERE/'evaluate.py')],check=True)
    write(d/'state.json',dict(status=('pilot_complete' if a.pilot_only else 'complete') if ok else 'needs_diagnosis',time=time.time(),exit_codes=[c.returncode for c in children]))
    if ok and not a.pilot_only:
        with tarfile.open(d/'complete_results.tar.gz','w:gz') as t:
            for item in sorted(d.rglob('*')):
                if item.is_file() and item.suffix not in ['.gz','.lock']:t.add(item,arcname=item.relative_to(d))


if __name__=='__main__':main()
