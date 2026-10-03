"""Frozen algorithm continuation only; never launches native/pilot or new methods."""
import fcntl,json,os,subprocess,sys,time,tarfile
from settings import *

def stage(model,ds):
    frozen();active={};attempts={};started=time.time()
    def launch(g):
        logs=HERE/'runs/logs';logs.mkdir(parents=True,exist_ok=True)
        attempt=len(list(logs.glob(f'{model}_{ds}_gpu{g}_attempt*.log')))+1
        assert attempt<=3,'OOM process-attempt limit'
        log=logs/f'{model}_{ds}_gpu{g}_attempt{attempt}.log'
        with log.open('x') as f:
            p=subprocess.Popen([ENVS[model],str(HERE/'worker_memory.py'),'--model',model,'--dataset',ds,'--phase','remaining80','--gpu',g],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=g,HF_HUB_OFFLINE='1'),stdout=f,stderr=subprocess.STDOUT)
        active[g]=p;attempts[g]=attempt
        write(HERE/f'runs/worker_gpu{g}.json',dict(pid=p.pid,model=model,dataset=ds,phase='remaining80',gpu=g,attempt=attempt,log=str(log),time=time.time()))
    for g in GPUS:
        if not (HERE/f'runs/{ds}/{model}/remaining80/gpu{g}/complete.json').exists():launch(g)
    try:
        while active:
            write(HERE/'runs/state.json',dict(status='running',model=model,dataset=ds,phase='remaining80',pid=os.getpid(),workers={g:p.pid for g,p in active.items()},time=time.time()))
            for g,p in list(active.items()):
                rc=p.poll()
                if rc is None:continue
                del active[g]
                if rc==42 and attempts[g]<3:launch(g)
                elif rc!=0:raise RuntimeError(f'{ds}/{model}/GPU{g} exit={rc}; retained raw/logs, needs diagnosis')
            if active:time.sleep(5)
    finally:
        for p in active.values():
            if p.poll() is None:p.terminate()
        for p in active.values():p.wait(timeout=60)
    with (HERE/'runs/stages.jsonl').open('a') as f:f.write(json.dumps(dict(model=model,dataset=ds,started_at=started,finished_at=time.time(),wall_seconds=time.time()-started))+'\n')
    subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True)

def main():
    frozen();lock=open('/tmp/avqa-source-memory-campaign.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'runs/launch.json').exists(),'No duplicate campaign or automatic restart'
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={s.split(',')[0].strip():int(s.split(',')[1]) for s in raw.splitlines()}
    assert all(used[g]<1000 for g in GPUS),'Authorized GPU busy; never displace another task'
    write(HERE/'runs/launch.json',dict(pid=os.getpid(),time=time.time(),gpus=GPUS,method='frozen_v13',new_predictions=9704,reused_predictions=2426,baseline_generation=False))
    try:
        for model in ['omni','videollama']:
            for ds in ['joint','avspeaker']:stage(model,ds)
        from summarize import main as summarize
        assert summarize();frozen()
        write(HERE/'runs/state.json',dict(status='packing',time=time.time()))
        p=HERE/'runs/complete_results.tar.gz'
        with tarfile.open(p,'w:gz') as tar:
            for f in sorted((HERE/'runs').rglob('*')):
                if f.is_file() and f.suffix not in ['.gz','.lock']:tar.add(f,arcname=str(f.relative_to(HERE)))
            for f in sorted((HERE/'data').glob('*')):tar.add(f,arcname=str(f.relative_to(HERE)))
            for f in sorted(HERE.glob('*')):
                if f.is_file():tar.add(f,arcname='source/'+f.name)
        write(HERE/'runs/state.json',dict(status='complete',time=time.time(),archive_sha256=sha(p)))
    except Exception as e:
        write(HERE/'runs/state.json',dict(status='needs_diagnosis',time=time.time(),error_type=type(e).__name__,error=str(e)));raise

if __name__=='__main__':main()
