"""One frozen complete-method campaign, same model on three independent GPUs."""
import fcntl,json,os,subprocess,sys,time,tarfile
from settings import *

def stage(model,ds,phase):
    c=frozen();active={};attempts={};start=time.time();failures=[]
    def launch(g):
        logs=HERE/'runs/logs';logs.mkdir(parents=True,exist_ok=True)
        attempt=len(list(logs.glob(f'{model}_{ds}_{phase}_gpu{g}_attempt*.log')))+1
        assert attempt<=3,'OOM allowance exhausted'
        log=logs/f'{model}_{ds}_{phase}_gpu{g}_attempt{attempt}.log'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=g,HF_HUB_OFFLINE='1')
        with log.open('x') as f:
            p=subprocess.Popen([ENVS[model],str(HERE/'worker_memory.py'),'--model',model,'--dataset',ds,'--phase',phase,'--gpu',g],cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT)
        active[g]=p;attempts[g]=attempt
        write(HERE/f'runs/worker_gpu{g}.json',dict(pid=p.pid,model=model,dataset=ds,phase=phase,gpu=g,attempt=attempt,log=str(log),time=time.time()))
    for g in GPUS:
        if not (HERE/f'runs/{ds}/{model}/{phase}/gpu{g}/complete.json').exists():launch(g)
    last=time.time()
    while active:
        write(HERE/'runs/state.json',dict(status='running',model=model,dataset=ds,phase=phase,pid=os.getpid(),workers={g:p.pid for g,p in active.items()},time=time.time()))
        for g,p in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            del active[g]
            if rc==42 and attempts[g]<3:launch(g)
            elif rc!=0:failures.append(dict(gpu=g,exit_code=rc))
        if time.time()-last>180:
            subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True);last=time.time()
        if active:time.sleep(5)
    event=dict(model=model,dataset=ds,phase=phase,started_at=start,finished_at=time.time(),wall_seconds=time.time()-start,failures=failures)
    with (HERE/'runs/stages.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    assert not failures,event
    if phase=='pilot':
        rr=[r for g in GPUS for r in readlines(HERE/f'runs/{ds}/{model}/pilot/gpu{g}/predictions.jsonl') if r['status']=='ok']
        assert len(rr)==6 and len({r['qid'] for r in rr})==6
        assert all(all(r['native_parity'].values()) and all(r['observation_repeat'].values()) for r in rr)
        valid=sum(r['arm']['scout_schema_valid'] for r in rr)
        assert valid>=4,('Observation schema unusable',valid)
        write(HERE/f'runs/{ds}/{model}/pilot/gate.json',dict(passed=True,valid_scouts=valid,n=6,
            criterion='Native media/text parity, repeated observation text/tokens, schema >=4/6, final not truncated. No labels or accuracy threshold.'))

def main():
    frozen();lock=open('/tmp/avqa-source-memory-campaign.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={s.split(',')[0].strip():int(s.split(',')[1]) for s in raw.splitlines()}
    assert all(used[g]<1000 for g in GPUS),'Authorized GPU busy; do not displace another job'
    write(HERE/'runs/launch.json',dict(pid=os.getpid(),time=time.time(),gpus=GPUS))
    try:
        for model in ['omni','videollama']:
            for ds in ['joint','avspeaker']:
                stage(model,ds,'pilot');stage(model,ds,'dev20')
                subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True)
        from summarize import main as summarize
        assert summarize()
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
