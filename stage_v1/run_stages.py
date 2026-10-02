"""Strict sequential model/method stages, with GPU0+3 independent replicas."""
import fcntl,json,os,subprocess,sys,time,tarfile
from common_stage import *

ENVS={'omni':'/root/anaconda3/envs/Joint-avqa/bin/python','videollama':'/root/anaconda3/envs/Joint-avqa-video/bin/python'}


def stage(model,ds,mode):
    c=config();pending=[]
    for gpu in ['0','3']:
        dest=HERE/f'runs/{ds}/{model}/{mode}/gpu{gpu}'
        if (dest/'complete.json').exists():continue
        if mode!='check' and not c['plan'][ds+'/'+model+'/'+mode]['gpu'+gpu]:continue
        pending.append(gpu)
    if not pending:return
    start=time.time();active={};attempts={}
    def launch(gpu):
        prior=list((HERE/'runs/logs').glob(f'{model}_{ds}_{mode}_gpu{gpu}_attempt*.log'))
        attempt=len(prior)+1;assert attempt<=3,'Retry allowance exhausted'
        log=HERE/f'runs/logs/{model}_{ds}_{mode}_gpu{gpu}_attempt{attempt}.log';log.parent.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        args=[ENVS[model],str(HERE/'worker_stage.py'),'--model',model,'--dataset',ds,'--mode',mode,'--gpu',gpu]
        with log.open('x') as f:p=subprocess.Popen(args,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT)
        active[gpu]=p;attempts[gpu]=attempt
        write(HERE/f'runs/worker_gpu{gpu}.json',dict(pid=p.pid,model=model,dataset=ds,mode=mode,gpu=gpu,attempt=attempt,log=str(log),time=time.time()))
    for g in pending:launch(g)
    last=0;failures=[]
    while active:
        write(HERE/'runs/state.json',dict(status='running',model=model,dataset=ds,mode=mode,pid=os.getpid(),workers={g:p.pid for g,p in active.items()},time=time.time()))
        for gpu,p in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            del active[gpu]
            if rc==42 and attempts[gpu]<3:launch(gpu)
            elif rc!=0:failures.append(dict(gpu=gpu,exit_code=rc))
        if time.time()-last>180:
            subprocess.run([sys.executable,str(HERE/'summarize_stage.py')],check=True);last=time.time()
        if active:time.sleep(5)
    event=dict(model=model,dataset=ds,mode=mode,started_at=start,finished_at=time.time(),wall_seconds=time.time()-start,failures=failures)
    with (HERE/'runs/stage_history.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    assert not failures,event


def main():
    c=config();lock=(HERE/'runs/campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={s.split(',')[0].strip():int(s.split(',')[1]) for s in raw.splitlines()}
    assert all(used[g]<1000 for g in ['0','3']),'GPU0/3 busy; no duplicate launch'
    try:
        for model in ['omni','videollama']:
            remaining=any(c['plan'][ds+'/'+model+'/'+mode]['gpu'+gpu] for ds in ['avspeaker','joint'] for mode in ['fixed','agent'] for gpu in ['0','3'])
            if not remaining:continue
            stage(model,'joint','check')
            for gpu in ['0','3']:
                g=json.loads((HERE/f'runs/joint/{model}/check/gpu{gpu}/complete.json').read_text())
                assert g['equivalence_passed'] and g['n']==2
            for mode in ['fixed','agent']:
                for ds in ['avspeaker','joint']:stage(model,ds,mode)
        from summarize_stage import main as summarize
        assert summarize(),'Incomplete final output'
        write(HERE/'runs/state.json',dict(status='packing',time=time.time()))
        archive=HERE/'runs/complete_results.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for p in sorted((HERE/'runs').rglob('*')):
                if p.is_file() and p.suffix not in ['.gz','.lock']:tar.add(p,arcname=p.relative_to(HERE/'runs'))
            for p in HERE.glob('*.py'):tar.add(p,arcname='stage_source/'+p.name)
            tar.add(HERE/'README.txt',arcname='stage_source/README.txt')
            tar.add(V5/'manifests/code_freeze.json',arcname='native_v5_code_freeze.json')
        write(HERE/'runs/state.json',dict(status='complete',time=time.time(),archive_sha256=sha(archive)))
    except Exception as e:
        write(HERE/'runs/state.json',dict(status='needs_diagnosis',time=time.time(),error_type=type(e).__name__,error=str(e)));raise


if __name__=='__main__':main()
