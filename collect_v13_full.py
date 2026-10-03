"""Collect this single staged campaign after completion; no GPU mutations."""
import hashlib,json,subprocess,sys,time
from pathlib import Path

BASE=Path(__file__).resolve().parent
LOCAL=BASE/'evaluation_v13_full/runs'
REMOTE='/data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/evaluation_v13_full/runs'
SSH=BASE.parents[1]/'ssh_remote.py'
COMMAND="""/root/anaconda3/envs/Joint-avqa/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/evaluation_v13_full/runs')
print(json.dumps(dict(state=json.loads((p/'state.json').read_text()),progress={str(f.relative_to(p)):json.loads(f.read_text()) for f in p.glob('*/*/*/gpu*/progress.json')})))
PY"""


def ssh(*args):return subprocess.run([sys.executable,str(SSH),*args],capture_output=True,text=True,encoding='utf-8',timeout=240,check=True).stdout


def save(value):
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'collector_state.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    deadline=time.monotonic()+24*3600;failures=0
    while time.monotonic()<deadline:
        try:
            info=json.loads(ssh('--command',COMMAND));failures=0
            save(dict(status='checking',time=time.time(),remote=info))
            if info['state']['status']=='needs_diagnosis':save(dict(status='needs_diagnosis',time=time.time(),remote=info));return
            if info['state']['status']=='complete':
                ssh('--get',REMOTE+'/complete_results.tar.gz',str(LOCAL/'complete_results.tar.gz'))
                digest=hashlib.sha256((LOCAL/'complete_results.tar.gz').read_bytes()).hexdigest()
                assert digest==info['state']['archive_sha256']
                for name in ['state.json','launch.json','manifest.json','preflight.json','stages.jsonl','reports/TOTAL_TABLE.json','reports/PAIRED_SUMMARY.json','reports/TASK_DETAILS.json','reports/REPORT_ZH.txt','reports/AGGREGATE.json']:
                    (LOCAL/name).parent.mkdir(parents=True,exist_ok=True)
                    ssh('--get',REMOTE+'/'+name,str(LOCAL/name))
                table=json.loads((LOCAL/'reports/TOTAL_TABLE.json').read_text(encoding='utf-8'))
                assert len(table)==12 and all(r['complete'] for r in table)
                save(dict(status='complete',time=time.time(),archive_sha256=digest,report='reports/REPORT_ZH.txt'));return
        except Exception as exc:
            failures+=1;save(dict(status='collection_retry',time=time.time(),error_type=type(exc).__name__,consecutive_failures=failures))
            if failures>=12:raise
        time.sleep(180)
    save(dict(status='collection_timeout',time=time.time()))


if __name__=='__main__':main()
