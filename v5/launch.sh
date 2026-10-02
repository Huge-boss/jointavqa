set -euo pipefail
cd /data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/v5
MODE="${1:-pilot}"
/root/anaconda3/envs/Joint-avqa/bin/python - "$MODE" <<'PY'
import sys,subprocess,json,time
from pathlib import Path
p=Path.cwd();(p/'runs').mkdir(exist_ok=True);mode=sys.argv[1]
assert mode in ['pilot','full']
record=p/'runs'/f'{mode}_launch.json'
assert not record.exists(),'Already launched; inspect processes/state first'
if mode=='full':
 for ds in ['avspeaker','joint']:
  for m in ['omni','videollama']:
   assert json.loads((p/f'runs/{ds}/{m}/pilot/gate.json').read_text())['passed']
args=['/root/anaconda3/envs/Joint-avqa/bin/python',str(p/'run_campaign.py')]+(['--pilot-only'] if mode=='pilot' else [])
with (p/'runs'/f'{mode}_scheduler.log').open('a') as f:
 proc=subprocess.Popen(args,cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
data=dict(pid=proc.pid,mode=mode,time=time.time(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip())
record.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
PY
