set -euo pipefail
cd /data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/stage_v1
/root/anaconda3/envs/Joint-avqa/bin/python - <<'PY'
import subprocess,json,time
from pathlib import Path
p=Path.cwd();record=p/'runs/launch.json'
assert not record.exists(),'Already launched; inspect before continuing'
with (p/'runs/scheduler.log').open('x') as f:
 proc=subprocess.Popen(['/root/anaconda3/envs/Joint-avqa/bin/python',str(p/'run_stages.py')],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,time=time.time(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip())
record.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
PY
