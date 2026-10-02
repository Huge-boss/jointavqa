set -euo pipefail
cd /data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/v2
test ! -f runs/campaign_launch.json
/root/anaconda3/envs/Joint-avqa/bin/python - <<'PY'
import json, pathlib, subprocess, time
p=pathlib.Path.cwd(); (p/'runs').mkdir(exist_ok=True)
f=(p/'runs/campaign.log').open('a')
proc=subprocess.Popen(['/root/anaconda3/envs/Joint-avqa/bin/python',str(p/'run_campaign.py')],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':proc.pid,'started_at':time.time(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),'scope':'Two models, two fixed20% datasets, agent and fixed control; no automatic full run'}
(p/'runs/campaign_launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
