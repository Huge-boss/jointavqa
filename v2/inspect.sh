set -euo pipefail
cd /data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/v2
/root/anaconda3/envs/Joint-avqa/bin/python - <<'PY'
import json,pathlib,time,os
p=pathlib.Path.cwd()
print('UTC',time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()))
for f in sorted((p/'runs').glob('*state.json')):
 print(f.name,json.loads(f.read_text()))
for f in sorted((p/'runs').glob('*worker.json')):
 r=json.loads(f.read_text());r['alive']=pathlib.Path('/proc',str(r['pid'])).exists();print(f.name,r)
for f in sorted((p/'runs').glob('*/*/*/progress.json')):
 print(str(f.relative_to(p)),json.loads(f.read_text()))
for f in sorted((p/'runs').glob('*/*/pilot/gate.json')):
 print(str(f.relative_to(p)),json.loads(f.read_text()))
print('recent campaign log:',(p/'runs/campaign.log').read_text()[-1500:] if (p/'runs/campaign.log').exists() else 'absent')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
