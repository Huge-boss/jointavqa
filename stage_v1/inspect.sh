set -euo pipefail
cd /data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/stage_v1
/root/anaconda3/envs/Joint-avqa/bin/python - <<'PY'
import json,os,time
from pathlib import Path
p=Path('runs');print('UTC',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
for f in [p/'state.json',*p.glob('worker_gpu*.json')]:
 if f.exists():
  d=json.loads(f.read_text())
  if 'pid' in d:d['alive']=os.path.exists('/proc/'+str(d['pid']))
  print(str(f),d)
for f in sorted(p.glob('*/*/*/gpu*/progress.json')):print(str(f),json.loads(f.read_text()))
if (p/'reports/TOTAL_TABLE.json').exists():
 for r in json.loads((p/'reports/TOTAL_TABLE.json').read_text()):print(r['dataset'],r['model'],r['method'],r['observed_n'],r['planned_n'],r['complete'])
for f in sorted((p/'logs').glob('*.log')):
 lines=f.read_text(errors='replace').splitlines()
 print(f.name,'\n'.join(lines[-2:]))
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
