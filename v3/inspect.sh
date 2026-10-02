set -euo pipefail
cd /data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/v3
date -u '+UTC %Y-%m-%d %H:%M:%S'
python - <<'PY'
from pathlib import Path
import json,os
for p in sorted(Path('runs').glob('*state.json')):print(str(p),json.loads(p.read_text()))
for p in sorted(Path('runs').glob('*worker.json')):
 d=json.loads(p.read_text());d['alive']=Path(f"/proc/{d['pid']}").exists();print(str(p),d)
for p in sorted(Path('runs').glob('*/*/*/progress.json')):print(str(p),json.loads(p.read_text()))
for p in sorted(Path('runs').glob('*/*/pilot/gate.json')):print(str(p),{'passed':json.loads(p.read_text())['passed']})
for p in sorted(Path('runs').glob('*scheduler.log')):
 text=p.read_text(errors='replace');print(str(p),text[-1600:])
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
