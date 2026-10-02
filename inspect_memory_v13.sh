#!/usr/bin/env bash
set -eu
cd /data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1
/root/anaconda3/envs/Joint-avqa/bin/python - <<'PY'
import pathlib,json,hashlib
p=pathlib.Path('memory_v13/runs')
for n in ['state.json','launch.json','pilot_release.json']:
 f=p/n
 if f.exists():print(n,f.read_text())
for f in sorted(p.glob('*/*/dev20/gpu*/progress.json')):print(str(f.relative_to(p)),f.read_text())
for f in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:a=f.read_bytes().decode().split('\0')
 except (OSError,UnicodeError):continue
 if len(a)>1 and 'python' in a[0] and ('memory_v13/' in a[1] or a[1]=='probe_v13_replay_path.py'):print('process',f.parent.name,a[:12])
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
