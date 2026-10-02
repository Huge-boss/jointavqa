"""Filesystem helpers only; importing this never reads labels or model outputs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = {'joint': Path('/data/lzq/data/JointAVBench'), 'avspeaker': Path('/data/lzq/data/AV-SpeakerBench')}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    temp.replace(p)


def readlines(p):
    p = Path(p)
    if not p.exists(): return []
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines(keepends=True) if s.endswith('\n') and s.strip()]


def code_hashes():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}
