"""Label-free paths and immutable experiment manifest."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
V5 = HERE.parent / 'v5'
ROOT = HERE.parents[2]
sys.path.insert(0, str(V5))
from support import sha, readlines, write
GPUS = ['0', '1', '3']
ENVS = {'omni': '/root/anaconda3/envs/Joint-avqa/bin/python', 'videollama': '/root/anaconda3/envs/Joint-avqa-video/bin/python'}

def hashes():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}

def frozen():
    c = json.loads((HERE/'runs/manifest.json').read_text())
    assert c['code'] == hashes(), 'Source changed: create a separate version'
    assert c['protocol'] == sha(HERE/'protocol.json')
    for rel, digest in c['dependencies'].items():
        assert sha(ROOT/rel) == digest, rel
    for rel, digest in c['assets'].items():
        assert sha(HERE/rel) == digest, rel
    for rel,digest in c['sources'].items():
        assert sha(ROOT/rel)==digest,rel
    return c

def jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
