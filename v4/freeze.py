"""Freeze new code and inherit previously verified native dependency hashes."""
import json
from support import HERE,sha,write,code_hashes

if __name__=='__main__':
    dest=HERE/'manifests/code_freeze.json'
    assert not dest.exists(),'Never overwrite a frozen version'
    old=json.loads((HERE.parent/'v2/manifests/code_freeze.json').read_text())
    write(dest,dict(python_sha256=code_hashes(),external_sha256=old['external_sha256'],
                   protocol_sha256=sha(HERE/'protocol.json'),
                   data_sha256={d:sha(HERE/'data'/f'{d}.jsonl') for d in ['joint','avspeaker']},
                   manifest_sha256={p.name:sha(p) for p in sorted((HERE/'manifests').glob('*.json'))},
                   basis='Same frozen20% subset, no labels used to select protocol, fixed checkpoint-native preprocessing'))
