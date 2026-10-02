"""Freeze repaired version before launch; reuse exact original20% and pilot."""
import json
from runtime import HERE, ROOT, sha
from worker import code_hashes

assert not list((HERE/'runs').rglob('predictions.jsonl')), 'Never refreeze a started run'
external = [p for p in (HERE.parent/'compat').rglob('*.py')]
external += [p for p in (ROOT/'reproduction/custom_v1/vendor/VideoLLaMA2-official').rglob('*.py') if '.git' not in p.parts]
obj = {'python_sha256':code_hashes(), 'external_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(external)}, 'protocol_sha256':sha(HERE/'protocol.json'), 'parent_version':'v1 preserved; tool formatting repair only'}
(HERE/'manifests/code_freeze.json').write_text(json.dumps(obj,indent=2)+'\n')
print('Frozen v2', len(obj['python_sha256']), 'Python files', len(external), 'external dependencies')
