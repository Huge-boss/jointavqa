"""Independent, label-blind inputs/parser; ground truth is loaded only by scoring."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DATA = Path('/data/lzq/data/JointAVBench')
MODELS = Path('/data/lzq/code/project/acoustic-focs/model')
RESULTS = ROOT/'results/custom_v1_20261002'
TASKS = ['STL','SPL','SOOG','SOER','SPER','MPTI','VSSR','CSA','MPO','PTG','AFA','PDP','AVDM','MESI','CRI']
ALIASES = dict(zip(['task'+str(n) for n in [1,2,4,5,6,7,8,9,10,11,12,13,15,16,17]], TASKS))
MODEL_NAMES = {'omni':'Qwen2.5-Omni-7B','vl':'Qwen2.5-VL-7B-Instruct','internvl':'InternVL2_5-8B','videollama':'VideoLLaMA2.1-7B-AV'}
PAPER = {
 'omni': [56.5,67.8,35.3,59.5,73.5,35.2,65.6,76.3,48.8,40.4,21.5,68.2,47.3,49.1,71.4,67.3],
 'vl': [47.7,33.0,38.4,55.3,61.0,27.9,58.0,47.6,29.2,41.3,32.2,60.8,36.6,40.7,65.2,61.6],
 'internvl': [51.7,26.1,40.2,60.1,71.1,31.9,62.7,52.2,40.8,44.2,27.5,59.7,40.9,50.0,67.7,67.1],
 'videollama': [46.8,20.0,42.4,53.0,67.0,33.9,50.0,47.6,24.0,33.3,30.2,58.5,35.5,39.8,63.2,59.4],
}

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2), b''):h.update(b)
 return h.hexdigest()

def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def read_jsonl(p):
 return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()] if Path(p).exists() else []

def inputs():
 p=ROOT/'reproduction/frozen_seed42/inputs.jsonl'
 assert digest(p)=='3af376e083b92501154aa9aed9d5be1466ce9a7b6e12f498a04e9cd60b7cee50'
 rows=read_jsonl(p)
 assert len(rows)==2853 and len({r['qid'] for r in rows})==2853
 for r in rows:
  assert not any(k in r for k in ['correct_answer','correct_prefix','answer_label'])
  assert r['task'] in TASKS and set(r['prefix2text'])==set('ABCD')
  assert len(set(r['prefix2text'].values()))==4
  p=(DATA/r['clip_path']).resolve()
  assert p.is_relative_to(DATA) and p.is_file()
 return rows

def pilot_inputs(rows):
 # Selection depends on task membership and input size only, never correctness.
 qids={next(r['qid'] for r in rows if r['task']==t) for t in TASKS}
 # Previously audited longest physical clip, selected from measured duration metadata.
 selection=json.loads((HERE/'pilot_selection.json').read_text())
 qids.add(selection['longest_qid'])
 return [r for r in rows if r['qid'] in qids]

def parse_answer(text, options):
 """Conservative explicit answer, else exact full option text; never use a label."""
 text=str(text).strip()
 # Formatting wrappers are presentation, not answer content.
 text=re.sub(r'^```(?:text)?\s*|\s*```$', '', text).strip().replace('**','')
 declared=re.findall(r'(?im)^\s*(?:the\s+)?(?:correct\s+|final\s+)?(?:answer|option|choice)\s*(?:is\s*|[:：]\s*)\(?([A-D])\)?(?=\s|[.,:;!)]|$)',text)
 leading=re.match(r'^\s*\(?([A-D])\)?(?=\s|[.,:;!)]|$)',text)
 letters=set(declared+([leading.group(1)] if leading else []))
 if len(letters)>1:return None
 if leading:
  tail=text[leading.end():]
  if re.match(r'\s*(?:/|or\b|and\b|,)\s*\(?[A-D]\)?(?:\W|$)',tail):return None
 if letters:return next(iter(letters))
 normalized=lambda s: re.sub(r'\s+',' ',s.strip().strip('"\'').rstrip('.')).casefold()
 matches=[k for k,v in options.items() if normalized(v)==normalized(text)]
 return matches[0] if len(matches)==1 else None

def labels(rows):
 p=DATA/'jointavbench.json'
 assert digest(p)=='818719a36308a1158061b92e575f01ad35ac283c1c518b7e161364af0f237d1c'
 truth={r['qid']:r for r in json.loads(p.read_text())}
 assert len(truth)==len(rows)==2853
 answer={}
 for r in rows:
  t=truth[r['qid']]
  assert ALIASES.get(t['task'],t['task'])==r['task']
  assert set(t['options'])==set(r['prefix2text'].values())
  assert len(t['options'])==4 and t['correct_answer'] in t['options']
  if t.get('answer_label'):assert t['options'][ord(t['answer_label'].strip().upper())-65]==t['correct_answer']
  answer[r['qid']]=next(k for k,v in r['prefix2text'].items() if v==t['correct_answer'])
 return answer

def latest_records(path):
 latest={}
 for r in read_jsonl(path):
  old=latest.get(r['qid'])
  if old and old['status']=='ok':raise ValueError('Duplicate successful qid: '+r['qid'])
  latest[r['qid']]=r
 return latest
