from pathlib import Path
import ast, hashlib, json, re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=Path('/data/lzq/data/AV-SpeakerBench')
MODELS=Path('/data/lzq/code/project/acoustic-focs/model')
RESULTS=ROOT/'results/avspeaker_v1_20261002'
TASKS=['Speaker Detection','Speaker Recognition','Speaker Counting','Attribute Recognition','Activity Recogntion','Visual Counting','Speech Recognition','Speech Duration','Speech Pitch','Speech Rate','Speech Intensity','Speech Counting']
PAPER={'omni':[46.64,51.76,44.31,36.57,50.49,50.49,44.88,58.71,51.27,57.28,47.85,49.51,29.51],
       'videollama':[37.67,34.19,36.02,31.25,35.29,37.38,41.46,29.85,40.25,49.03,44.02,45.15,31.25]}
LEADERBOARD={'omni':[42.31,47.54,41.23,34.83,42.65,38.83,43.41,53.23,47.03,51.94,42.11,43.20,29.17], 'videollama':PAPER['videollama']}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def readlines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines()] if Path(p).exists() else []

def load_inputs():
 m=read(HERE/'input_manifest.json');p=HERE/'inputs.jsonl';assert sha(p)==m['inputs_sha256']
 rows=readlines(p);assert len(rows)==3212 and len({x['qid'] for x in rows})==3212
 assert all(not set(r)&{'answer','correct_answer','label'} for r in rows)
 return rows

def upstream_parse(text):
 # Frozen function body copied with AST from upstream, without importing its model runtime.
 # Preserve first matching letter behavior; do not use its random guess for None/failed inference.
 if text is None:return None
 source=HERE/'reference/upstream/model/__init__.py'
 tree=ast.parse(source.read_text(encoding='utf-8'))
 selected=[node for node in tree.body if (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='answer_prefixes' for t in node.targets)) or (isinstance(node,ast.FunctionDef) and node.name=='extract_characters_regex')]
 space={'re':re};exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),space)
 ans=space['extract_characters_regex'](text)
 return ans if ans in ['A','B','C','D'] else None

def strict_parse(text):
 if not isinstance(text,str):return None
 text=text.strip().replace('**','').strip('`').strip()
 bare=re.fullmatch(r'[\(\[]?([A-D])[\)\]]?[.!]?',text)
 if bare:return bare.group(1)
 declarations=re.findall(r'(?i)\b(?:the\s+)?(?:correct\s+|best\s+|final\s+)?(?:answer|option|choice)\s*(?:is\s*|[:：]\s*)[\(\[]?([A-D])[\)\]]?(?=[\s.,:;!]|$)',text)
 leading=re.match(r'^([A-D])(?:[.)：:]\s|[.)]$)',text)
 letters={x.upper() for x in declarations}
 if leading:letters.add(leading.group(1))
 if len(letters)!=1:return None
 # Reject explicitly conflicting letter alternatives even if one is declared.
 if re.search(r'\b[A-D]\s*(?:or|and|/)\s*[A-D]\b',text):return None
 return next(iter(letters))

def code_hashes():
 return {p.name:sha(p) for p in sorted(HERE.glob('*.py')) if p.name not in ['download_data.py']}
