"""Scheduling-only continuation of frozen v5 inference."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
V5=HERE.parent/'v5'
ROOT=HERE.parents[2]
sys.path.insert(0,str(V5))
from support import sha,readlines,write


def code_hashes():return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}


def split_missing(ids,completed):
    assert len(ids)==len(set(ids)) and set(completed)<=set(ids)
    missing=[q for q in ids if q not in set(completed)]
    return missing[::2],missing[1::2]


def config():
    c=json.loads((HERE/'runs/manifest.json').read_text())
    assert c['stage_code_sha256']==code_hashes(),'Stage source changed'
    f=json.loads((V5/'manifests/code_freeze.json').read_text())
    assert c['native_freeze_sha256']==sha(V5/'manifests/code_freeze.json')
    assert f['python_sha256']=={p.name:sha(p) for p in sorted(V5.glob('*.py'))}
    assert f['protocol_sha256']==sha(V5/'protocol.json')
    for rel,h in f['external_sha256'].items():assert sha(ROOT/rel)==h,rel
    for ds,h in c['input_sha256'].items():assert sha(V5/'data'/f'{ds}.jsonl')==h
    return c


def source_records(ds,model):
    paths=([ROOT/f'results/avspeaker_v1_20261002/{model}/full/predictions.jsonl'] if ds=='avspeaker'
           else [ROOT/f'results/custom_v1_20261002/{model}/full/gpu{g}/predictions.jsonl' for g in [0,3]])
    result={}
    for p in paths:
        h=sha(p)
        for line,r in enumerate(readlines(p),1):
            if r['status']=='ok':
                assert r['qid'] not in result
                result[r['qid']]=(r,dict(path=str(p.relative_to(ROOT)),sha256=h,line=line,kind='historical_baseline'))
    return result


def jsonlines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def arm_records(ds,model,mode):
    dest=HERE/f'runs/{ds}/{model}/{mode}'
    result={};errors=[]
    for p in [dest/'inherited.jsonl']+sorted(dest.glob('gpu*/predictions.jsonl')):
        for r in readlines(p):
            if r['status']=='ok':
                assert r['qid'] not in result,('duplicate successful arm',p,r['qid'])
                result[r['qid']]=r
            else:errors.append(r)
    return result,errors
