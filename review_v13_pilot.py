"""Read-only pilot audit followed by evidence archive; never reads labels or scores."""
import json,pathlib,hashlib,collections,re,tarfile,time,sys
BASE=pathlib.Path(__file__).resolve().parent
P=BASE/'memory_v13'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 state=json.loads((P/'runs/state.json').read_text());assert state['status']=='awaiting_label_blind_pilot_review',state
 manifest=json.loads((P/'runs/manifest.json').read_text())
 assert all(sha(P/f)==h for f,h in manifest['code'].items())
 assert sha(P/'protocol.json')==manifest['protocol']
 out=dict(passed=False,manifest_sha256=sha(P/'runs/manifest.json'),created_at=time.time(),groups={},limitations=[
  'Technical review only, no labels or accuracy loaded.',
  'Factual correctness, speaker binding and frame-event correspondence not certified.',
  'Constrained candidate syntax differs from V11, retained as a disclosed method change.',
  'NONE/GLOBAL abstention is permitted; not forced into arbitrary replay.',
  'A valid frame index is not proof of event localization.'])
 allrows=[]
 for ds in ['joint','avspeaker']:
  for m in ['omni','videollama']:
   rows=[json.loads(x) for f in (P/f'runs/{ds}/{m}/pilot').glob('gpu*/predictions.jsonl') for x in f.read_text().splitlines()]
   assert len(rows)==6 and {r['qid'] for r in rows}==set(manifest['pilot_qids'][ds])
   for r in rows:
    assert r['status']=='ok' and all(r['native_parity'].values()) and all(r['observation_repeat'].values())
    a=r['arm'];c=a['calls'][0];fs=a['scout_fields'];frames=c['ledger_media']['frames']
    assert a['scout_schema_valid'] and fs['ANCHOR'] in ['NONE']+[f['index'] for f in frames]
    assert len(frames)==c['frames'] and 1<=len(a['calls'])<=3
    assert sum(c['generated_tokens'] for c in a['calls'])<=240
    assert not a['calls'][-1]['reached_token_limit']
    for call in a['calls']:
     assert call['audio_tokens']>0 and call['video_tokens']>0
     if call['stage']=='disagreement_verification':
      context=json.dumps(call['serialized_prompt']);assert 'Available input-frame addresses:' not in context
      assert 'Observation proposal_ledger' not in context
    if a['review']:
     assert a['window_source'].startswith('actual_input_frame_') and a['review_window'][1]-a['review_window'][0]<=12.001
   descriptive=sum(bool(re.search('[A-Za-z]{2,}',r['arm']['scout_fields']['EVIDENCE'])) for r in rows)
   assert descriptive>=4,'Numeric/non-descriptive interface failure, not an accuracy decision'
   out['groups'][ds+'/'+m]=dict(n=6,descriptive_evidence=descriptive,decisions=dict(collections.Counter(r['arm']['decision'] for r in rows)),
    reviews=sum(r['arm']['review'] for r in rows),peak_gpu_gib=max(c['peak_gpu_gib'] for r in rows for c in r['arm']['calls']),
    partial_fields=sum(any(r['arm']['scout_fields']['field_token_caps'].values()) for r in rows),
    evidence=[dict(qid=r['qid'],ledger=r['arm']['scout_fields'],window=r['arm']['review_window']) for r in rows])
   allrows+=rows
 out['pilot_n']=len(allrows);out['new_method_work_seconds']=sum(r['arm']['new_work_seconds'] for r in allrows)
 out['pilot_replay_paths_exercised']=sum(r['arm']['review'] for r in allrows)
 probes=[json.loads((P/f'runs/path_probe/{ds}_{m}.json').read_text()) for ds in ['joint','avspeaker'] for m in ['omni','videollama']]
 assert all(r['passed'] and not r['verify']['reached_token_limit'] for r in probes)
 out['forced_path_checks']=dict(n=4,work_seconds=sum(r['work_seconds'] for r in probes),load_seconds=sum(r['load_seconds'] for r in probes),local_truncations=sum(r['local']['reached_token_limit'] for r in probes),note='Separate implementation-path checks, not scored method cases; synthetic anchor never enters perception or verification prompt.')
 out['audit_tools']={n:sha(BASE/n) for n in ['review_v13_pilot.py','probe_v13_replay_path.py']}
 out['passed']=True
 (P/'runs/pilot_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 archive=P/'runs/pilot_evidence.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for folder in ['runs','data']:
   for f in sorted((P/folder).rglob('*')):
    if f.is_file() and f.suffix not in ['.gz','.lock']:tar.add(f,arcname=str(f.relative_to(P)))
  for n in out['audit_tools']:tar.add(BASE/n,arcname='audit_tools/'+n)
  for f in sorted(P.iterdir()):
   if f.is_file():tar.add(f,arcname='source/'+f.name)
 print(json.dumps(dict(pilot_n=len(allrows),passed=True,archive_sha256=sha(archive),replays=out['pilot_replay_paths_exercised'],groups={k:{a:b for a,b in v.items() if a!='evidence'} for k,v in out['groups'].items()})))

if __name__=='__main__':main()
