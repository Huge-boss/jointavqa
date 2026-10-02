"""Archive only an already-complete stage; no inference or rescoring."""
import argparse,json,pathlib,hashlib,tarfile,time
BASE=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--version',choices=['v13'],required=True);a.add_argument('--dataset',choices=['joint','avspeaker'],required=True);a.add_argument('--model',choices=['omni','videollama'],required=True);args=a.parse_args()
 p=BASE/('memory_'+args.version);runs=p/'runs';ds=args.dataset;m=args.model;key=ds+'/'+m
 manifest=json.loads((runs/'manifest.json').read_text());assert all(sha(p/f)==h for f,h in manifest['code'].items())
 table=json.loads((runs/'reports/TOTAL_TABLE.json').read_text());r=next(r for r in table if r['dataset']==ds and r['model']==m);assert r['complete']
 for g in ['0','1','3']:assert json.loads((runs/f'{ds}/{m}/dev20/gpu{g}/complete.json').read_text())['complete']
 pairs=json.loads((runs/f'reports/{ds}_{m}_paired_questions.json').read_text())
 raw=[json.loads(x) for f in (runs/f'{ds}/{m}/dev20').glob('gpu*/predictions.jsonl') for x in f.read_text().splitlines()]
 ok=[x for x in raw if x['status']=='ok'];assert len(ok)==len({x['qid'] for x in ok})==r['planned']
 assert {x['qid'] for x in ok}=={x['qid'] for x in pairs}
 details=json.loads((runs/'reports/TASK_DETAILS.json').read_text())[key]
 assert sum(t['denominator'] for t in details)==r['planned'] and sum(t['correct'] for t in details)==r['correct']
 pair=json.loads((runs/'reports/PAIRED_SUMMARY.json').read_text())[key]
 dest=runs/f'stage_snapshots/{ds}_{m}_completed';dest.mkdir(parents=True,exist_ok=True)
 archive=dest/'complete_evidence.tar.gz'
 if archive.exists():
  receipt=json.loads((dest/'receipt.json').read_text());assert sha(archive)==receipt['archive_sha256'];print(json.dumps(receipt));return
 summary=dict(table=r,paired=pair,tasks=details,manifest_sha256=sha(runs/'manifest.json'),created_at=time.time(),note='Completed single stage only; not four-group method verdict.')
 (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (dest/'paired_questions.json').write_text(json.dumps(pairs,ensure_ascii=False)+'\n',encoding='utf-8')
 files=[*sorted((runs/ds/m).rglob('*')),*sorted((p/'data').glob(ds+'*')),*sorted(p.glob('*'))]
 files += [runs/n for n in ['manifest.json','pilot_release.json','pilot_audit.json','stages.jsonl']]
 files += [dest/'summary.json',dest/'paired_questions.json']
 with tarfile.open(archive,'w:gz') as tar:
  for f in files:
   if f.is_file():tar.add(f,arcname=str(f.relative_to(p)))
 receipt=dict(dataset=ds,model=m,planned=r['planned'],archive_sha256=sha(archive),manifest_sha256=summary['manifest_sha256'])
 (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
