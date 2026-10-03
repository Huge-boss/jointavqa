"""Archive the fixed, failed OCR gate without scoring QA or mutating frozen code."""
import json,sys,time,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;HERE=BASE/'memory_v14'
sys.path.insert(0,str(HERE))
from settings import frozen,sha,write

def main():
    c=frozen();manifest_sha=sha(HERE/'runs/manifest.json');reports=[]
    for ds in ['joint','avspeaker']:
        for model in ['omni','videollama']:
            r=json.loads((HERE/f'runs/address_probe/{ds}_{model}.json').read_text())
            assert r['manifest_sha256']==manifest_sha and r['qid']==c['pilot_qids'][ds][0]
            assert r['qa_labels_loaded'] is False and r['qa_score'] is None
            reports.append(r)
    assert sum(r['status']=='complete' for r in reports)==3
    assert not any(r['passed'] for r in reports)
    assert not (HERE/'runs/pilot_release.json').exists()
    assert not list((HERE/'runs').glob('*/*/dev20'))
    assert not list((HERE/'runs').glob('*/*/pilot'))
    rows=[]
    for r in reports:
        out=r.get('output',{});marker=r.get('marker',{})
        rows.append(dict(dataset=r['dataset'],model=r['model'],qid=r['qid'],status=r['status'],
            expected=r.get('expected_code'),observed=r.get('observed_code'),passed=r['passed'],
            token_cap=out.get('reached_token_limit'),generated_tokens=out.get('generated_tokens'),
            load_seconds=r['load_seconds'],nonload_work_seconds=r['nonload_work_seconds'],
            total_seconds=r['total_seconds'],error=r.get('error'),
            occluded_fraction=marker.get('occluded_fraction')))
    audit=dict(status='technical_failed_not_released',manifest_sha256=manifest_sha,
        probe_attempts=4,completed_readouts=3,pre_generation_instrumentation_errors=1,strict_gate_passed=0,
        model_calls=3,qa_scored_questions=0,formal_questions=0,pilot24_started=False,rows=rows,
        load_seconds_sum=sum(r['load_seconds'] for r in reports),
        nonload_work_seconds_sum=sum(r['nonload_work_seconds'] for r in reports),
        total_worker_seconds_sum=sum(r['total_seconds'] for r in reports),
        first_start_to_last_finish_seconds=max(r['finished_at'] for r in reports)-min(r['started_at'] for r in reports),
        limitation='Fixed artificial readout fixtures, not natural localization/QA. Omni correct code prefixes with extra text; Joint Video wrong readout; AV Video unassessed due BF16 NumPy audit conversion error. No waiver, retuning or repeated inference.',
        next_decision='Do not release or restart V14. Preserve failure; summarize the address-readability and evidence-acquisition bottleneck before proposing any new method.',
        frozen_code_dependencies_assets_verified=True)
    outdir=HERE/'runs/technical_audit';outdir.mkdir(exist_ok=True)
    write(outdir/'AUDIT.json',audit)
    write(HERE/'runs/address_probe_gate.json',dict(passed=False,manifest_sha256=manifest_sha,reasons=audit['limitation']))
    write(HERE/'runs/state.json',dict(status=audit['status'],time=time.time(),manifest_sha256=manifest_sha))
    files=[p for p in HERE.rglob('*') if p.is_file() and 'cache' not in p.relative_to(HERE).parts and '__pycache__' not in p.parts and p.suffix!='.gz']
    hashes={p.relative_to(HERE).as_posix():sha(p) for p in files}
    write(outdir/'SHA256.json',hashes)
    archive=HERE/'runs/technical_failure_evidence.tar.gz'
    assert not archive.exists(),'Preserve prior archive'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files+[outdir/'SHA256.json']):tar.add(p,arcname=p.relative_to(HERE).as_posix())
    receipt=dict(archive_sha256=sha(archive),verified_files=len(files),status=audit['status'])
    write(HERE/'runs/technical_failure_receipt.json',receipt)
    print(json.dumps(dict(receipt=receipt,audit=audit),ensure_ascii=False))

if __name__=='__main__':main()
