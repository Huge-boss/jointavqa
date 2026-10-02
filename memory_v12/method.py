"""Event-index ledger and isolated verification; never loads labels."""
import math,re,time,json,subprocess
from settings import HERE,sha
from runtime import Runtime as NativeRuntime
import sys
sys.path.insert(0,str(HERE.parent/"stage_roundoff_v2"))
from roundoff_fix import install
install()
import structured
from ledger_grammar import LedgerGrammar

def source_coverage(meta,duration,offset=0.):
    spans=[]
    for a,b in coverage(meta,duration):
        a=max(0.,min(duration,a));b=max(a,min(duration,b))
        if b>a:spans.append([a+offset,b+offset])
    union=[]
    for a,b in sorted(spans):
        if union and a<=union[-1][1]:union[-1][1]=max(union[-1][1],b)
        else:union.append([a,b])
    supplied=sum(b-a for a,b in spans);unique=sum(b-a for a,b in union)
    return dict(source_intervals=spans,source_union=union,unique_seconds=unique,
                repeated_seconds=max(0.,supplied-unique),source_offset=offset,
                note='Source seconds, not concatenated or zero-padded encoder time. Sampling overlap is not proof of repeated words.')

def media_ledger(meta,path,duration):
    vm=meta.get('video_metadata',{})
    indices=meta.get('frame_indices',vm.get('frames_indices',vm.get('frame_indices')))
    fps=meta.get('source_fps',vm.get('fps'))
    if fps is None:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=avg_frame_rate','-of','json',str(path)],timeout=60))
        a,b=info['streams'][0]['avg_frame_rate'].split('/');fps=float(a)/float(b)
    fps=float(fps)
    assert math.isfinite(fps) and fps>0 and indices is not None
    indices=[int(x) for x in indices]
    assert len(indices)==meta['frames'] and 0<len(indices)<=128
    frames=[dict(index=f'F{i:03d}',source_frame=x,source_seconds=x/fps) for i,x in enumerate(indices)]
    assert all(0<=f['source_seconds']<=duration+.25 for f in frames)
    return dict(frames=frames,source_fps=fps,audio=source_coverage(meta,duration),
                address_warning='Frame address validity does not establish event localization or factual accuracy.')

def coverage_text(value):
    fmt=lambda spans:', '.join(f'{a:.3f}-{b:.3f}' for a,b in spans)
    return ('Actual source-audio intervals (seconds): '+fmt(value['source_intervals'])+
        '. Their union: '+fmt(value['source_union'])+
        f". Unique coverage {value['unique_seconds']:.3f}s; overlapping coverage {value['repeated_seconds']:.3f}s. "
        'Overlapping source intervals and silence padding are not additional events or speakers. '
        'Unheard intervals remain unknown; do not invent their content.')

def scout_prompt(row,duration,policy):
    # The runtime replaces this placeholder only after real media preparation.
    return row['question_prompt']

def ledger_prompt(row,ledger):
    frame_text=', '.join(f"{f['index']}={f['source_seconds']:.3f}s" for f in ledger['frames'])
    return ('Answer the question using the supplied video and audio. First output one provisional option letter A, B, C, or D. '
        'Then EVIDENCE: one short statement of directly perceived question-relevant evidence, or UNKNOWN. '
        'ANCHOR: select one listed input-frame address identifying the event the question refers to, or NONE if it cannot be located. '
        'RELATION: BEFORE, AROUND, or AFTER indicates where closer inspection relative to that event would help; '
        'GLOBAL for a whole-clip judgment or no justified local inspection. '
        'Do not fabricate addresses or claim that an address proves a fact. '
        'The index table describes the actual supplied frames, not dense video coverage.\n'
        'Input-frame source-time table: '+frame_text+'\n'+coverage_text(ledger['audio'])+'\n'+row['question_prompt'])

def interval_for(fs,ledger,duration):
    if duration<=12.001:return None,'short_clip_global_verification'
    evidence=fs.get('EVIDENCE','').strip()
    if not re.search(r'[A-Za-z]{2,}',evidence) or evidence.upper().startswith(('UNKNOWN','NONE')):
        return None,'no_descriptive_event_evidence'
    if fs.get('field_token_caps',{}).get('EVIDENCE'):return None,'partial_event_evidence'
    frames={f['index']:f['source_seconds'] for f in ledger['frames']}
    anchor=fs.get('ANCHOR');relation=fs.get('RELATION')
    if anchor not in frames or relation=='GLOBAL':return None,'no_justified_local_anchor'
    assert relation in ['BEFORE','AROUND','AFTER']
    anchor_time=frames[anchor]
    start=anchor_time+{'BEFORE':-12.,'AROUND':-6.,'AFTER':0.}[relation]
    start=max(0.,min(start,duration-12.))
    return [round(start,6),round(start+12.,6)],'actual_input_frame_'+anchor+'_'+relation.lower()

class Runtime(NativeRuntime):
    def call(self,row,prompt,interval,stage,limit):
        self.active_stage=stage;self.active_window=interval
        return super().call(row,prompt,interval,stage,limit)

    def generate(self,row,path,prompt,prepared,limit,structured=False):
        ledger=None;actual_coverage=None
        if structured:
            ledger=media_ledger(prepared['meta'],path,self.active_duration)
            structured_module=globals()['structured']
            structured_module.MemoryGrammar=lambda tok,eos:LedgerGrammar(tok,eos,len(ledger['frames']))
            prompt=ledger_prompt(row,ledger)
        elif self.active_stage in ['local_observation','disagreement_verification']:
            offset=self.active_window[0] if self.active_window else 0.
            duration=(self.active_window[1]-offset) if self.active_window else self.active_duration
            actual_coverage=source_coverage(prepared['meta'],duration,offset)
            prompt=coverage_text(actual_coverage)+'\n'+prompt
        out=super().generate(row,path,prompt,prepared,limit,structured=structured)
        if ledger:out['ledger_media']=ledger
        if actual_coverage:out['actual_source_audio']=actual_coverage
        return out

def question_only(row):
    lines = []
    for s in row['question_prompt'].splitlines():
        if re.match(r'^\s*[A-D][.\):]', s): continue
        if s.startswith(('Watch the video', 'Select the best answer', 'Respond with', 'The best answer')): continue
        lines.append(re.sub(r'^Question:\s*', '', s))
    return '\n'.join(lines).strip()


def route_answer(text,row):
    # Frozen label-blind parser also recognizes an explicit answer followed by explanation.
    # This is for control only; independent scoring remains unchanged.
    from joint_parser import parse
    return parse(text,row.get('prefix2text',row.get('options',{})))['answer']



def coverage(call, duration):
    if 'audio_snippet_starts_seconds' in call:
        return [[float(t), min(duration, float(t)+2)] for t in call['audio_snippet_starts_seconds']]
    if call.get('native_audio_clip_timepoints'):
        return [[float(a), float(b)] for a,b in call['native_audio_clip_timepoints']]
    seconds = call.get('audio_seconds_encoded', call.get('audio_seconds_supplied'))
    return [[0., min(duration, float(seconds))]] if seconds is not None else []


def memory_text(memories):
    parts = []
    for m in memories:
        parts.append(f"Observation {m['id']}, source interval {m['interval']}, audio coverage {m['audio_coverage']}, "
                     f"visual frames {m['frames']}, generated note (fallible, not an annotation):\n{m['text']}")
    return '\n\n'.join(parts)


def note(call, source_sha, duration, identifier):
    offset = call['window'][0] if call['window'] else 0
    local_duration = call['window'][1]-offset if call['window'] else duration
    audio = [[offset+a, offset+b] for a,b in coverage(call,local_duration)]
    text = call['model_output']
    rejected=[]
    if call.get('memory_fields'):
        pieces=[]
        for key in ['VISUAL','AUDIO','LINK']:
            value=call['memory_fields'][key]
            if not re.search(r'[A-Za-z]{2,}',value):
                rejected.append(key);value='UNKNOWN (tool returned no descriptive observation)'
            capped=call['memory_fields'].get('field_token_caps',{}).get(key,False)
            pieces.append(key+': '+value+(' [PARTIAL: token cap reached]' if capped else ''))
        text='\n'.join(pieces)
    # Retrieval metadata must not be passed downstream as perceived facts.
    text = re.split(r'(?im)^\s*NEED\s*:', text)[0].strip()
    return dict(id=identifier, clip_sha256=source_sha, interval=call['window'] or [0.,duration],
                audio_coverage=audio, frames=call.get('frames'), text=text,
                kind='model_observation_not_verified_truth', rejected_nondescriptive_fields=rejected, truncated=call['reached_token_limit'], field_token_caps=call.get('memory_fields',{}).get('field_token_caps',{}),
                video_tensor_sha256=call.get('video_tensor_sha256',call.get('raw_frames_sha256')),
                crop_sha256=call.get('crop',{}).get('sha256'))


def isolated_local_prompt(row, interval):
    return (f'The supplied media is ONLY [{interval[0]:.3f}, {interval[1]:.3f}] seconds of the source clip. '
        'Observe it afresh. Describe only visible actions, heard words/sounds, and direct audiovisual links relevant to the question. '
        'Use one short sentence per modality. Say UNKNOWN for an unsupported link. '
        'Do not infer whole-video counts or give an option answer. '
        'Do not invent timestamps. Question: '+question_only(row))



def isolated_verify_prompt(row, local_memories):
    text=('Inspect the ORIGINAL GLOBAL video and audio afresh to answer this question. '
          'Do not infer speech from a visible face alone. Check the complete clip for whole-clip counts and ordering. ')
    if local_memories:
        text+=('An additional local tool observation is available below. It may be wrong or incomplete. '
               'Only the recorded source interval was viewed; a local absence is not global absence. '
               'Use the original media over conflicting text.\n<local_observation>\n'
               +memory_text(local_memories)+'\n</local_observation>\n')
    return text+row['question_prompt']+'\nReturn ONLY one option letter A, B, C, or D.'


def run(runtime,row,duration,baseline,policy):
    begin=time.perf_counter();runtime.clear();runtime.active_duration=duration
    from support import DATA
    source=DATA[runtime.dataset]/row['clip_path'];clip_sha=sha(source)
    first=runtime.call(row,scout_prompt(row,duration,policy),None,'initial_memory',128)
    fs=first['memory_fields'];ledger=first['ledger_media'];calls=[first]
    valid=all(fs.get(k) for k in ['CANDIDATE','EVIDENCE','ANCHOR','RELATION'])
    valid=valid and fs['CANDIDATE'] in 'ABCD' and fs['ANCHOR'] in ['NONE']+[f['index'] for f in ledger['frames']]
    assert valid
    old=route_answer(baseline['model_output'],row);new=fs['CANDIDATE']
    confirmation=None;review=False;window=None;window_source=None;verification_memories=[]
    proposal_record=dict(id='proposal_ledger',clip_sha256=clip_sha,interval=[0.,duration],
        text=fs['EVIDENCE'],anchor=fs['ANCHOR'],relation=fs['RELATION'],
        audio_coverage=ledger['audio']['source_union'],frames=first['frames'],
        kind='candidate_conditioned_generated_evidence_not_verified_truth',
        field_token_caps=fs.get('field_token_caps',{}),truncated=first['reached_token_limit'])
    if first['reached_token_limit']:
        output=baseline['model_output'];decision='truncated_ledger_keep_native'
    elif new==old:
        output=baseline['model_output'];decision='agreement'
    else:
        window,window_source=interval_for(fs,ledger,duration)
        review=window is not None
        if review:
            local=runtime.call(row,isolated_local_prompt(row,window),window,'local_observation',96)
            calls.append(local);local_note=note(local,clip_sha,duration,'verification_local')
            local_note['actual_source_audio']=local['actual_source_audio']
            if local['reached_token_limit']:local_note['text']+=' [PARTIAL: generation token cap reached]'
            verification_memories=[local_note]
        check=runtime.call(row,isolated_verify_prompt(row,verification_memories),None,'disagreement_verification',16)
        calls.append(check);confirmation=route_answer(check['model_output'],row)
        accepted=confirmation==new and not check['reached_token_limit']
        output=new if accepted else baseline['model_output']
        decision='verified_revision' if accepted else 'unconfirmed_revision_keep_native'
    assert 1<=len(calls)<=3 and all(c['audio_tokens']>0 and c['video_tokens']>0 for c in calls)
    assert sum(c['generated_tokens'] for c in calls)<=240
    incremental=time.perf_counter()-begin
    return dict(model_output=output,calls=calls,memories=[proposal_record]+verification_memories,
        proposal_memory_ids=['proposal_ledger'],verification_memory_ids=[m['id'] for m in verification_memories],
        verifier_context_policy='raw_global_and_fresh_local_only_no_candidate_conditioned_ledger',
        scout_fields=fs,scout_schema_valid=valid,review=review,review_window=window,window_source=window_source,
        decision=decision,native_answer=old,proposal_answer=new,confirmation_answer=confirmation,
        reused_native_source=baseline['source'],baseline_cost_seconds=baseline['seconds'],
        new_work_seconds=incremental,seconds=incremental+baseline['seconds'],logical_calls=1+len(calls),executed_calls=len(calls),
        cache_start='empty_application_cache_per_question',
        cost_note='Historical native cost plus all measured new work including metadata, CPU/IO; not matched-epoch speed benchmark.')


