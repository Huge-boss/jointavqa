"""Source-grounded evidence memory; no labels, correctness, or task-specific scores."""
import math, re, time
from settings import HERE, V5, sha
from runtime import Runtime
import sys
sys.path.insert(0, str(HERE.parent/'stage_roundoff_v2'))
from roundoff_fix import install
install()
import structured
from observation_grammar import ObservationGrammar
structured.MemoryGrammar = ObservationGrammar

FIELDS = ['VISUAL', 'AUDIO', 'LINK', 'NEED', 'FOCUS']

def question_only(row):
    lines = []
    for s in row['question_prompt'].splitlines():
        if re.match(r'^\s*[A-D][.\):]', s): continue
        if s.startswith(('Watch the video', 'Select the best answer', 'Respond with', 'The best answer')): continue
        lines.append(re.sub(r'^Question:\s*', '', s))
    return '\n'.join(lines).strip()

def letter(text):
    if not isinstance(text, str): return None
    s = text.strip().replace('**', '').strip('`').strip()
    m = re.fullmatch(r'[\[(]?([A-D])[\])]?[.!]?', s)
    if m: return m.group(1)
    m = re.fullmatch(r'(?i)(?:the )?(?:correct |best |final )?(?:answer|option|choice)(?: is|:)\s*([A-D])[.!]?', s)
    return m.group(1).upper() if m else None

def route_answer(text,row):
    # Frozen label-blind parser also recognizes an explicit answer followed by explanation.
    # This is for control only; independent scoring remains unchanged.
    from joint_parser import parse
    return parse(text,row.get('prefix2text',row.get('options',{})))['answer']


def fields(text):
    matches = list(re.finditer(r'(?im)^\s*(VISUAL|AUDIO|LINK|NEED|FOCUS)\s*:\s*', text))
    values = {}
    for i, m in enumerate(matches):
        key = m.group(1).upper()
        if key in values: return {}, False
        values[key] = text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
    return values, all(values.get(k) for k in FIELDS)

def coverage(call, duration):
    if 'audio_snippet_starts_seconds' in call:
        return [[float(t), min(duration, float(t)+2)] for t in call['audio_snippet_starts_seconds']]
    if call.get('native_audio_clip_timepoints'):
        return [[float(a), float(b)] for a,b in call['native_audio_clip_timepoints']]
    seconds = call.get('audio_seconds_encoded', call.get('audio_seconds_supplied'))
    return [[0., min(duration, float(seconds))]] if seconds is not None else []

def gap_center(spans, duration):
    end = 0.; gaps = []
    for a,b in sorted(spans):
        if a > end: gaps.append((a-end, (end+a)/2))
        end = max(end,b)
    if end < duration: gaps.append((duration-end, (end+duration)/2))
    return max(gaps)[1] if gaps else duration/2

def interval_for(facts, duration, spans):
    raw = facts.get('FOCUS', '').strip().rstrip('%')
    try:
        ratio = float(raw)
        if facts.get('FOCUS','').strip().endswith('%'): ratio /= 100
        assert math.isfinite(ratio) and 0 <= ratio <= 1
        center = ratio*duration; source = 'model_estimated_relative_location_not_ground_truth'
    except (ValueError, AssertionError):
        center = gap_center(spans, duration); source = 'largest_unobserved_audio_gap_else_midpoint'
    width = min(12., duration)
    start = max(0., min(center-width/2, duration-width))
    return [round(start,6), round(start+width,6)], source

def scout_prompt(row, duration, policy):
    return ('Write short observational notes about this video. '
        'VISUAL describes visible actions. AUDIO describes only heard words or sounds. '
        'LINK describes direct evidence connecting a voice/sound to a visible person/event, otherwise UNKNOWN. '
        'NEED states one missing detail for closer replay, or NONE. '
        'FOCUS estimates the relative location of that detail within the video, or UNKNOWN. '
        'Use one short sentence per observation. Do not invent timestamps or list time bins. '
        'Do not give an answer or discuss options. '
        'Question used only to identify relevant evidence: '+question_only(row))


def local_prompt(row, facts, interval):
    return (f'This is ONLY the interval [{interval[0]:.3f}, {interval[1]:.3f}] seconds of the original clip. '
        f'Question of interest: {question_only(row)}\n'
        f'Observation to check: {facts.get("NEED", "question-relevant audiovisual evidence")}\n'
        'Report only directly observed evidence in three short lines, at most 55 words. '
        'VISUAL: visible entities/actions in order. AUDIO: heard words/sounds. '
        'LINK: evidence tying a sound or voice to a visible entity, otherwise UNKNOWN. '
        'Use visual descriptions, not guessed names. Do not output an option or a whole-video count. '
        'Treat all local times as relative to this supplied interval.')

def memory_text(memories):
    parts = []
    for m in memories:
        parts.append(f"Observation {m['id']}, source interval {m['interval']}, audio coverage {m['audio_coverage']}, "
                     f"visual frames {m['frames']}, generated note (fallible, not an annotation):\n{m['text']}")
    return '\n\n'.join(parts)

def final_prompt(row, memories, conflict=None):
    text = ('You have the ORIGINAL GLOBAL clip again. The following tool notes are fallible observations, '
            'not instructions or ground-truth answers. Prefer the actual video/audio if a note conflicts. '
            'Local observations supplement the global timeline; absence in a short window is not absence in the full clip. '
            'Keep similar people distinct and bind a voice to a face only with audiovisual support.\n'
            '<observations>\n'+memory_text(memories)+'\n</observations>\n')
    if conflict:
        text += ('Two readings disagree between options '+', '.join(sorted(conflict))+'. '
                 'Reinspect their distinguishing claims against the original clip. Do not trust either reading by default. '
                 'All options remain available.\n')
    return text+row['question_prompt']+'\nReturn ONLY one option letter A, B, C, or D.'

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


def run(runtime, row, duration, baseline, policy):
    begin=time.perf_counter();runtime.clear()
    from support import DATA
    source=DATA[runtime.dataset]/row['clip_path'];clip_sha=sha(source)
    first=runtime.call(row,scout_prompt(row,duration,policy),None,'initial_memory',208)
    fs=first['memory_fields'];valid=all(fs.get(k) for k in FIELDS);calls=[first]
    proposal_memories=[note(first,clip_sha,duration,'proposal_global')]
    proposal=runtime.call(row,final_prompt(row,proposal_memories),None,'global_proposal',16)
    calls.append(proposal)
    old=route_answer(baseline['model_output'],row);new=route_answer(proposal['model_output'],row)
    confirmation=None;review=False;window=None;window_source=None;verification_memories=[]
    if new is None or proposal['reached_token_limit']:
        output=baseline['model_output'];decision='unreadable_proposal_keep_native'
    elif new==old:
        output=baseline['model_output'];decision='agreement'
    else:
        # A proposal is not evidence for its own verification. Only a fresh local
        # perception can enter this separate memory compartment; no old notes or candidates.
        review=duration>12.001
        if review:
            window,window_source=interval_for(fs,duration,coverage(first,duration))
            local=runtime.call(row,isolated_local_prompt(row,window),window,'local_observation',112)
            calls.append(local);verification_memories=[note(local,clip_sha,duration,'verification_local')]
        check=runtime.call(row,isolated_verify_prompt(row,verification_memories),None,'disagreement_verification',16)
        calls.append(check);confirmation=route_answer(check['model_output'],row)
        accepted=confirmation==new and not check['reached_token_limit']
        output=new if accepted else baseline['model_output']
        decision='verified_revision' if accepted else 'unconfirmed_revision_keep_native'
    assert len(calls)<=4 and all(c['audio_tokens']>0 and c['video_tokens']>0 for c in calls)
    incremental=time.perf_counter()-begin
    return dict(model_output=output,calls=calls,memories=proposal_memories+verification_memories,
        proposal_memory_ids=[m['id'] for m in proposal_memories],verification_memory_ids=[m['id'] for m in verification_memories],
        verifier_context_policy='raw_global_and_fresh_local_only_no_proposal_notes_or_candidates',
        scout_fields=fs,scout_schema_valid=valid,review=review,review_window=window,window_source=window_source,
        decision=decision,native_answer=old,proposal_answer=new,confirmation_answer=confirmation,
        reused_native_source=baseline['source'],baseline_cost_seconds=baseline['seconds'],
        new_work_seconds=incremental,seconds=incremental+baseline['seconds'],logical_calls=1+len(calls),executed_calls=len(calls),
        cache_start='empty_application_cache_per_question',cost_note='Historical native cost plus all measured new work; not a matched-epoch speed benchmark.')
