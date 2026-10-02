"""Prospectively fixed, label-blind working memory and one-review controller."""
import math
import re

VERSION = 'av-memory-agent-v5'
WINDOW_SECONDS = 12.0


def question(row):
    text = row['question_prompt']
    if '\nQuestion: ' in text:
        return text.split('\nQuestion: ', 1)[1]
    if text.startswith('Select the best answer'):
        return '\n'.join(text.splitlines()[1:]).removesuffix('\nThe best answer is:')
    return text


def window(center, duration):
    assert math.isfinite(duration) and duration > 0
    width = min(WINDOW_SECONDS, duration)
    start = min(max(center - width / 2, 0), duration - width)
    return [round(start, 3), min(round(start + width, 3), duration)]


def parse_memory(text, duration):
    """Ambiguous or incomplete records trigger one bounded fallback, never guessing."""
    clean = text.replace('**', '').strip().strip('`')
    fields = {}
    patterns = {
        'answer': r'\bANSWER\s*[:=]\s*([A-D])(?=\s|[.;,]|$)',
        'seek': r'\bSEEK\s*[:=]\s*(NONE|-?\d+(?:\.\d+)?)(?=\s|[;,]|\.(?!\d)|$)',
    }
    for key, pattern in patterns.items():
        values = re.findall(pattern, clean, re.I)
        fields[key] = values[0].upper() if len(values) == 1 else None
    match = re.search(r'\bEVIDENCE\s*[:=]\s*(.+?)(?=\b(?:SEEK|ANSWER)\s*[:=]|$)', clean, re.I | re.S)
    evidence = match.group(1).strip()[:700] if match else ''
    no_review=fields['seek']=='NONE'
    seek = float(fields['seek']) if fields['seek'] not in [None,'NONE'] else None
    valid_seek = seek is not None and math.isfinite(seek) and 0 <= seek <= duration
    valid=fields['answer'] is not None and bool(evidence) and (no_review or valid_seek)
    review=not valid or not no_review
    return dict(candidate=fields['answer'], evidence=evidence, schema_valid=valid,
                review=review, center=seek if valid_seek else duration / 2,
                tool_fallback=review and not valid_seek,
                reason='invalid_memory' if not valid else ('sufficient_evidence' if no_review else 'requested_missing_or_conflicting_evidence'))


def initial_prompt(row, duration, policy):
    return f'''Inspect this audiovisual clip and write a brief evidence note before selecting an answer.
The local timeline is 0 to {duration:.3f} seconds. Input coverage: {policy}.
Decide whether one closer synchronized 12-second view is necessary because specific evidence is missing or conflicting.
Do not request review merely to increase confidence. For order/counting retain evidence across the full timeline.
Visual presence alone does not establish who is speaking. Do not invent inaudible speech or future events.
QUESTION AND OPTIONS:
{question(row)}
Start with EVIDENCE, not with the option letter. Use exactly these three short lines:
EVIDENCE=<at most 25 words: observed audio/visual clues and timestamps, including any specific missing or conflicting clue>
SEEK=<NONE if observed evidence suffices; otherwise select one time-bin code from {', '.join(str(i)+'='+format(duration*(i+.5)/8,'.3f')+'s' for i in range(8))}>
ANSWER=<your provisional best choice: one letter A, B, C or D>
Write the evidence and SEEK decision before the answer. Do not output the answer alone.'''


def structured_memory(fields,duration):
    """Decision values selected by native logits under syntax constraints; no labels."""
    candidate=fields['answer'];seek=fields['seek'];evidence=fields['evidence']
    assert candidate in 'ABCD' and seek in ['NONE']+[str(i) for i in range(8)]
    valid=bool(evidence.strip());review=seek!='NONE' or not valid
    return dict(candidate=candidate,evidence=evidence[:700],schema_valid=valid,review=review,
                center=duration*(int(seek)+.5)/8 if seek!='NONE' else duration/2,
                tool_fallback=not valid,reason='invalid_memory' if not valid else ('sufficient_evidence' if seek=='NONE' else 'requested_evidence'),
                selected_bin=seek,evidence_limited=fields['evidence_limited'])


def final_prompt(row, duration, interval, memory):
    return f'''Give the final answer after this single audiovisual review.
The original clip is {duration:.3f} seconds. This excerpt covers original local time [{interval[0]:.3f}, {interval[1]:.3f}] seconds.
Add {interval[0]:.3f} to excerpt-relative timestamps. The excerpt is only part of the original clip.
Working memory from your first observation (fallible, not an annotation):
{memory}
Keep full-clip context for order/counting; overlapping observations are not new events.
Correct the initial impression only when observed evidence supports it. Do not infer speaker identity from appearance alone.
QUESTION AND OPTIONS:
{question(row)}
Respond with ONLY the correct option letter A, B, C or D.'''
