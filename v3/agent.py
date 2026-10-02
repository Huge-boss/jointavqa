"""Prospectively fixed, label-blind working memory and one-review controller."""
import math
import re

VERSION = 'av-memory-agent-v3'
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
        'review': r'\bREVIEW\s*[:=]\s*(YES|NO)\b',
        'gap': r'\bGAP\s*[:=]\s*(NONE|MISSING|CONFLICT)\b',
        'seek': r'\bSEEK\s*[:=]\s*(-?\d+(?:\.\d+)?)(?=\s|[;,]|\.(?!\d)|$)',
    }
    for key, pattern in patterns.items():
        values = re.findall(pattern, clean, re.I)
        fields[key] = values[0].upper() if len(values) == 1 else None
    match = re.search(r'\bEVIDENCE\s*[:=]\s*(.+)', clean, re.I | re.S)
    evidence = match.group(1).strip()[:700] if match else ''
    seek = float(fields['seek']) if fields['seek'] is not None else None
    valid_seek = seek is not None and math.isfinite(seek) and 0 <= seek <= duration
    complete = all(fields[k] is not None for k in ['answer', 'review', 'gap']) and bool(evidence)
    consistent = (fields['review'] == 'NO' and fields['gap'] == 'NONE') or (fields['review'] == 'YES' and fields['gap'] in ['MISSING', 'CONFLICT'])
    valid = complete and consistent and valid_seek
    review = not valid or fields['review'] == 'YES'
    return dict(candidate=fields['answer'], evidence=evidence, schema_valid=valid,
                review=review, gap=fields['gap'], center=seek if valid_seek else duration / 2,
                tool_fallback=not valid_seek, reason='invalid_memory' if not valid else fields['gap'].lower())


def initial_prompt(row, duration, policy):
    return f'''Answer this audiovisual multiple-choice question from the supplied clip.
The local timeline is 0 to {duration:.3f} seconds. Input coverage: {policy}.
Decide whether one closer synchronized 12-second view is necessary because specific evidence is missing or conflicting.
Do not request review merely to increase confidence. For order/counting retain evidence across the full timeline.
Visual presence alone does not establish who is speaking. Do not invent inaudible speech or future events.
QUESTION AND OPTIONS:
{question(row)}
Return exactly these five short fields, without introductory prose:
ANSWER=<one letter A, B, C or D; your provisional best choice>
REVIEW=<YES if specific evidence is missing/conflicting, otherwise NO>
GAP=<MISSING, CONFLICT or NONE; consistent with REVIEW>
SEEK=<one numeric time from 0 to {duration:.3f}; use the midpoint if no review needed>
EVIDENCE=<at most 35 words: observed audio/visual clues and local timestamps; if reviewing, identify the specific missing or conflicting clue>'''


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
