"""Post-hoc, label-blind answer audit. Does not modify any inference or primary score.

Rules frozen after inspecting unlabeled Omni response formats, before scoring.
Apply these same rules to every output of every model, including older baselines.
"""
import re

VERSION = 'explicit-answer-v2-20261002'
LETTER = r'\(?([A-D])\)?(?=\s|[.,:;!)]|$)'
PATTERNS = [
    ('declaration', re.compile(
        r'(?i:\b(?:the\s+)?(?:(?:correct|final|best|most\s+likely)\s+)?'
        r'(?:answer|option|choice|order)\s*(?:is\s+|would\s+be\s+|[:：]\s*))' + LETTER)),
    ('pronoun_declaration', re.compile(r"(?i:\b(?:it['’]s|it\s+is)\s+)" + LETTER)),
    ('first_person_selection', re.compile(
        r"(?i:\bI(?:['’]d|\s+would)?\s+(?:choose|select|pick|go\s+with|say)\s+)" + LETTER)),
]


def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().strip('\"\'').rstrip('.')).casefold()


def parse(text, options):
    """Only response text and unlabeled options are accepted, never ground truth."""
    text = re.sub(r'^```(?:text)?\s*|\s*```$', '', str(text).strip()).strip().replace('**', '')
    exact = [k for k, v in options.items() if normalize(v) == normalize(text)]
    if len(exact) == 1:
        return {'answer': exact[0], 'rule': 'exact_option_text'}
    candidates = []
    leading = re.match(r'^\s*\(?([A-D])(?:\)|(?=\s*(?:[.,:;!]|$)))', text)
    if leading:
        candidates.append(('leading_letter', leading))
    for rule, pattern in PATTERNS:
        candidates.extend((rule, m) for m in pattern.finditer(text))
    for rule, m in candidates:
        # Never turn alternatives, hypothetical statements or negated claims into answers.
        tail = text[m.end():]
        if re.match(r'\s*(?:/|or\b|and\b|,|->|→)\s*\(?[A-D]\)?(?:\W|$)', tail):
            return {'answer': None, 'rule': 'ambiguous_alternatives'}
        if rule != 'leading_letter':
            clause = re.split(r'[.!?;\n]', text[:m.start()])[-1]
            if re.search(r"(?i)\b(?:if|suppose|example|not|never|don['’]t|isn['’]t)\b", clause):
                return {'answer': None, 'rule': 'qualified_or_negated_declaration'}
    letters = {m.group(1) for _, m in candidates}
    if len(letters) > 1:
        return {'answer': None, 'rule': 'conflicting_declarations'}
    if letters:
        return {'answer': next(iter(letters)), 'rule': '+'.join(sorted({r for r, _ in candidates}))}
    return {'answer': None, 'rule': 'no_unambiguous_answer'}


def self_test():
    options = {'A': 'A cat', 'B': 'A man walks', 'C': 'Calm', 'D': '(b) (a) (c)'}
    cases = [
        ('B.', 'B'), ('(C)', 'C'), ('D', 'D'), ('**A.** A cat', 'A'),
        ('A man walks', 'B'), ('Calm.', 'C'), ('A man walks into a room.', None),
        ('The correct order is D. First b, then A, then c.', 'D'),
        ('Hmm, let me think. So, the answer is C.Got another question?', 'C'),
        ("Hmm, I'd say it's B. The man walks.", 'B'),
        ("So, I'd go with A.", 'A'), ('The most likely option is C.', 'C'),
        ('A or B', None), ('A/B', None), ('A, B', None), ('A -> B -> C', None),
        ('The answer is A or B.', None), ('The answer is A/B.', None),
        ('The answer is A, B, or C.', None), ('The answer is not A.', None),
        ("I don't think the answer is A.", None),
        ('If the answer is A, then the object is red.', None),
        ('The answer is A. The final answer is B.', None),
        ('B. The answer is A.', None), ('The answer is A, not B.', 'A'),
        ('a casual scene', None), ('The answer is E.', None),
        ('I cannot determine the answer.', None), ('The order is b, a, c.', None),
    ]
    for text, expected in cases:
        assert parse(text, options)['answer'] == expected, (text, parse(text, options), expected)
    return len(cases)


if __name__ == '__main__':
    print('Passed', self_test(), 'synthetic parser contracts')
