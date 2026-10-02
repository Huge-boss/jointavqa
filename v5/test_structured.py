import unittest
from structured import MemoryGrammar
from agent import structured_memory,window


class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(map(ord,text))
    def decode(self,tokens,skip_special_tokens=True):return ''.join(map(chr,tokens))


def generate(seek,answer,cap=False,bos=False):
    grammar=MemoryGrammar(Tokenizer(),[0],evidence_limit=5)
    text='EVIDENCE=clues\nSEEK='+seek+'\nANSWER='+answer
    # Native EOS after the first clue forces the rest of the structured decision.
    if not cap:text=text.replace('clues','x')
    tokens=[]
    for ch in text:
        preferred=0 if len(tokens)>=len('EVIDENCE=')+(5 if cap else 1) else ord(ch)
        allowed=grammar.allowed(tokens,preferred)
        if isinstance(allowed,tuple):assert ord(ch) not in allowed[1]
        elif allowed is not None:assert ord(ch) in allowed,(ch,allowed)
        tokens.append(ord(ch))
    assert grammar.allowed(tokens,0)==[0]
    return grammar.fields(([999] if bos else [])+tokens+[0])


class GrammarTests(unittest.TestCase):
    def test_eos_is_replaced_by_complete_decision(self):
        f=generate('NONE','C')
        self.assertEqual((f['evidence'],f['seek'],f['answer']),('x','NONE','C'))
        self.assertFalse(f['evidence_limited'])
        self.assertFalse(structured_memory(f,31)['review'])

    def test_body_cap_and_virtual_bos(self):
        f=generate('7','A',cap=True,bos=True)
        self.assertEqual(f['evidence'],'clues');self.assertTrue(f['evidence_limited'])
        m=structured_memory(f,1742.64)
        self.assertTrue(m['review']);self.assertAlmostEqual(m['center'],1742.64*7.5/8)

    def test_all_decision_values_and_window_bounds(self):
        for seek in ['NONE']+[str(i) for i in range(8)]:
            for answer in 'ABCD':
                f=generate(seek,answer)
                m=structured_memory(f,3.970637)
                a,b=window(m['center'],3.970637)
                self.assertEqual((a,b),(0,3.970637));self.assertTrue(m['schema_valid'])

    def test_invalid_and_incomplete_trace(self):
        grammar=MemoryGrammar(Tokenizer(),0)
        with self.assertRaises(AssertionError):grammar.fields([0])
        with self.assertRaises(AssertionError):grammar.choice([ord('Z')],grammar.answers)


if __name__=='__main__':unittest.main()
