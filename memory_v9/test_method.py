import unittest
from unittest.mock import patch
from observation_grammar import ObservationGrammar
from method import question_only,fields,interval_for,gap_center,note,letter,final_prompt,run

class FakeRuntime:
    dataset='joint'
    def clear(self):pass

class Tests(unittest.TestCase):
    def test_question_excludes_options(self):
        q=question_only(dict(question_prompt='Watch the video carefully.\nQuestion: Who speaks?\nA. Man\nB. Woman'))
        self.assertEqual(q,'Who speaks?')
    def test_fields_duplicate_unknown(self):
        d,v=fields('VISUAL: man\nAUDIO: unknown\nLINK: UNKNOWN\nNEED: words\nFOCUS: 0.75')
        self.assertTrue(v);self.assertEqual(d['FOCUS'],'0.75')
        self.assertEqual(fields('VISUAL: x\nVISUAL: y'),({},False))
    def test_windows_and_missing_coverage(self):
        for d in [3.970637,12.,31.,1742.64]:
            for f in ['0','1','0.25','UNKNOWN','NaN','-1','1.1']:
                w,_=interval_for({'FOCUS':f},d,[[0,min(d,300)]])
                self.assertGreaterEqual(w[0],0);self.assertLessEqual(w[1],d+1e-6)
                self.assertGreater(w[1],w[0]);self.assertLessEqual(w[1]-w[0],12.000001)
        self.assertEqual(gap_center([[0,10],[20,22]],40),31)
    def test_source_timestamps_not_generated_times(self):
        c=dict(window=[20,32],model_output='VISUAL: at 500s a man\nAUDIO: a word\nLINK: UNKNOWN\nNEED: more\nFOCUS: 1',
            audio_snippet_starts_seconds=[0,10],reached_token_limit=False,frames=8)
        m=note(c,'abc',40,'local')
        self.assertEqual(m['interval'],[20,32]);self.assertEqual(m['audio_coverage'],[[20,22],[30,32]])
        self.assertNotIn('NEED:',m['text'])
    def test_final_keeps_global_warning(self):
        p=final_prompt({'question_prompt':'Q?\nA. One\nB. Two'},[],['B','A'])
        self.assertIn('ORIGINAL GLOBAL',p);self.assertIn('A, B',p)
        self.assertNotIn('baseline',p.lower())
    def test_control_parser_rejects_ambiguous(self):
        self.assertEqual(letter('A'),'A');self.assertEqual(letter('The answer is B.'),'B')
        self.assertIsNone(letter('A or B'));self.assertIsNone(letter('A. Some text B'))
    def test_tool_syntax_complete_without_answer_field(self):
        class Tok:
            def encode(self,s,add_special_tokens=False):return list(map(ord,s))
            def decode(self,x,**kw):return ''.join(map(chr,x))
        grammar=ObservationGrammar(Tok(),0);g=[]
        for _ in range(400):
            allowed=grammar.allowed(g,ord('x'))
            token=ord('x') if allowed is None or isinstance(allowed,tuple) else allowed[0]
            g.append(token)
            if token==0:break
        f=grammar.fields(g)
        self.assertEqual(set(f)-{'field_token_caps','syntax'},set(['VISUAL','AUDIO','LINK','NEED','FOCUS']))
        self.assertTrue(all(f['field_token_caps'].values()))
    def test_revision_requires_second_agreement_and_global_input(self):
        class R:
            dataset='joint'
            def __init__(self,last):self.last=last;self.seen=[]
            def clear(self):pass
            def call(self,row,prompt,interval,stage,limit):
                self.seen.append((interval,stage,prompt))
                text='B' if stage=='global_proposal' else self.last if stage=='disagreement_verification' else 'VISUAL: man\nAUDIO: unknown\nLINK: unknown'
                return dict(model_output=text,window=interval,stage=stage,reached_token_limit=False,
                    audio_tokens=10,video_tokens=20,frames=4,audio_seconds_encoded=5,seconds=1,
                    memory_fields=dict(VISUAL='man',AUDIO='unknown',LINK='unknown',NEED='NONE',FOCUS='UNKNOWN',field_token_caps={}))
        row=dict(clip_path='dummy.mp4',question_prompt='Q?\nA. One\nB. Two')
        baseline=dict(model_output='A',seconds=2,source={'path':'historical'})
        with patch('method.sha',return_value='sha'):
            agree=R('B');a=run(agree,row,5,baseline,'native')
            disagree=R('A');b=run(disagree,row,5,baseline,'native')
        self.assertEqual(a['model_output'],'B');self.assertEqual(b['model_output'],'A')
        self.assertTrue(all(interval is None for interval,stage,p in agree.seen))
        self.assertNotIn('Two readings',agree.seen[1][2])
        self.assertEqual(a['logical_calls'],4)

if __name__=='__main__':unittest.main()
