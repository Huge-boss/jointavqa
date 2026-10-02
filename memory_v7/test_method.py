import unittest
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

if __name__=='__main__':unittest.main()
