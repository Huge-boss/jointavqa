import unittest
from agent import parse_memory,window


def record(seek='NONE',answer='B'):
    return f'EVIDENCE=At 3s a visible person speaks.\nSEEK={seek}\nANSWER={answer}'


class ControllerTests(unittest.TestCase):
    def test_early_exit_and_review(self):
        self.assertFalse(parse_memory(record(),10)['review'])
        for seek in ['0','3.2','10']:
            p=parse_memory(record(seek=seek),10)
            self.assertTrue(p['review']);self.assertTrue(p['schema_valid'])

    def test_unsafe_and_ambiguous_requests(self):
        for value in ['NaN','inf','-1','11','1e2','1.2.3','__import__(os)']:
            p=parse_memory(record(seek=value),10)
            self.assertTrue(p['review'],value);self.assertTrue(p['tool_fallback'],value)
        for suffix in ['\nSEEK=2','\nANSWER=C','\nSEEK=NONE']:
            self.assertTrue(parse_memory(record()+suffix,10)['review'])

    def test_incomplete_or_conflicting_memory(self):
        for text in ['B',record().replace('EVIDENCE=At 3s a visible person speaks.',''),record().replace('ANSWER=B','')]:
            self.assertTrue(parse_memory(text,10)['review'])

    def test_tolerant_punctuation_and_case(self):
        p=parse_memory(record(seek='3.2').lower().replace('seek=3.2','seek: 3.2.').replace('answer=b','**answer: b**'),10)
        self.assertTrue(p['schema_valid']);self.assertEqual(p['candidate'],'B')

    def test_short_long_and_boundary_windows(self):
        for d in [3.970637,5,12,31,1742.64]:
            for center in [0,d/2,d]:
                a,b=window(center,d)
                self.assertGreaterEqual(a,0);self.assertLessEqual(b,d);self.assertLessEqual(b-a,12.001);self.assertLess(a,b)


if __name__=='__main__':unittest.main()
