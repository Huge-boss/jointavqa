import unittest,tempfile
from pathlib import Path
import common_stage as c


class StageTests(unittest.TestCase):
    def test_partition_has_exactly_missing_questions(self):
        ids=[str(i) for i in range(571)];completed=ids[:96]
        a,b=c.split_missing(ids,completed)
        self.assertEqual(len(a),238);self.assertEqual(len(b),237)
        self.assertFalse(set(a)&set(b));self.assertEqual(set(a+b),set(ids)-set(completed))
        self.assertEqual(c.split_missing(ids,ids),([],[]))

    def test_invalid_manifest_rejected(self):
        with self.assertRaises(AssertionError):c.split_missing(['a','a'],[])
        with self.assertRaises(AssertionError):c.split_missing(['a'],['x'])

    def test_imported_and_new_records_cannot_duplicate_success(self):
        original=c.HERE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                c.HERE=Path(tmp)
                root=c.HERE/'runs/joint/omni/fixed'
                c.jsonlines(root/'inherited.jsonl',[dict(qid='a',status='ok')])
                c.jsonlines(root/'gpu0/predictions.jsonl',[dict(qid='b',status='error'),dict(qid='b',status='ok')])
                found,errors=c.arm_records('joint','omni','fixed')
                self.assertEqual(set(found),{'a','b'});self.assertEqual(len(errors),1)
                c.jsonlines(root/'gpu3/predictions.jsonl',[dict(qid='a',status='ok')])
                with self.assertRaises(AssertionError):c.arm_records('joint','omni','fixed')
        finally:c.HERE=original


if __name__=='__main__':unittest.main()
