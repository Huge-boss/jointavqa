"""Behavioral checks for tool safety, fixed budgets, parsing and deterministic splits."""
import unittest
from agent import window, seek, run_arm
from prepare import choose
from joint_parser import self_test


class Contracts(unittest.TestCase):
    def test_windows_never_leave_published_clip(self):
        for duration in [4, 12, 65, 3600]:
            for center in [0, duration / 2, duration]:
                a, b = window(center, duration)
                self.assertGreaterEqual(a, 0)
                self.assertLessEqual(b, duration)
                self.assertLessEqual(b-a, 12.001)

    def test_invalid_tools_use_deterministic_fallback(self):
        for bad in ['SEEK=nan', 'SEEK=999', 'SEEK=-1', 'SEEK=1\nSEEK=2', 'execute rm -rf /']:
            self.assertTrue(seek(bad, 30, 10)['fallback'])
        self.assertEqual(seek('SEEK=28.2\nA person appears.', 30, 10)['center'], 28.2)

    def test_adaptive_second_lookup_and_equal_calls(self):
        row = {'question_prompt': 'Question? A. One B. Two C. Three D. Four'}
        global_call = {'model_output': 'SEEK=5\nInitial evidence'}
        seen = []
        def infer(prompt, interval, stage):
            seen.append((interval, stage))
            return {'model_output': 'SEEK=26\nNew evidence' if stage.endswith('inspect_1') else 'B'}
        agent = run_arm(row, 30, global_call, 'agent', infer)
        self.assertEqual(agent['decisions'][1]['center'], 26)
        self.assertEqual(len(seen), 3)
        seen.clear()
        fixed = run_arm(row, 30, global_call, 'fixed', infer)
        self.assertEqual([d['center'] for d in fixed['decisions']], [10, 20])
        self.assertEqual(len(seen), 3)
        self.assertEqual(agent['logical_calls'], fixed['logical_calls'])

    def test_split_independent_of_input_order(self):
        rows = [{'qid': str(i), 'task': str(i % 3)} for i in range(103)]
        a, counts = choose(rows, 'test'); b, _ = choose(list(reversed(rows)), 'test')
        self.assertEqual(a, b); self.assertEqual(len(a), 21)
        with self.assertRaises(AssertionError):
            choose([dict(rows[0], correct_answer='A')], 'test')

    def test_existing_parser_contracts(self):
        self.assertEqual(self_test(), 29)


if __name__ == '__main__':
    unittest.main()
