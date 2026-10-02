import unittest
from unittest.mock import patch
from method import run


class Runtime:
    dataset = 'joint'

    def __init__(self, proposal='B', verify='B', truncated=False):
        self.proposal, self.verify, self.truncated = proposal, verify, truncated
        self.seen = []

    def clear(self):
        pass

    def call(self, row, prompt, interval, stage, limit):
        self.seen.append((stage, prompt, interval))
        text = {'initial_memory': 'PROPOSAL_SECRET', 'global_proposal': self.proposal,
                'local_observation': 'FRESH_LOCAL', 'disagreement_verification': self.verify}[stage]
        return dict(model_output=text, stage=stage, window=interval, audio_tokens=10,
                    video_tokens=20, frames=8, audio_seconds_encoded=10,
                    reached_token_limit=self.truncated and stage == 'disagreement_verification',
                    memory_fields=(dict(VISUAL='PROPOSAL_SECRET', AUDIO='AUDIO_SECRET', LINK='LINK_SECRET',
                                        NEED='NEED_SECRET', FOCUS='0.5', field_token_caps={})
                                   if stage == 'initial_memory' else {}))


class Isolation(unittest.TestCase):
    row = dict(clip_path='dummy.mp4', question_prompt='Who?\nA. Man\nB. Woman',
               options={'A': 'Man', 'B': 'Woman', 'C': 'Child', 'D': 'Nobody'})
    baseline = dict(model_output='A', seconds=2., source={'path': 'historical'})

    def evaluate(self, r, duration=30):
        with patch('method.sha', return_value='hash'):
            return run(r, self.row, duration, self.baseline, 'native')

    def test_compartments_do_not_leak(self):
        r = Runtime(); a = self.evaluate(r)
        proposal = next(p for s,p,w in r.seen if s == 'global_proposal')
        local = next(p for s,p,w in r.seen if s == 'local_observation')
        verify = next(p for s,p,w in r.seen if s == 'disagreement_verification')
        self.assertIn('PROPOSAL_SECRET', proposal)
        for secret in ['PROPOSAL_SECRET', 'AUDIO_SECRET', 'LINK_SECRET', 'NEED_SECRET']:
            self.assertNotIn(secret, verify); self.assertNotIn(secret, local)
        self.assertIn('FRESH_LOCAL', verify)
        self.assertNotIn('Two readings', verify)
        self.assertEqual(a['verification_memory_ids'], ['verification_local'])
        self.assertEqual(a['model_output'], 'B')

    def test_agreement_does_not_crop_or_verify(self):
        r = Runtime(proposal='A'); a = self.evaluate(r)
        self.assertEqual([s for s,p,w in r.seen], ['initial_memory', 'global_proposal'])
        self.assertFalse(a['review']); self.assertEqual(a['model_output'], 'A')

    def test_truncated_verification_never_changes_native(self):
        a = self.evaluate(Runtime(truncated=True))
        self.assertEqual(a['model_output'], 'A')

    def test_short_clip_uses_only_global_verification(self):
        r = Runtime(); a = self.evaluate(r, 8.)
        self.assertEqual(len(r.seen), 3)
        self.assertTrue(all(w is None for s,p,w in r.seen))
        self.assertEqual(a['verification_memory_ids'], [])


if __name__ == '__main__':
    unittest.main()
