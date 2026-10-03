"""Label-free route, source-clock and grammar tests; no GPU needed."""
import unittest
from unittest.mock import patch
from method import interval_for,source_coverage,media_ledger,run,isolated_verify_prompt,ledger_prompt
from ledger_grammar import LedgerGrammar

class Tokenizer:
 def encode(self,s,add_special_tokens=False):return [ord(c) for c in s]
 def decode(self,ids,**kw):return ''.join(chr(i) for i in ids if i)

class Checks(unittest.TestCase):
 def setUp(self):
  self.fs=dict(CANDIDATE='B',EVIDENCE='A person lifts an object.',ANCHOR='P01',RELATION='AROUND',field_token_caps={})
  self.ledger=dict(pairs=[dict(address='P00',source_seconds=[0.,1.]),dict(address='P01',source_seconds=[170.24,172.24])],frames=[dict(index='F000',source_seconds=0.),dict(index='F001',source_seconds=171.24)],audio={'source_union':[[0.,300.]]})
 def test_unknown_never_gap(self):
  self.fs['ANCHOR']='NONE';self.assertIsNone(interval_for(self.fs,self.ledger,530.92)[0])
 def test_actual_anchor(self):self.assertEqual(interval_for(self.fs,self.ledger,530.92)[0],[165.24,177.24])
 def test_before(self):
  self.fs['RELATION']='BEFORE';self.assertEqual(interval_for(self.fs,self.ledger,530.92)[0],[158.24,170.24])
 def test_after(self):
  self.fs['RELATION']='AFTER';self.assertEqual(interval_for(self.fs,self.ledger,530.92)[0],[172.24,184.24])
 def test_edge(self):
  self.fs['ANCHOR']='P00';self.assertEqual(interval_for(self.fs,self.ledger,20.)[0],[0.,12.])
 def test_short(self):self.assertIsNone(interval_for(self.fs,self.ledger,12.)[0])
 def test_partial(self):
  self.fs['field_token_caps']={'EVIDENCE':True};self.assertIsNone(interval_for(self.fs,self.ledger,530.92)[0])
 def test_nondescriptive(self):
  self.fs['EVIDENCE']='14.2';self.assertIsNone(interval_for(self.fs,self.ledger,530.92)[0])
 def test_global(self):
  self.fs['RELATION']='GLOBAL';self.assertIsNone(interval_for(self.fs,self.ledger,530.92)[0])
 def test_sparse_pair_no_unseen_midpoint(self):
  self.ledger['pairs'][1]['source_seconds']=[150.,175.]
  self.assertIsNone(interval_for(self.fs,self.ledger,530.92)[0])
 def test_source_union(self):
  c=source_coverage({'native_audio_clip_timepoints':[[0,2],[1,3],[4,6]]},5,10)
  self.assertEqual(c['source_union'],[[10,13],[14,15]]);self.assertEqual(c['unique_seconds'],4);self.assertEqual(c['repeated_seconds'],1)
 def test_source_frames(self):
  m=media_ledger({'frames':2,'frame_indices':[0,100],'source_fps':25,'audio_seconds_supplied':4},'',4.1)
  self.assertEqual(m['frames'][1]['source_seconds'],4)
 def test_av_frames(self):
  m=media_ledger({'frames':2,'video_metadata':{'frames_indices':[0,30],'fps':30},'audio_seconds_supplied':1},'',1.1)
  self.assertEqual(m['frames'][1]['index'],'F001')
 def test_grammar(self):
  tok=Tokenizer();g=LedgerGrammar(tok,0,2);out=[]
  target='B\nEVIDENCE: A red object.\nANCHOR: P01\nRELATION: AFTER'
  for ch in target:
   allowed=g.allowed(out,ord(ch))
   if isinstance(allowed,list):self.assertIn(ord(ch),allowed)
   out.append(ord(ch))
  self.assertEqual(g.allowed(out,0),[0]);out.append(0)
  self.assertEqual(g.fields(out)['ANCHOR'],'P01')
 def test_decimal_not_early_termination(self):
  g=LedgerGrammar(Tokenizer(),0,2);out=[]
  target='B\nEVIDENCE: At 1.5 s a man moves.\nANCHOR: P01\nRELATION: AFTER'
  for ch in target:
   allowed=g.allowed(out,ord(ch))
   if isinstance(allowed,list):self.assertIn(ord(ch),allowed)
   out.append(ord(ch))
  self.assertEqual(g.allowed(out,0),[0]);out.append(0)
  self.assertEqual(g.fields(out)['EVIDENCE'],'At 1.5 s a man moves.')
 def test_initial_context_not_numeric_table(self):
  l=dict(self.ledger,audio={'source_intervals':[[0,300]]})
  text=ledger_prompt({'question_prompt':'Question?'},l)
  self.assertNotIn('171.24',text);self.assertNotIn('300',text);self.assertIn('P01',text)
 def test_context(self):
  text=isolated_verify_prompt({'question_prompt':'What happens?\nA. left\nB. right'},[])
  self.assertNotIn('CANDIDATE',text);self.assertNotIn('Input-frame',text)
 def route(self,candidate='B',confirmation='B',anchor='P01'):
  fs=dict(self.fs,CANDIDATE=candidate,ANCHOR=anchor)
  ledger=self.ledger
  class Fake:
   dataset='joint'
   def clear(self):pass
   def call(self,row,prompt,interval,stage,limit):
    base=dict(model_output=confirmation,generated_tokens=3,reached_token_limit=False,audio_tokens=2,video_tokens=2,frames=2,window=interval,stage=stage)
    if stage=='initial_memory':base.update(memory_fields=fs,ledger_media=ledger)
    if stage=='local_observation':base.update(model_output='Visible motion.',actual_source_audio={})
    return base
  with patch('method.sha',return_value='hash'):
   return run(Fake(),{'clip_path':'a.mp4','question_prompt':'Q?','options':{'A':'one','B':'two','C':'three','D':'four'}},530.92,{'model_output':'A','source':{},'seconds':2},'')
 def test_agreement_one_call(self):
  r=self.route(candidate='A');self.assertEqual(r['executed_calls'],1);self.assertEqual(r['model_output'],'A')
 def test_anchor_three_calls(self):
  r=self.route();self.assertEqual(r['executed_calls'],3);self.assertEqual(r['model_output'],'B');self.assertTrue(r['review'])
 def test_unknown_two_calls(self):
  r=self.route(anchor='NONE');self.assertEqual(r['executed_calls'],2);self.assertFalse(r['review'])
 def test_verifier_reject(self):self.assertEqual(self.route(confirmation='C')['model_output'],'A')

if __name__=='__main__':unittest.main()
