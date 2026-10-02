"""Official model classes, explicit native audio sampling, audited multimodal length."""
import json
import sys
import types
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from PIL import Image
from transformers import AutoTokenizer,GenerationConfig
from core import HERE,MODELS,digest,write_json
from media import read_audio

VENDOR=HERE.parents[3]/'reproduction/custom_v1/vendor/VideoLLaMA2-official'
sys.path.insert(0,str(VENDOR))
from videollama2.model.videollama2_qwen2 import Videollama2Qwen2Config,Videollama2Qwen2ForCausalLM
from videollama2.mm_utils import tokenizer_multimodal_token

def restore_checkpoint_vision(model,path):
 """Delayed vision construction must not discard vision weights bundled in the AV checkpoint."""
 from safetensors import safe_open
 mapping=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
 prefix='model.vision_tower.vision_tower.'
 wanted={k:v for k,v in mapping.items() if k.startswith(prefix)}
 if not wanted:raise ValueError('AV checkpoint does not contain expected bundled vision weights')
 state={}
 for shard in set(wanted.values()):
  with safe_open(path/shard,framework='pt',device='cpu') as f:
   for key,filename in wanted.items():
    if filename==shard:state[key[len(prefix):]]=f.get_tensor(key)
 tower=model.get_vision_tower().vision_tower
 assert set(state)==set(tower.state_dict()),'Incomplete bundled vision state'
 tower.load_state_dict(state,strict=True)
 return len(state)

class Adapter:
 def __init__(self,path):
  self.tokenizer=AutoTokenizer.from_pretrained(str(path),use_fast=False,local_files_only=True)
  self.tokenizer.pad_token=self.tokenizer.eos_token
  cfg=Videollama2Qwen2Config.from_pretrained(str(path),local_files_only=True)
  cfg.mm_vision_tower=str(MODELS/'siglip-so400m-patch14-384')
  cfg.mm_audio_tower=str(path/'audio_tower.bin');cfg.num_frames=32
  self.model,info=Videollama2Qwen2ForCausalLM.from_pretrained(str(path),config=cfg,torch_dtype=torch.bfloat16,
    device_map={'':0},attn_implementation='flash_attention_2',low_cpu_mem_usage=True,local_files_only=True,output_loading_info=True)
  # Fail loudly if any text, audio or projector weights are absent. No random initialization fallback.
  bad=[k for k in info['missing_keys'] if not k.startswith('model.vision_tower.')]
  if bad:raise ValueError('Missing checkpoint weights: '+str(bad[:20]))
  if info.get('mismatched_keys') or info.get('error_msgs'):raise ValueError(str(info))
  tower=self.model.get_vision_tower()
  if not tower.is_loaded:tower.load_model()
  self.restored_vision_keys=restore_checkpoint_vision(self.model,path)
  self.processor=tower.image_processor
  self.model.to(device='cuda',dtype=torch.bfloat16).eval()
  assert self.model.get_audio_tower() is not None
  self.stats={}
  def audit_video(module,args,out):self.stats['video_tokens']=out.numel()//out.shape[-1]
  def audit_audio(module,args,out):self.stats['audio_tokens']=out.numel()//out.shape[-1]
  self.model.get_model().mm_projector.register_forward_hook(audit_video)
  self.model.get_model().mm_projector_a.register_forward_hook(audit_audio)
  original=self.model.prepare_inputs_labels_for_multimodal
  def checked(*args,**kwargs):
   result=original(*args,**kwargs)
   if result[3] is not None:
    n=int(result[3].shape[1]);self.stats['input_tokens']=n
    if n+256>self.model.config.max_position_embeddings:raise ValueError('Multimodal context capacity exceeded')
   return result
  self.model.prepare_inputs_labels_for_multimodal=checked
  self.loading_info=info

 def infer(self,row,path,raw,meta):
  self.stats={}
  audio,ameta=read_audio(path)
  # The official AV encoder consumes 8 uniform 2-second snippets then pads to 30s.
  # Bound the eligible window to the same first-300s policy used by Omni.
  seconds=len(audio)/16000;starts=np.linspace(0,max(0,seconds-2),8)
  clips=[torch.from_numpy(audio[int(s*16000):min(len(audio),int(s*16000)+32000)].copy()) for s in starts]
  selected=torch.cat(clips);used=len(selected)/16000
  if len(selected)>480000:raise ValueError('Native audio sample budget exceeded')
  padded=torch.nn.functional.pad(selected,(0,480000-len(selected)))
  fbank=kaldi.fbank(padded.unsqueeze(0)*32768,num_mel_bins=128,sample_frequency=16000,frame_length=25,frame_shift=10,dither=0).unsqueeze(0)
  pixels=self.processor.preprocess([Image.fromarray(f) for f in raw],return_tensors='pt')['pixel_values']
  media={'video':pixels.to(device='cuda',dtype=torch.bfloat16),'audio':fbank.to(device='cuda',dtype=torch.bfloat16)}
  msg=[{'role':'user','content':'<video>\n'+row['question_prompt']}]
  prompt=self.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
  ids=tokenizer_multimodal_token(prompt,self.tokenizer,'<video>',return_tensors='pt').unsqueeze(0).long().cuda()
  assert int((ids<0).sum())==1
  conf=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=256,repetition_penalty=1.0,
     eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.eos_token_id)
  with torch.inference_mode():
   out=self.model.generate(ids,attention_mask=torch.ones_like(ids),images=[(media,'video')],generation_config=conf,use_cache=True)
  assert self.stats.get('audio_tokens',0)>0 and self.stats.get('video_tokens',0)>0
  text=self.tokenizer.batch_decode(out,skip_special_tokens=True)[0].strip()
  return dict(model_output=text,generated_tokens=int(out.shape[1]),reached_token_limit=out.shape[1]>=256,
    audio_seconds_encoded=used,audio_native_padded_seconds=30,audio_snippet_starts_seconds=starts.tolist(),
    audio_native_policy='8 uniform 2-second snippets inside first300s, then zero-pad30s',
    processed_frame_shape=list(pixels.shape),serialized_prompt=prompt,**ameta,**self.stats)
