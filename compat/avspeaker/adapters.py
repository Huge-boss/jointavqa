"""Native model processing; no access to answer labels."""
from common import *
import sys, hashlib, numpy as np, torch

SYSTEM='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'

class Omni:
 def __init__(self):
  from transformers import Qwen2_5OmniForConditionalGeneration,Qwen2_5OmniProcessor
  path=MODELS/'Qwen2.5-Omni-7B'
  self.model=Qwen2_5OmniForConditionalGeneration.from_pretrained(str(path),torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='flash_attention_2',enable_audio_output=False,local_files_only=True).eval()
  self.processor=Qwen2_5OmniProcessor.from_pretrained(str(path),local_files_only=True,use_fast=False)
  self.model.thinker.generation_config.do_sample=False
  self.model.thinker.generation_config.repetition_penalty=1.0
  self.model.thinker.generation_config.num_beams=1
 def infer(self,row):
  import qwen_omni_utils as q
  path=DATA/row['clip_path'];item={'type':'video','video':str(path),'fps':1.0}
  msg=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},{'role':'user','content':[item,{'type':'text','text':row['question_prompt']}]}]
  # Native utility returns actual sample fps after even-frame rounding.
  vision=q.process_mm_info.__globals__['process_vision_info']
  fetch=vision.__globals__['fetch_video']
  (video,meta),fps=fetch(item,return_video_sample_fps=True,return_video_metadata=True)
  audio=q.process_mm_info.__globals__['process_audio_info'](msg,True)
  assert len(audio)==1 and len(audio[0])>0 and np.isfinite(audio[0]).all()
  prompt=self.processor.apply_chat_template(msg,add_generation_prompt=True,tokenize=False)
  x=self.processor(text=prompt,audio=audio,videos=[video],return_tensors='pt',padding=True,use_audio_in_video=True,fps=fps,audio_kwargs={'truncation':False})
  n=x.input_ids.shape[1];cfg=self.model.config.thinker_config
  encoded=float(x.feature_attention_mask.sum())*self.processor.feature_extractor.hop_length/16000
  assert abs(encoded-len(audio[0])/16000)<0.05,'Audio unexpectedly truncated'
  assert n+256<=cfg.text_config.max_position_embeddings,'Input exceeds native context'
  stats={'frames':len(video),'video_metadata':meta,'processor_fps':fps,'pixel_shape':list(video.shape),
   'input_tokens':n,'audio_tokens':int((x.input_ids==cfg.audio_token_index).sum()),'video_tokens':int((x.input_ids==cfg.video_token_index).sum()),
   'audio_seconds_supplied':len(audio[0])/16000,'audio_seconds_encoded':encoded,'audio_sha256':hashlib.sha256(audio[0].tobytes()).hexdigest(),
   'video_tensor_sha256':hashlib.sha256(video.numpy().tobytes()).hexdigest(),'serialized_prompt':prompt}
  assert stats['audio_tokens']>0 and stats['video_tokens']>0
  x=x.to(self.model.device).to(torch.bfloat16)
  with torch.inference_mode():out=self.model.generate(**x,use_audio_in_video=True,return_audio=False,thinker_do_sample=False,thinker_num_beams=1,thinker_repetition_penalty=1.0,thinker_max_new_tokens=256)
  generated=out[:,n:]
  text=self.processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
  return {'model_output':text,'generated_tokens':int(generated.shape[1]),'reached_token_limit':generated.shape[1]>=256,**stats}

class VideoLLaMA:
 def __init__(self):
  from transformers import AutoTokenizer
  vendor=ROOT/'reproduction/custom_v1/vendor/VideoLLaMA2-official'
  sys.path.insert(0,str(vendor))
  import videollama2.mm_utils as mm
  from videollama2.model.videollama2_qwen2 import Videollama2Qwen2Config,Videollama2Qwen2ForCausalLM
  from safetensors import safe_open
  self.mm=mm;path=MODELS/'VideoLLaMA2.1-7B-AV';self.tokenizer=AutoTokenizer.from_pretrained(str(path),use_fast=False,local_files_only=True)
  if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
  cfg=Videollama2Qwen2Config.from_pretrained(str(path),local_files_only=True)
  cfg.mm_vision_tower=str(MODELS/'siglip-so400m-patch14-384');cfg.mm_audio_tower=str(path/'audio_tower.bin');cfg.num_frames=8
  self.model,info=Videollama2Qwen2ForCausalLM.from_pretrained(str(path),config=cfg,torch_dtype=torch.float16,device_map={'':0},attn_implementation='flash_attention_2',low_cpu_mem_usage=True,local_files_only=True,output_loading_info=True)
  assert not [k for k in info['missing_keys'] if not k.startswith('model.vision_tower.')],info
  assert not info.get('mismatched_keys') and not info.get('error_msgs'),info
  tower=self.model.get_vision_tower()
  if not tower.is_loaded:tower.load_model()
  mapping=read(path/'model.safetensors.index.json')['weight_map'];prefix='model.vision_tower.vision_tower.'
  wanted={k:v for k,v in mapping.items() if k.startswith(prefix)};state={}
  for shard in set(wanted.values()):
   with safe_open(path/shard,framework='pt',device='cpu') as f:
    for k,v in wanted.items():
     if v==shard:state[k[len(prefix):]]=f.get_tensor(k)
  assert set(state)==set(tower.vision_tower.state_dict()) and len(state)==448
  tower.vision_tower.load_state_dict(state,strict=True);del state
  self.processor=tower.image_processor;self.model.to(device='cuda',dtype=torch.float16).eval();self.stats={}
  self.model.get_model().mm_projector.register_forward_hook(lambda m,a,o:self.stats.update(video_tokens=o.numel()//o.shape[-1]))
  self.model.get_model().mm_projector_a.register_forward_hook(lambda m,a,o:self.stats.update(audio_tokens=o.numel()//o.shape[-1]))
  original=self.model.prepare_inputs_labels_for_multimodal
  def checked(*a,**kw):
   result=original(*a,**kw)
   if result[3] is not None:
    n=int(result[3].shape[1]);self.stats['input_tokens']=n
    assert n+256<=self.model.config.max_position_embeddings,'Context overflow'
   return result
  self.model.prepare_inputs_labels_for_multimodal=checked
  load=mm.load_audio_from_video
  def checked_load(*a,**kw):
   wav,sr=load(*a,**kw)
   assert len(wav)>0 and np.isfinite(wav).all() and sr==16000
   self.stats.update(audio_seconds_supplied=len(wav)/sr,audio_sha256=hashlib.sha256(wav.tobytes()).hexdigest(),native_audio_decode_ok=True)
   return wav,sr
  mm.load_audio_from_video=checked_load
  sampler=mm.get_clip_timepoints
  def timepoints(*a,**kw):
   times=sampler(*a,**kw);self.stats['native_audio_clip_timepoints']=[[float(x) for x in t] for t in times];return times
  mm.get_clip_timepoints=timepoints
  frame_sample=mm.frame_sample
  def frames(*a,**kw):
   indices=frame_sample(*a,**kw);self.stats['frame_indices']=indices.tolist();return indices
  mm.frame_sample=frames
 def infer(self,row):
  import copy
  self.stats={}
  media=self.mm.process_video(str(DATA/row['clip_path']),self.processor,aspect_ratio=None,num_frames=8,va=True)
  assert self.stats.get('native_audio_decode_ok'), 'Native audio loader failed; silent-zero fallback rejected'
  assert media['video'].shape[0]==8 and torch.isfinite(media['audio']).all()
  self.stats['pixel_shape']=list(media['video'].shape);self.stats['audio_feature_shape']=list(media['audio'].shape)
  self.stats['video_tensor_sha256']=hashlib.sha256(media['video'].numpy().tobytes()).hexdigest()
  self.stats['frames']=8
  media={k:v.to(device='cuda',dtype=torch.float16) for k,v in media.items()}
  prompt=self.tokenizer.apply_chat_template([{'role':'user','content':'<video>\n'+row['question_prompt']}],tokenize=False,add_generation_prompt=True)
  ids=self.mm.tokenizer_multimodal_token(prompt,self.tokenizer,'<video>',return_tensors='pt').unsqueeze(0).long().cuda()
  conf=copy.deepcopy(self.model.generation_config)
  conf.do_sample=False;conf.num_beams=1;conf.max_new_tokens=256;conf.repetition_penalty=1.0;conf.pad_token_id=self.tokenizer.eos_token_id
  stop=self.mm.KeywordsStoppingCriteria([self.tokenizer.eos_token],self.tokenizer,ids)
  with torch.inference_mode():out=self.model.generate(ids,attention_mask=ids.ne(self.tokenizer.pad_token_id).long(),images=[(media,'video')],generation_config=conf,use_cache=True,stopping_criteria=[stop])
  assert self.stats.get('audio_tokens',0)>0 and self.stats.get('video_tokens',0)>0
  text=self.tokenizer.batch_decode(out,skip_special_tokens=True)[0].strip()
  return {'model_output':text,'generated_tokens':int(out.shape[1]),'reached_token_limit':out.shape[1]>=256,'serialized_prompt':prompt,**self.stats}
