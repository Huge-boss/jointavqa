import torch
from transformers import Qwen2_5OmniForConditionalGeneration,Qwen2_5OmniProcessor,GenerationConfig
from media import qwen_video,read_audio

SYSTEM='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
class Adapter:
 def __init__(self,path):
  self.model=Qwen2_5OmniForConditionalGeneration.from_pretrained(str(path),torch_dtype=torch.bfloat16,
   device_map={'':0},attn_implementation='flash_attention_2',enable_audio_output=False,local_files_only=True).eval()
  self.processor=Qwen2_5OmniProcessor.from_pretrained(str(path),local_files_only=True,use_fast=False)
  self.processor.tokenizer.padding_side='left'
  assert not any(str(v) in ['cpu','disk'] for v in self.model.hf_device_map.values())
  self.model.thinker.generation_config.do_sample=False
  self.model.thinker.generation_config.repetition_penalty=1.0
  self.model.thinker.generation_config.num_beams=1
  self.model.thinker.generation_config.top_p=1.0
  self.model.thinker.generation_config.temperature=1.0
  self.model.thinker.generation_config.top_k=50
 def infer(self,row,path,raw,meta):
  audio,ameta=read_audio(path);video=qwen_video(raw)
  messages=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
   {'role':'user','content':[{'type':'video','video':str(path)},{'type':'text','text':row['question_prompt']}]}]
  text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  x=self.processor(text=text,audio=[audio],videos=[video],return_tensors='pt',padding=True,
    use_audio_in_video=True,fps=meta['effective_fps'],audio_kwargs={'truncation':False})
  n=x.input_ids.shape[1];conf=self.model.config.thinker_config
  used=float(x.feature_attention_mask.sum())*self.processor.feature_extractor.hop_length/16000
  ameta.update(audio_seconds_encoded=used,audio_tokens=int((x.input_ids==conf.audio_token_index).sum()),
    video_tokens=int((x.input_ids==conf.video_token_index).sum()),input_tokens=n,
    processor_fps=meta['effective_fps'],processed_frame_shape=list(video.shape),serialized_prompt=text)
  assert ameta['audio_tokens']>0 and ameta['video_tokens']>0
  assert abs(used-len(audio)/16000)<0.05,'Unexpected audio truncation'
  if n+256>conf.text_config.max_position_embeddings:raise ValueError('Context capacity exceeded')
  x=x.to(self.model.device).to(torch.bfloat16)
  with torch.inference_mode():
   out=self.model.generate(**x,use_audio_in_video=True,return_audio=False,thinker_do_sample=False,
        thinker_num_beams=1,thinker_repetition_penalty=1.0,thinker_max_new_tokens=256)
  gen=out[:,n:];text=self.processor.batch_decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
  return dict(model_output=text,generated_tokens=int(gen.shape[1]),reached_token_limit=gen.shape[1]>=256,**ameta)
