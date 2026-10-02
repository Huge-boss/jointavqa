"""Native baseline preprocessing split from generation, with bounded CPU caches."""
import copy
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from support import HERE, ROOT, DATA, sha


def probe(path):
    d = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], timeout=60))
    v = next(s for s in d['streams'] if s['codec_type'] == 'video')
    assert any(s['codec_type'] == 'audio' for s in d['streams'])
    duration = float(v.get('duration', d['format']['duration']))
    assert math.isfinite(duration) and duration > 0
    return duration


def excerpt(source, interval, dest):
    a, b = interval
    assert 0 <= a < b and b-a <= 12.001
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-ss', str(a), '-i', str(source),
                    '-t', str(b-a), '-map', '0:v:0', '-map', '0:a:0', '-sn', '-dn',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', '-threads', '2',
                    '-c:a', 'pcm_s16le', '-map_metadata', '-1', '-y', str(dest)], check=True, capture_output=True, timeout=240)
    observed = probe(dest)
    assert abs(observed-(b-a)) < .25
    return dict(requested_window=interval, duration=observed, sha256=sha(dest), audio_storage='PCM16', video_storage='lossless x264 crf0')


class Runtime:
    def __init__(self, dataset, model):
        import torch
        self.torch, self.dataset, self.name = torch, dataset, model
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
        sys.path.insert(0, str(HERE.parent / 'compat' / dataset))
        if dataset == 'joint':
            core = importlib.import_module('core')
            self.media = importlib.import_module('media')
            self.adapter = importlib.import_module('adapter_' + model).Adapter(core.MODELS/core.MODEL_NAMES[model])
        else:
            mod = importlib.import_module('adapters')
            self.adapter = mod.Omni() if model == 'omni' else mod.VideoLLaMA()
        self.dtype = torch.float16 if dataset == 'avspeaker' and model == 'videollama' else torch.bfloat16
        self.cache = {}
        self.cache_dir = HERE / 'cache' / (dataset + '_' + model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def seed(self):
        from transformers import set_seed
        set_seed(42)

    def clear(self):
        self.cache.clear()
        self.adapter.stats = {}

    def prepare(self, path):
        import numpy as np
        torch, adapter = self.torch, self.adapter
        meta = {}
        if self.dataset == 'joint':
            raw, meta = self.media.read_video(path)
            audio, ameta = self.media.read_audio(path)
            meta.update(ameta)
            if self.name == 'omni':
                pixels = self.media.qwen_video(raw)
                meta['processed_frame_shape'] = list(pixels.shape)
                return dict(video=pixels, audio=audio, fps=meta['effective_fps'], meta=meta)
            from PIL import Image
            import torchaudio.compliance.kaldi as kaldi
            seconds = len(audio)/16000
            starts = np.linspace(0, max(0, seconds-2), 8)
            clips = [torch.from_numpy(audio[int(s*16000):min(len(audio),int(s*16000)+32000)].copy()) for s in starts]
            selected = torch.cat(clips)
            padded = torch.nn.functional.pad(selected, (0,480000-len(selected)))
            fbank = kaldi.fbank(padded.unsqueeze(0)*32768,num_mel_bins=128,sample_frequency=16000,frame_length=25,frame_shift=10,dither=0).unsqueeze(0)
            pixels = adapter.processor.preprocess([Image.fromarray(f) for f in raw],return_tensors='pt')['pixel_values']
            meta.update(audio_seconds_encoded=len(selected)/16000, audio_native_padded_seconds=30,
                        audio_snippet_starts_seconds=starts.tolist(), processed_frame_shape=list(pixels.shape))
            return dict(media=dict(video=pixels,audio=fbank), meta=meta)
        if self.name == 'omni':
            import qwen_omni_utils as q
            item = dict(type='video',video=str(path),fps=1.0)
            vision = q.process_mm_info.__globals__['process_vision_info']
            (video, vm), fps = vision.__globals__['fetch_video'](item,return_video_sample_fps=True,return_video_metadata=True)
            audio = q.process_mm_info.__globals__['process_audio_info']([{'role':'user','content':[item]}],True)
            assert len(audio)==1 and len(audio[0]) and np.isfinite(audio[0]).all()
            meta.update(frames=len(video),video_metadata=vm,processor_fps=fps,pixel_shape=list(video.shape),
                        audio_seconds_supplied=len(audio[0])/16000,audio_sha256=hashlib.sha256(audio[0].tobytes()).hexdigest(),
                        video_tensor_sha256=hashlib.sha256(video.numpy().tobytes()).hexdigest())
            return dict(video=video,audio=audio[0],fps=fps,meta=meta)
        adapter.stats = {}
        media = adapter.mm.process_video(str(path),adapter.processor,aspect_ratio=None,num_frames=8,va=True)
        assert adapter.stats.get('native_audio_decode_ok') and media['video'].shape[0]==8
        assert torch.isfinite(media['audio']).all()
        meta = copy.deepcopy(adapter.stats)
        meta.update(pixel_shape=list(media['video'].shape),audio_feature_shape=list(media['audio'].shape),frames=8,
                    video_tensor_sha256=hashlib.sha256(media['video'].numpy().tobytes()).hexdigest())
        return dict(media=media,meta=meta)

    def generate(self, row, path, prompt, prepared, limit):
        torch, adapter = self.torch, self.adapter
        stats = copy.deepcopy(prepared['meta'])
        if self.name == 'omni':
            system = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
            item = {'type':'video','video':str(path)}
            if self.dataset == 'avspeaker': item['fps']=1.0
            messages = [{'role':'system','content':[{'type':'text','text':system}]},
                        {'role':'user','content':[item,{'type':'text','text':prompt}]}]
            serialized = adapter.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
            x = adapter.processor(text=serialized,audio=[prepared['audio']],videos=[prepared['video']],return_tensors='pt',padding=True,
                                  use_audio_in_video=True,fps=prepared['fps'],audio_kwargs={'truncation':False})
            n = x.input_ids.shape[1]; cfg = adapter.model.config.thinker_config
            encoded = float(x.feature_attention_mask.sum())*adapter.processor.feature_extractor.hop_length/16000
            assert abs(encoded-len(prepared['audio'])/16000)<.05
            assert n+limit<=cfg.text_config.max_position_embeddings
            stats.update(input_tokens=n,audio_seconds_encoded=encoded,processor_fps=prepared['fps'],
                         audio_tokens=int((x.input_ids==cfg.audio_token_index).sum()),video_tokens=int((x.input_ids==cfg.video_token_index).sum()))
            x = x.to(adapter.model.device).to(self.dtype)
            with torch.inference_mode():
                out = adapter.model.generate(**x,use_audio_in_video=True,return_audio=False,thinker_do_sample=False,
                                             thinker_num_beams=1,thinker_repetition_penalty=1.0,thinker_max_new_tokens=limit)
            generated = out[:,n:]
            text = adapter.processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        else:
            adapter.stats = copy.deepcopy(stats)
            media = {k:v.to(device='cuda',dtype=self.dtype) for k,v in prepared['media'].items()}
            serialized = adapter.tokenizer.apply_chat_template([{'role':'user','content':'<video>\n'+prompt}],tokenize=False,add_generation_prompt=True)
            if self.dataset == 'joint':
                from videollama2.mm_utils import tokenizer_multimodal_token
                from transformers import GenerationConfig
                ids = tokenizer_multimodal_token(serialized,adapter.tokenizer,'<video>',return_tensors='pt').unsqueeze(0).long().cuda()
                cfg = GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=limit,repetition_penalty=1.0,
                                       eos_token_id=adapter.tokenizer.eos_token_id,pad_token_id=adapter.tokenizer.eos_token_id)
                mask = torch.ones_like(ids); extra={}
            else:
                ids = adapter.mm.tokenizer_multimodal_token(serialized,adapter.tokenizer,'<video>',return_tensors='pt').unsqueeze(0).long().cuda()
                cfg = copy.deepcopy(adapter.model.generation_config)
                cfg.do_sample=False;cfg.num_beams=1;cfg.max_new_tokens=limit;cfg.repetition_penalty=1.0;cfg.pad_token_id=adapter.tokenizer.eos_token_id
                mask = ids.ne(adapter.tokenizer.pad_token_id).long()
                extra = {'stopping_criteria':[adapter.mm.KeywordsStoppingCriteria([adapter.tokenizer.eos_token],adapter.tokenizer,ids)]}
            with torch.inference_mode():
                generated = adapter.model.generate(ids,attention_mask=mask,images=[(media,'video')],generation_config=cfg,use_cache=True,**extra)
            text = adapter.tokenizer.batch_decode(generated,skip_special_tokens=True)[0].strip()
            stats.update(adapter.stats)
        assert stats['audio_tokens']>0 and stats['video_tokens']>0 and text
        return dict(model_output=text,generated_tokens=int(generated.shape[1]),reached_token_limit=generated.shape[1]>=limit,
                    max_new_tokens=limit,serialized_prompt=serialized,**stats)

    def call(self, row, prompt, interval, stage, limit):
        begin = time.perf_counter()
        source = (DATA[self.dataset]/row['clip_path']).resolve()
        assert source.is_relative_to(DATA[self.dataset]) and source.is_file()
        key = (str(source), tuple(interval) if interval else None)
        self.seed(); self.torch.cuda.reset_peak_memory_stats()
        crop = {}; crop_seconds=0; prepare_seconds=0
        hit = key in self.cache
        if hit:
            path, prepared, crop = self.cache[key]
        else:
            assert len(self.cache)<2, 'Cache capacity is two observations per arm'
            path = source
            if interval:
                path = self.cache_dir/'review.mov'
                t=time.perf_counter();crop=excerpt(source,interval,path);crop_seconds=time.perf_counter()-t
            # Seed reset ensures caching never changes native feature randomness.
            self.seed();t=time.perf_counter();prepared=self.prepare(path);prepare_seconds=time.perf_counter()-t
            self.cache[key]=(path,prepared,crop)
        self.seed();t=time.perf_counter()
        out = self.generate(row,path,prompt,prepared,limit)
        generation_seconds=time.perf_counter()-t
        out.update(stage=stage,window=interval,crop=crop,cache_hit=hit,crop_seconds=crop_seconds,
                   cache_build_seconds=prepare_seconds,processor_and_generation_seconds=generation_seconds,
                   seconds=time.perf_counter()-begin,peak_gpu_gib=self.torch.cuda.max_memory_allocated()/2**30)
        return out
