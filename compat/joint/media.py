"""Explicit common temporal selection; native spatial transforms live in adapters."""
import hashlib
import math
import subprocess
import numpy as np

def read_video(path):
 from decord import VideoReader, cpu
 vr=VideoReader(str(path),ctx=cpu(0),num_threads=2)
 total=len(vr);fps=float(vr.get_avg_fps());duration=total/fps
 if total<2 or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid video metadata')
 count=int(min(max(duration*2,4),32,total))//2*2
 indices=np.rint(np.linspace(0,total-1,count)).astype('int64')
 video=vr.get_batch(indices.tolist()).asnumpy()
 times=vr.get_frame_timestamp(indices.tolist())[:,0].tolist()
 effective=(count-1)/(times[-1]-times[0])
 if not np.isfinite(video).all() or effective<=0:raise ValueError('Invalid video tensor/time')
 return video,{'frames':count,'source_frames':total,'source_fps':fps,'duration_seconds':duration,
               'frame_indices':indices.tolist(),'timestamps_seconds':times,'effective_fps':effective,
               'raw_frames_sha256':hashlib.sha256(video.tobytes()).hexdigest()}

def read_audio(path):
 p=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(path),'-t','300',
                   '-map','0:a:0','-vn','-ac','1','-ar','16000','-f','f32le','pipe:1'],
                   check=True,capture_output=True,timeout=150)
 a=np.frombuffer(p.stdout,dtype=np.float32).copy()
 if len(a)<1600 or not np.isfinite(a).all():raise ValueError('Invalid audio')
 return a,{'audio_window_start':0,'audio_seconds_supplied':len(a)/16000,
           'audio_sha256':hashlib.sha256(a.tobytes()).hexdigest()}

def qwen_video(raw):
 import torch
 from torchvision.transforms.functional import resize
 from torchvision.transforms import InterpolationMode
 h,w=raw.shape[1:3];factor=28;lo=128*28*28;hi=768*28*28
 rh=max(factor,round(h/factor)*factor);rw=max(factor,round(w/factor)*factor)
 if rh*rw>hi:
  scale=math.sqrt(h*w/hi);rh=max(factor,math.floor(h/scale/factor)*factor);rw=max(factor,math.floor(w/scale/factor)*factor)
 elif rh*rw<lo:
  scale=math.sqrt(lo/(h*w));rh=math.ceil(h*scale/factor)*factor;rw=math.ceil(w*scale/factor)*factor
 return resize(torch.from_numpy(raw).permute(0,3,1,2),[rh,rw],interpolation=InterpolationMode.BICUBIC,antialias=True).float()
