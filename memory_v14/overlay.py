"""Immutable CPU marker for initial proposal copies only."""
import hashlib
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def annotate(video, source_times, font_path, encoding='rgb255', mean=None, std=None):
    """Tag native-sized TCHW tensors with one visible address per temporal pair.

    Only the declared top-left rectangle changes. This is a media modification,
    not pixel-identical preprocessing. The original array is never modified.
    """
    assert video.ndim == 4 and video.shape[1] == 3
    t, _, h, w = video.shape
    assert t % 2 == 0 and 2 <= t <= 128 and len(source_times) == t
    assert np.isfinite(video).all() and all(np.isfinite(source_times))
    assert all(a <= b for a,b in zip(source_times,source_times[1:]))
    assert encoding in ['rgb255','normalized']
    font_size=max(12,round(min(h,w)*.052))
    font=ImageFont.truetype(str(font_path),font_size)
    bbox=font.getbbox('P63');pad=max(2,round(font_size*.15))
    rw=bbox[2]-bbox[0]+2*pad;rh=bbox[3]-bbox[1]+2*pad
    assert rw<=w and rh<=h and rw*rh/(w*h)<=.04
    out=video.copy();mask=np.zeros((h,w),dtype=bool);mask[:rh,:rw]=True;mapping=[]
    for pair in range(t//2):
        address=f'P{pair:02d}'
        plate=Image.new('RGB',(rw,rh),'black')
        ImageDraw.Draw(plate).text((pad-bbox[0],pad-bbox[1]),address,font=font,fill='white')
        pixels=np.asarray(plate).astype(np.float32).transpose(2,0,1)
        if encoding=='normalized':
            assert mean is not None and std is not None and len(mean)==len(std)==3 and min(std)>0
            pixels=(pixels/255-np.asarray(mean,dtype=np.float32)[:,None,None])/np.asarray(std,dtype=np.float32)[:,None,None]
        out[2*pair:2*pair+2,:,:rh,:rw]=pixels.astype(video.dtype)
        mapping.append(dict(address=address,input_positions=[2*pair,2*pair+1],source_seconds=list(source_times[2*pair:2*pair+2]),
            center_seconds=float(sum(source_times[2*pair:2*pair+2])/2)))
    assert np.array_equal(out[:,:,~mask],video[:,:,~mask])
    return out,dict(mapping=mapping,rectangle_xywh=[0,0,rw,rh],occluded_fraction=rw*rh/(w*h),
        font_size=font_size,font_sha256=hashlib.sha256(open(font_path,'rb').read()).hexdigest(),
        original_tensor_sha256=digest(video),marked_tensor_sha256=digest(out),shape=list(video.shape),dtype=str(video.dtype),
        encoding=encoding,note='Pair span denotes supplied frames, not continuous observed video; valid address is not semantic localization.')
