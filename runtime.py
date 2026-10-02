"""Read-only baseline compatibility and synchronized, bounded excerpt tool."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = {'joint': Path('/data/lzq/data/JointAVBench'), 'avspeaker': Path('/data/lzq/data/AV-SpeakerBench')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def probe(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], timeout=60))
    video = next(s for s in data['streams'] if s['codec_type'] == 'video')
    assert any(s['codec_type'] == 'audio' for s in data['streams'])
    return float(video.get('duration', data['format']['duration']))


def excerpt(source, interval, dest):
    # Accurate post-input seeking. Preserve yuv420 video with lossless x264 coding;
    # decoded audio is written as uncompressed 16-bit PCM, with no AAC re-encoding.
    start, end = interval
    assert 0 <= start < end and end - start <= 12.001
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', str(source),
           '-ss', str(start), '-t', str(end - start), '-map', '0:v:0', '-map', '0:a:0',
           '-sn', '-dn', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', '-threads', '2',
           '-c:a', 'pcm_s16le', '-map_metadata', '-1', '-y', str(dest)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=240)
    observed = probe(dest)
    assert abs(observed - (end - start)) < .25, (interval, observed)
    return {'requested_window': interval, 'decoded_duration_seconds': observed,
            'excerpt_sha256': sha(dest), 'audio_storage': 'decoded PCM16; no lossy audio re-encoding',
            'video_storage': 'x264 crf0; original dimensions; no spatial crop'}


class Runtime:
    def __init__(self, dataset, model, cache):
        self.dataset, self.model, self.cache = dataset, model, cache
        sys.path.insert(0, str(HERE / 'compat' / dataset))
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        if dataset == 'joint':
            mod = importlib.import_module('adapter_' + model)
            core = importlib.import_module('core')
            self.adapter = mod.Adapter(core.MODELS / core.MODEL_NAMES[model])
            self.read_video = importlib.import_module('media').read_video
        else:
            mod = importlib.import_module('adapters')
            self.adapter = mod.Omni() if model == 'omni' else mod.VideoLLaMA()

    def seed(self):
        from transformers import set_seed
        set_seed(42)

    def call(self, row, prompt, interval, stage):
        import gc
        begin = time.time()
        source = (DATA[self.dataset] / row['clip_path']).resolve()
        assert source.is_relative_to(DATA[self.dataset]) and source.is_file()
        path = source
        crop_meta = {}
        if interval is not None:
            path = self.cache / 'window.mov'
            crop_meta = excerpt(source, interval, path)
        r = dict(row, question_prompt=prompt, clip_path=str(path))
        self.seed()
        self.torch.cuda.reset_peak_memory_stats()
        if self.dataset == 'joint':
            raw, meta = self.read_video(path)
            out = {**meta, **self.adapter.infer(r, path, raw, meta)}
            del raw
        else:
            out = self.adapter.infer(r)
        assert isinstance(out['model_output'], str) and out['model_output'].strip()
        assert out.get('audio_tokens', 0) > 0 and out.get('video_tokens', 0) > 0
        out.update(stage=stage, window=interval, crop=crop_meta, seconds=time.time() - begin,
                   peak_gpu_gib=self.torch.cuda.max_memory_allocated() / 2**30)
        gc.collect()
        return out
