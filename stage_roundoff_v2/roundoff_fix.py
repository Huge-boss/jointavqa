"""Repair only the floating-point comparison; preserve crop and generation inputs."""
import math
import subprocess
import runtime


def valid_interval(interval):
    a, b = interval
    width = b - a
    return (math.isfinite(a) and math.isfinite(b) and 0 <= a < b
            and (width <= 12.001 or math.isclose(width, 12.001, rel_tol=0, abs_tol=1e-12)))


def excerpt(source, interval, dest):
    a, b = interval
    assert valid_interval(interval), repr(interval)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-ss', str(a), '-i', str(source),
                    '-t', str(b-a), '-map', '0:v:0', '-map', '0:a:0', '-sn', '-dn',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', '-threads', '2',
                    '-c:a', 'pcm_s16le', '-map_metadata', '-1', '-y', str(dest)], check=True, capture_output=True, timeout=240)
    observed = runtime.probe(dest)
    assert abs(observed-(b-a)) < .25
    return dict(requested_window=interval, duration=observed, sha256=runtime.sha(dest), audio_storage='PCM16', video_storage='lossless x264 crf0')


def install():
    runtime.excerpt = excerpt
