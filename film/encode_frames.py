"""Encode PNG frame sequences build/<group>/<shot>/%04d.png to build/<group>/<shot>.mp4 (H.264, crf 18, yuv420p).

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python film/encode_frames.py [group=service] [shot ...]
"""

import os
import subprocess
import sys

import imageio_ffmpeg

FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 24


def encode(folder, out, fps=FPS):
    frames = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    if not frames:
        raise SystemExit(f"no frames in {folder}")
    n = len(frames)
    expected = int(frames[-1][:4]) + 1
    if n != expected:
        raise SystemExit(f"{folder}: {n} frames but numbering ends at {expected - 1} (missing frames)")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(folder, "%04d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    print(f"wrote {out} ({n} frames, {n / fps:.2f} s)")


if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else "service"
    only = sys.argv[2:]
    base = os.path.join(FILM, "build", group)
    for shot in sorted(os.listdir(base)):
        folder = os.path.join(base, shot)
        if os.path.isdir(folder) and (not only or shot in only):
            encode(folder, os.path.join(base, f"{shot}.mp4"))
