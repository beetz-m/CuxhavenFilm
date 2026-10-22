"""Assemble the Cuxhaven film from the edit decision list (film/edl.py): clips, stills with slow camera moves, title
cards, chapter titles and fact captions, crossfades, and the procedural score.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python film/assemble.py [--preview]   (--preview: 960x540, fast)
Writes build/cuxhaven_film.mp4 (1920x1080, 30 fps, H.264 + AAC) and build/contact_sheet.jpg (one frame per 8 s).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edl as EDL  # noqa: E402
import soundtrack  # noqa: E402
import titles  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
FPS = 30
PREVIEW = "--preview" in sys.argv
W, H = (960, 540) if PREVIEW else (1920, 1080)
FF = imageio_ffmpeg.get_ffmpeg_exe()


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# %% sources


def clip_frames(shot):
    """Frames of a video segment or a clean frame sequence: trimmed, retimed, cropped (normalised x, y, w, h), graded,
    blended up to the film's frame rate and scaled to fill the frame."""
    t0, t1 = shot["in"], shot["out"]
    speed = shot.get("speed", 1.0)
    crop = shot.get("crop")
    vf = []
    if crop:
        x, y, w, h = crop
        vf.append(f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}")
    if "seq" in shot:                 # clean frames (no burned-in caption bar), numbered from 0 at src_fps
        src_fps = shot["src_fps"]
        inp = ["-framerate", f"{src_fps * speed}", "-start_number", str(int(round(t0 * src_fps))),
               "-i", os.path.join(shot["seq"], shot.get("pattern", "%05d.jpg"))]
        vf += [f"framerate=fps={FPS}"]
    else:
        inp = ["-ss", f"{t0}", "-i", shot["clip"]]
        vf += [f"setpts=PTS/{speed}", f"fps={FPS}"]
    if shot.get("grade"):
        vf.append(shot["grade"])
    vf += [f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={W}:{H}", "setsar=1"]
    cmd = [FF, "-v", "error"] + inp + ["-t", f"{(t1 - t0) / speed}", "-vf", ",".join(vf),
                                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    path = shot.get("seq", shot.get("clip"))
    n = frames_of(shot)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)   # stopped early on purpose
    last = None
    for _ in range(n):
        buf = proc.stdout.read(W * H * 3)
        if len(buf) == W * H * 3:
            last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        if last is None:
            raise RuntimeError(f"no frames from {path} at {t0}")
        yield last                     # hold the last frame if the source is a few frames short
    proc.kill()
    proc.stdout.close()
    proc.wait()


def still_frames(shot):
    """A still with a slow eased zoom and pan (Ken Burns)."""
    im = Image.open(shot["still"]).convert("RGB")
    z0, z1 = shot.get("zoom", (1.0, 1.12))
    (cx0, cy0), (cx1, cy1) = shot.get("pan", ((0.5, 0.5), (0.5, 0.5)))
    iw, ih = im.size
    base = min(iw / W, ih / H)            # the crop that fills 16:9 at zoom 1
    n = frames_of(shot)
    for i in range(n):
        u = ease(i / max(1, n - 1))
        z = z0 + (z1 - z0) * u
        cw, ch = W * base / z, H * base / z
        cx = (cx0 + (cx1 - cx0) * u) * iw
        cy = (cy0 + (cy1 - cy0) * u) * ih
        x0 = min(max(0.0, cx - cw / 2), iw - cw)
        y0 = min(max(0.0, cy - ch / 2), ih - ch)
        yield np.asarray(im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch)))


def card_frames(shot):
    img = shot["card"]()
    arr = np.asarray(img.convert("RGB").resize((W, H), Image.LANCZOS))
    for _ in range(frames_of(shot)):
        yield arr


def frames_of(shot):
    if "clip" in shot or "seq" in shot:
        return int(round((shot["out"] - shot["in"]) / shot.get("speed", 1.0) * FPS))
    return int(round(shot["dur"] * FPS))


def source(shot):
    if "clip" in shot or "seq" in shot:
        return clip_frames(shot)
    if "still" in shot:
        return still_frames(shot)
    return card_frames(shot)


# %% overlays


def rgba(img):
    a = np.asarray(img.resize((W, H), Image.LANCZOS) if img.size != (W, H) else img).astype(np.float32) / 255.0
    return a[..., :3] * a[..., 3:4], a[..., 3:4]          # premultiplied colour, alpha


def overlays(shot):
    """[(premult, alpha, t_in, t_out)] in seconds from the shot start."""
    out = []
    dur = frames_of(shot) / FPS
    if shot.get("chapter"):
        num, title, kicker = shot["chapter"]
        out.append((*rgba(titles.chapter(num, title, kicker)), 0.5, min(dur - 0.3, 5.5)))
    if shot.get("fact"):
        text, keys, src = shot["fact"]
        t = shot.get("fact_at", 1.2)
        out.append((*rgba(titles.fact(text, keys, src)), t, min(dur - 0.3, t + shot.get("fact_dur", 5.0))))
    for mult, a, b in shot.get("timelapse", ()):        # source seconds where the twin itself runs as a time-lapse
        sp = shot.get("speed", 1.0)
        t0, t1 = max(0.0, (a - shot["in"]) / sp), min(dur, (b - shot["in"]) / sp)
        if t1 - t0 > 0.8:
            out.append((*rgba(titles.badge(f"TIME-LAPSE ×{mult * sp:g}")), t0, t1))
    if "card" not in shot:
        out.append((*rgba(WATERMARK), 0.0, dur + 1.0))
    return out


def composite(frame, ovs, t, fade=0.45):
    if not ovs:
        return frame
    f = frame.astype(np.float32) / 255.0
    for pre, a, t0, t1 in ovs:
        k = min(ease((t - t0) / fade), ease((t1 - t) / fade))
        if k <= 0:
            continue
        f = f * (1 - a * k) + pre * k
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


# %% assembly


def main():
    global WATERMARK
    WATERMARK = titles.watermark()
    shots = EDL.shots()
    os.makedirs(BUILD, exist_ok=True)
    video = os.path.join(BUILD, "film_video.mp4" if not PREVIEW else "film_preview_video.mp4")
    writer = imageio_ffmpeg.write_frames(video, (W, H), fps=FPS, codec="libx264", macro_block_size=1,
                                         output_params=["-crf", "18" if not PREVIEW else "26", "-preset",
                                                        "medium" if not PREVIEW else "veryfast", "-pix_fmt", "yuv420p"])
    writer.send(None)
    held = []                 # tail frames of the previous shot, waiting to be crossfaded
    t_film = 0.0
    cuts, starts, sheet = [], [], []
    started = time.time()
    for k, shot in enumerate(shots):
        n = frames_of(shot)
        xf = int(shot.get("xfade", 0.6) * FPS) if k > 0 else 0
        xf = min(xf, len(held))
        nxt = int(shots[k + 1].get("xfade", 0.6) * FPS) if k + 1 < len(shots) else 0
        ovs = overlays(shot)
        start_time = t_film - xf / FPS
        starts.append(start_time)
        if shot.get("chapter"):
            cuts.append(start_time)
        for i, fr in enumerate(source(shot)):
            fr = composite(fr, ovs, i / FPS)
            if i < xf:                                    # crossfade with the held tail of the previous shot
                u = ease((i + 1) / (xf + 1))
                fr = (held[i].astype(np.float32) * (1 - u) + fr.astype(np.float32) * u).astype(np.uint8)
            if i >= n - nxt and k + 1 < len(shots):       # hold back this shot's tail for the next crossfade
                held_next = held_next if i > n - nxt else []
                held_next.append(fr)
                continue
            writer.send(np.ascontiguousarray(fr))
            t_film += 1 / FPS
            if int(t_film * FPS) % (8 * FPS) == 0:
                sheet.append(Image.fromarray(fr).resize((384, 216)))
        held = held_next if k + 1 < len(shots) and nxt > 0 else []
        held_next = []
        print(f"[{k + 1}/{len(shots)}] {shot.get('name', '')} {n / FPS:.1f}s  t={t_film:.1f}s  "
              f"{time.time() - started:.0f}s elapsed", flush=True)
    writer.close()

    duration = t_film
    pulse = [(starts[i], starts[i + 1] if i + 1 < len(starts) else duration)
             for i, s in enumerate(shots) if s.get("pulse")]
    pulse = merge(pulse)
    wav = os.path.join(BUILD, "score.wav")
    soundtrack.write_wav(wav, soundtrack.score(duration + 0.5, cuts=cuts[1:], pulse_spans=pulse))
    final = os.path.join(BUILD, "cuxhaven_film.mp4" if not PREVIEW else "cuxhaven_film_preview.mp4")
    subprocess.run([FF, "-v", "error", "-y", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    if sheet:
        cols = 6
        rows = (len(sheet) + cols - 1) // cols
        cs = Image.new("RGB", (cols * 384, rows * 216))
        for i, im in enumerate(sheet):
            cs.paste(im, ((i % cols) * 384, (i // cols) * 216))
        cs.save(os.path.join(BUILD, "contact_sheet.jpg" if not PREVIEW else "contact_sheet_preview.jpg"), quality=85)
    print(f"wrote {final}: {duration:.1f} s, {len(shots)} shots, {len(cuts)} chapters, "
          f"{time.time() - started:.0f} s to build")


def merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 0.1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


if __name__ == "__main__":
    main()
