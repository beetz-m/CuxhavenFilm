"""Lens and sensor post effects for the re-rendered hall frames: the twin's scripts/realism.py post() (read only, by
path) with its shared defaults (grain 0.012, vignette 0.22, chromatic aberration 0.8 px, bloom 0.10) so hall and ship
clips match; the grain is seeded with the frame index so it changes from frame to frame.

build/hall[_next]/<shot>/%05d.jpg -> build/hall_post/<shot>/%05d.jpg (quality 95, meta.json copied)
Run: ~/Documents/LabAutomationTwin/env_cad/bin/python film/hall/post_frames.py [--next] [--test] [shot ...] [--jobs 6]
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool

from PIL import Image

FILM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
ARGS = sys.argv[1:]
SRC = os.path.join(FILM, "build", "hall_next" if "--next" in ARGS else "hall")
DST = os.path.join(FILM, "build", "hall_post")
JOBS = int(ARGS[ARGS.index("--jobs") + 1]) if "--jobs" in ARGS else 6
ONLY = [a for i, a in enumerate(ARGS) if not a.startswith("--") and (i == 0 or ARGS[i - 1] != "--jobs")]

_spec = importlib.util.spec_from_file_location("twin_realism", os.path.join(TWIN, "scripts", "realism.py"))
RZ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RZ)


def one(job):
    src, dst, idx = job
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        RZ.post(src, tmp.name, seed=idx)          # defaults: grain 0.012, vignette 0.22, CA 0.8 px, bloom 0.10
        Image.open(tmp.name).convert("RGB").save(dst, quality=95)
    os.unlink(tmp.name)


def main_test():
    """--test: the test frames build/hall/test[_next]/<shot>_<j>.jpg -> build/hall/test[_next]_post/ (seed = j)."""
    src = os.path.join(FILM, "build", "hall", "test_next" if "--next" in ARGS else "test")
    dst = src + "_post"
    os.makedirs(dst, exist_ok=True)
    jobs = [(os.path.join(src, f), os.path.join(dst, f), int(f[:-4].split("_")[-1]))
            for f in sorted(os.listdir(src)) if f.endswith(".jpg") and f[:-4].split("_")[-1].isdigit()
            and (not ONLY or any(f.startswith(o + "_") for o in ONLY))]
    with Pool(JOBS) as pool:
        list(pool.imap_unordered(one, jobs))
    print(f"[post] {len(jobs)} test frames -> {dst}")


def main():
    if "--test" in ARGS:
        return main_test()
    shots = [d for d in sorted(os.listdir(SRC)) if os.path.exists(os.path.join(SRC, d, "meta.json"))]
    shots = [s for s in shots if not ONLY or s in ONLY]
    jobs = []
    for s in shots:
        os.makedirs(os.path.join(DST, s), exist_ok=True)
        shutil.copy(os.path.join(SRC, s, "meta.json"), os.path.join(DST, s, "meta.json"))
        frames = sorted(f for f in os.listdir(os.path.join(SRC, s)) if f.endswith(".jpg"))
        jobs += [(os.path.join(SRC, s, f), os.path.join(DST, s, f), int(f[:-4])) for f in frames]
    with Pool(JOBS) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
            if i % 500 == 0:
                print(f"[post] {i}/{len(jobs)}", flush=True)
    print(f"[post] {len(jobs)} frames, {len(shots)} shots -> {DST}")


if __name__ == "__main__":
    main()
