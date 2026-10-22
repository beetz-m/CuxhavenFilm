"""Lens and sensor pass for rendered frame sequences: the twin's scripts/realism.py post() (read-only) with the shared
defaults (grain 0.012, vignette 0.22, chromatic aberration 0.8 px, bloom 0.10) and seed = frame index, so the grain
changes from frame to frame and ship and hall clips match.

  ~/Documents/LabAutomationTwin/env_cad/bin/python film/post_frames.py <raw dir> [<raw dir> ...]
Each <...>/ships_raw/<seq>/ (or any dir with NNNNN.jpg) is written to the sibling 'ships' tree (raw -> final), JPEG 95.
"""

import os
import sys
import tempfile
from multiprocessing import Pool

TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
sys.path.insert(0, os.path.join(TWIN, "scripts"))


def out_dir(d):
    d = os.path.abspath(d)
    for a, b in (("ships_raw", "ships"), ("ships_test_raw", "ships_test")):
        if f"/{a}/" in d + "/":
            return d.replace(f"/{a}", f"/{b}", 1)
    return d + "_post"


def one(job):
    src, dst, k = job
    import realism as RZ
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        RZ.post(src, tmp, seed=k)
        Image.open(tmp).save(dst, quality=95)
    finally:
        os.remove(tmp)
    return dst


def main(dirs):
    jobs = []
    for d in dirs:
        od = out_dir(d)
        os.makedirs(od, exist_ok=True)
        for f in sorted(os.listdir(d)):
            if f.endswith(".jpg"):
                jobs.append((os.path.join(d, f), os.path.join(od, f), int(f.split(".")[0])))
            elif f == "fps.txt":
                open(os.path.join(od, f), "w").write(open(os.path.join(d, f)).read())
    with Pool(min(8, os.cpu_count() or 4)) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if i % 200 == 0:
                print(f"[post] {i}/{len(jobs)}", flush=True)
    print(f"[post] {len(jobs)} frames", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
