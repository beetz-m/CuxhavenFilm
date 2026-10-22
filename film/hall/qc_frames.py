"""QC of rendered hall frames: per shot the mean absolute difference between consecutive frames (downscaled, grey);
spikes above 4x the shot's median (and > 3 grey levels) are flagged as pops. Prints a table; writes build/hall/qc_next.json.
Run: ~/Documents/LabAutomationTwin/env_cad/bin/python film/hall/qc_frames.py [dir]"""
import json
import os
import sys

import numpy as np
from PIL import Image

D = sys.argv[1] if len(sys.argv) > 1 else "build/hall_post"
W = json.load(open("film/hall/windows_next.json"))
rep = {}
for k, w in W.items():
    if k.startswith("_"):
        continue
    key = f"{w['sid']}_{w['new'][0]}_{w['new'][1]}"
    d = os.path.join(D, key)
    if not os.path.isdir(d):
        continue
    fs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    prev, diffs = None, []
    for f in fs:
        a = np.asarray(Image.open(os.path.join(d, f)).convert("L").resize((240, 135)), np.float32)
        if prev is not None:
            diffs.append(float(np.abs(a - prev).mean()))
        prev = a
    diffs = np.array(diffs)
    med = float(np.median(diffs))
    spikes = [(i + 1, round(float(x), 2)) for i, x in enumerate(diffs) if x > max(4 * med, 3.0)]
    first = json.load(open(os.path.join(d, "meta.json")))["first_second"]
    rep[key] = dict(frames=len(fs), median_diff=round(med, 2), max_diff=round(float(diffs.max()), 2),
                    pops=[(i, round(first + i / 24, 2), x) for i, x in spikes])
    print(f"{key:10s} {len(fs):4d} fr  median {med:5.2f}  max {diffs.max():5.2f}  pops {rep[key]['pops'][:6]}")
json.dump(rep, open("build/hall/qc_next.json", "w"), indent=1)
