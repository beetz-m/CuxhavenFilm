"""Export the twin's vehicle lanes and station envelopes (read-only, by path) for the hall-clip renderer.
Run: ~/Documents/LabAutomationTwin/env_cad/bin/python film/hall/export_lanes.py  -> build/hall/lanes.json"""
import json
import os
import sys

TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
sys.path.insert(0, os.path.join(TWIN, "site"))
sys.path.insert(0, TWIN)
import layout as L  # noqa: E402
import traffic as TR  # noqa: E402

out = {"floor_z": getattr(L, "FLOOR_Z", 0.0), "hall": L.HALL, "lanes": [], "stations": {}}
for l in TR.lanes():
    if l["kind"] in ("agv_return", "logistics_axis", "cross_aisle"):
        out["lanes"].append(dict(id=l["id"], kind=l["kind"], line=[list(map(float, p)) for p in l["line"]],
                                 width=float(l["width"])))
for s in L.STATIONS:
    try:
        env = TR.station_envelope(s["id"])
        out["stations"][s["id"]] = [list(map(float, c)) for c in env.exterior.coords]
    except Exception as e:  # noqa: BLE001
        out["stations"][s["id"]] = None
os.makedirs("build/hall", exist_ok=True)
json.dump(out, open("build/hall/lanes.json", "w"), indent=1)
print(len(out["lanes"]), "lanes;", sum(v is not None for v in out["stations"].values()), "station envelopes; floor", out["floor_z"])
for l in out["lanes"]:
    print(l["id"], l["kind"], l["line"], l["width"])
