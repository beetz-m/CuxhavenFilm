"""Negative control for check_plan.py: movers placed through the G3 station and into a hall column must be reported."""
import json
import os
import sys

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_plan as CP  # noqa: E402

TWIN = CP.TWIN
sys.path.insert(0, os.path.join(TWIN, "site"))
import layout as L  # noqa: E402

key = "G3_24_40"
plan = json.load(open(CP.PLAN))[key]
st = {s["id"]: s for s in L.STATIONS}["G3"]
v = dict(plan["vehicles"][0])
v.update(base=[st["x"] - 20.0, st["y"]], d=[1.0, 0.0], heading=0.0, speed=2.0)     # drives through the G3 deck
H = L.HALL
y0 = L.BAYS[0]["y0"]
j = round((st["x"] - H["x0"]) / L.FRAME_PITCH)
i = min(range(L.N_BAYS + 1), key=lambda i: abs(y0 + i * L.BAY_W - st["y"]))
c0 = (H["x0"] + j * L.FRAME_PITCH, y0 + i * L.BAY_W)                              # a hall column near G3
p = dict(plan["people"][0])
p.update(pos=[c0[0], c0[1], 0.0])                                                  # stands inside the column
CP.write_stage("NEG_" + key, "G3", plan, [p], [v])
import shutil  # noqa: E402
hits = CP.CC.check("NEG_" + key)["collisions"]
mine = CP.mine(hits)
print("NEGATIVE CONTROL film-mover hits:", {k: sorted(set(x.split("/")[2] + "/" + x.split("/")[3] for x in v))[:4]
                                            for k, v in mine.items()})
print("person in column at", c0)
