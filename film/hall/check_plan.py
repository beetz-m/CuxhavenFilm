"""Collision-check the film's hall movers (vehicles, workers) and re-plan until they are collision free.

For every shot of build/hall/plan.json (written by film/render_hall_clips.py --test: the chosen people and vehicles
plus all visible, load-clear candidates) a film-side check stage is written to build/hall/coll/<shot>.usda: it
sublayers the twin's process scene, deactivates what the renderer deactivates (the twin's worker figures, its
hand-built forklifts and tuggers) and adds the movers as proxy meshes (vehicle envelope incl. its load, a worker
envelope per posture), visible only during the shot. The twin's checker (scripts/collision_check.py, python-fcl,
triangle-exact, loaded read-only by path) then tests them against all static geometry and the moving loads. A
colliding mover is replaced by the next candidate (another lane / direction / work point) or dropped. Resting
contact (wheels, feet) is filtered by the checker's 5 cm lift.

Writes film/hall/final/<shot>.json (replayed by the renderer) and build/hall/coll/report_<shot>.json; shots without
film movers are written without a check. Shots are independent, so run them in parallel (one process per shot).
Run: ~/Documents/CuxhavenSiteTwin/.venv_coll/bin/python film/hall/check_plan.py [shot ...]
Env HALL_PROCESS_SET=process|process_next selects the twin's process set (as for the renderer).
"""

import importlib.util
import json
import math
import os
import sys

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

FILM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sets as SETS  # noqa: E402

NEXT = SETS.use_next(sys.argv)
SUFFIX = "_next" if NEXT else ""
PROCESS_SET = "process_next" if NEXT else "process"
COLL = os.path.join(FILM, "build", "hall", "coll" + SUFFIX)
PLAN = os.path.join(FILM, "build", "hall", f"plan{SUFFIX}.json")
FINAL = os.path.join(FILM, "film", "hall", "final" + SUFFIX)
os.environ["COLL_PROC_DIR"], os.environ["COLL_OUT"] = COLL, COLL
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
DT = 0.5           # a vehicle moves <= 0.9 m per step, less than its 1.4-3.1 m length: the sweep has no gaps

spec = importlib.util.spec_from_file_location("twin_cc", os.path.join(TWIN, "scripts", "collision_check.py"))
CC = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]                       # the checker parses argv at import
spec.loader.exec_module(CC)
CC.PROC, CC.OUT, CC.DT = COLL, COLL, DT


def open_masked(path):
    """As the checker's own, for a film-side layer that sublayers a process scene: hall content + process + movers."""
    lay = Sdf.Layer.FindOrOpen(SETS.site_layer(NEXT))
    names = [c.name for c in lay.GetPrimAtPath("/World").nameChildren if c.name not in CC.SKIP_TOP]
    mask = Usd.StagePopulationMask()
    for n in names + ["Process", "Film"]:
        mask.Add(Sdf.Path(f"/World/{n}"))
    return Usd.Stage.OpenMasked(path, mask)


CC.open_masked = open_masked

# proxy envelopes in the mover's own frame (x forward): (x0, x1, y0, y1, z0, z1)
VEHICLE_BOX = {"stacker": (-2.15, 0.89, -0.60, 0.60, 0.0, 2.93),     # forklift_b incl. forks, mast, pallet + bins
               "agv": (-1.04, 0.40, -0.42, 0.42, 0.0, 0.65)}         # iw.hub + pallet + bins
PERSON_BOX = {"crouch": (-0.30, 0.45, -0.30, 0.30, 0.0, 1.15), "stand": (-0.25, 0.30, -0.30, 0.30, 0.0, 1.85)}
YAW0 = {"stacker": 0.0, "agv": 0.0}


def box_mesh(stage, path, b):
    x0, x1, y0, y1, z0, z1 = b
    P = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    F = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(P))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in F for i in f]))


def posture(motion):
    return "crouch" if motion.startswith("Market_Sales") else "stand"


def write_stage(key, sid, plan, people, vehicles):
    src = SETS.process_layer(sid, NEXT)
    base = Usd.Stage.Open(src)
    path = os.path.join(COLL, f"{key}.usda")
    lay = Sdf.Layer.CreateNew(path)
    lay.subLayerPaths.append(src)
    st = Usd.Stage.Open(lay)
    tcps = base.GetTimeCodesPerSecond()
    st.SetTimeCodesPerSecond(tcps)
    st.SetStartTimeCode(base.GetStartTimeCode())
    st.SetEndTimeCode(min(base.GetEndTimeCode(), plan["t1"] * tcps))      # the checker samples 0 .. end
    st.OverridePrim(f"/World/Process/{sid}/Workers").SetActive(False)
    fleet = base.GetPrimAtPath("/World/Logistics/Fleet")
    for c in fleet.GetChildren() if fleet else []:
        if c.GetName().startswith(("FL3", "FL16", "Tugger")):
            st.OverridePrim(str(c.GetPath())).SetActive(False)
    t0, t1 = plan["t0"], plan["t1"]
    times = np.arange(t0, t1 + 1e-6, DT)

    def mover(name, poses, box):
        """A proxy mover; outside the shot window it is parked 1 km below the floor (the checker builds every mover's
        geometry at t = 0 from VISIBLE prims, so it must not be hidden)."""
        x = UsdGeom.Xform.Define(st, f"/World/Film/Movers/{name}")
        tr, rz = x.AddTranslateOp(), x.AddRotateZOp()
        park = (0.0, 0.0, -1000.0)
        tr.Set(park, 0.0)
        rz.Set(0.0, 0.0)
        tr.Set(park, t0 * tcps - 0.01)
        rz.Set(0.0, t0 * tcps - 0.01)
        for t, (p, yaw) in poses:
            tr.Set(tuple(map(float, p)), t * tcps)
            rz.Set(float(yaw), t * tcps)
        tr.Set(park, t1 * tcps + 0.01)
        box_mesh(st, f"/World/Film/Movers/{name}/Box", box)

    for k, q in enumerate(people):          # static people: two identical samples make them movers
        mover(f"P{k}", [(t0, (q["pos"], q["yaw"])), (t1, (q["pos"], q["yaw"]))], PERSON_BOX[posture(q["motion"])])
    for k, v in enumerate(vehicles):
        b, d = np.array(v["base"]), np.array(v["d"])
        poses = [(t, (np.r_[b + d * v["speed"] * (t - v["tmid"]), 0.0], v["heading"] + YAW0[v["kind"]]))
                 for t in times]
        mover(f"V{k}", poses, VEHICLE_BOX[v["kind"]])
    lay.Save()
    return path


def person_fix(key, sid, plan, q):
    """Fast static pre-check of one worker (fcl, one query per candidate against the station geometry within 3 m):
    steps of 0 .. 1.0 m towards the part, standing first, crouching when standing does not fit (low decks). Returns
    the first collision-free placement, or None. The full swept check still verifies the result."""
    import fcl
    path = write_stage(key + "_probe", sid, plan, [], [])
    st = CC.open_masked(path)
    p0 = np.array(q["pos"], float)
    near = []
    bc = UsdGeom.BBoxCache(0, ["default", "render"])
    for top in st.GetPrimAtPath("/World").GetChildren():          # the checker's static groups
        if top.GetName() in CC.SKIP_TOP or top.GetName() in ("Process", "Film"):
            continue
        for c in top.GetChildren():
            kids = list(c.GetChildren())
            for g in (kids if kids and top.GetName() in ("Production", "Equipment", "Cells", "Storage") else [c]):
                r = bc.ComputeWorldBound(g).ComputeAlignedRange()
                if r.IsEmpty() or not g.IsActive():
                    continue
                lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
                if np.all(lo[:2] - 3 < p0[:2]) and np.all(p0[:2] < hi[:2] + 3) and lo[2] < p0[2] + 3 and hi[2] > p0[2] - 1:
                    gg = CC.gather(st, g, Usd.TimeCode(0), np.eye(4))
                    if gg is not None and len(gg[1]):
                        near.append(fcl.CollisionObject(CC.bvh(*gg), fcl.Transform()))
    f = np.array([math.cos(math.radians(q["yaw"])), math.sin(math.radians(q["yaw"])), 0.0])
    stand = posture(q["motion"]) == "stand"
    for post in (["stand", "crouch"] if stand else ["crouch"]):
        x0, x1, y0, y1, z0, z1 = PERSON_BOX[post]
        P = np.array([(x, y, z) for z in (z0 + CC.LIFT, z1) for y in (y0, y1) for x in (x0, x1)], float)
        F = np.array([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3),
                      (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)])
        for step in np.arange(0.0, 1.01, 0.1):
            pos = p0 + f * step
            c, sn = math.cos(math.radians(q["yaw"])), math.sin(math.radians(q["yaw"]))
            Pw = P @ np.array([[c, sn, 0], [-sn, c, 0], [0, 0, 1]]) + pos
            m = fcl.BVHModel()
            m.beginModel(len(Pw), len(F))
            m.addSubModel(Pw, F)
            m.endModel()
            o = fcl.CollisionObject(m, fcl.Transform())
            if not any(fcl.collide(o, n, fcl.CollisionRequest(), fcl.CollisionResult()) for n in near):
                motion = q["motion"] if post == posture(q["motion"]) else "Market_Sales_SortOut_M"
                return dict(q, pos=pos.tolist(), motion=motion, shift_m=round(float(step), 2), posture=post)
    return None


def mine(hits):
    """Hits that involve the film movers: {mover name: [other, ...]}."""
    out = {}
    for h in hits:
        for a, b in ((h["mover"], h["other"]), (h["other"], h["mover"])):
            if a.startswith("/World/Film/Movers/"):
                out.setdefault(a.split("/")[4], []).append(b)
    return out


def check(key):
    return CC.check(key)["collisions"]


def main():
    os.makedirs(COLL, exist_ok=True)
    os.makedirs(FINAL, exist_ok=True)
    plans = json.load(open(PLAN))
    for key, plan in plans.items():
        if ONLY and key not in ONLY:
            continue
        sid = key.split("_")[0]
        people = [dict(q) for q in plan["people"]]
        vehicles = [dict(v) for v in plan["vehicles"]]
        cands_p = [c for c in plan["candidates"]["people"]
                   if not any(np.allclose(c["p"], q["pos"]) for q in people)]
        cands_v = [c for c in plan["candidates"]["vehicles"]
                   if not any(c["lane"] == v["lane"] and c["direction"] == v.get("direction") and c["kind"] == v["kind"]
                              for v in vehicles)]
        replaced, dropped = [], []
        hits = []
        fixed = []
        for q in people:                              # fast static pre-check / placement of every worker
            r = person_fix(key, sid, plan, q)
            if r is None:
                dropped.append(("person", q["person"], "no free placement"))
            else:
                if r["shift_m"] or r["motion"] != q["motion"]:
                    replaced.append(("person", q["person"], f"stepped {r['shift_m']} m, {r['posture']}"))
                fixed.append(r)
        people = fixed
        if not people and not vehicles:
            json.dump(dict(people=[], vehicles=[]), open(os.path.join(FINAL, f"{key}.json"), "w"), indent=1)
            json.dump(dict(people=0, vehicles=0, checked=False, note="no film movers"),
                      open(os.path.join(COLL, f"report_{key}.json"), "w"), indent=1)
            print(f"[{key}] no film movers", flush=True)
            continue
        for it in range(8):
            write_stage(key, sid, plan, people, vehicles)
            hits = check(key)
            bad = mine(hits)
            print(f"[{key}] iteration {it}: film-mover hits {bad if bad else 'none'}", flush=True)
            if not bad:
                break
            for name in sorted(bad, reverse=True):
                k = int(name[1:])
                if name[0] == "V":
                    lanes = {v["lane"] for j, v in enumerate(vehicles) if j != k}
                    nxt = next((c for c in cands_v if c["lane"] not in lanes), None)
                    if nxt is not None:
                        cands_v.remove(nxt)
                        replaced.append((vehicles[k]["kind"], vehicles[k]["lane"], "->", nxt["kind"], nxt["lane"]))
                        vehicles[k] = dict(nxt)
                    else:
                        dropped.append(("vehicle", vehicles[k]["lane"]))
                        vehicles.pop(k)
                else:
                    # the twin's stand points can sit on a deck edge (inside its railing): first step towards the
                    # part (the worker's facing direction), then try another work point
                    q = people[k]
                    tried = q.setdefault("shift_m", 0.0)
                    if tried < 0.45:
                        step = 0.25 if tried == 0.0 else 0.20
                        f = np.array([math.cos(math.radians(q["yaw"])), math.sin(math.radians(q["yaw"])), 0.0])
                        q["pos"] = (np.array(q["pos"]) + f * step).tolist()
                        q["shift_m"] = tried + step
                        replaced.append(("person", q["person"], f"stepped {q['shift_m']:.2f} m towards the part"))
                        continue
                    others = [q for j, q in enumerate(people) if j != k]
                    nxt = next((c for c in cands_p
                                if all(np.linalg.norm(np.array(c["p"][:2]) - np.array(q["pos"][:2])) >= 2.5
                                       for q in others)), None)
                    if nxt is not None:
                        cands_p.remove(nxt)
                        replaced.append(("person", people[k]["person"], "->", [round(v, 1) for v in nxt["p"]]))
                        people[k] = dict(people[k], pos=nxt["p"], yaw=nxt["yaw"], shift_m=0.0)
                    else:
                        dropped.append(("person", people[k]["person"]))
                        people.pop(k)
        else:
            raise SystemExit(f"{key}: still colliding after 8 re-plans")
        twin_pairs = [h for h in hits if not (h["mover"].startswith("/World/Film/")
                                              or h["other"].startswith("/World/Film/"))]
        json.dump(dict(people=people, vehicles=vehicles), open(os.path.join(FINAL, f"{key}.json"), "w"), indent=1)
        json.dump(dict(people=len(people), vehicles=len(vehicles), checked=True, film_mover_hits=0,
                       iterations=it + 1, replaced=replaced, dropped=dropped, process_set=PROCESS_SET,
                       twin_colliding_pairs_in_window=len(twin_pairs)),
                  open(os.path.join(COLL, f"report_{key}.json"), "w"), indent=1)
        print(f"[{key}] clean: {len(people)} people, {len(vehicles)} vehicles; twin's own colliding pairs "
              f"{len(twin_pairs)}", flush=True)


if __name__ == "__main__":
    main()
