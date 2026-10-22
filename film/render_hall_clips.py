"""Re-render the film's hall process shots with few, varied, animated workers and realistic vehicles.

The Cuxhaven twin is read only: its process stages (scenes/process/<sid>.usda, 24 fps time codes, authored camera
/World/Process/<sid>/Camera) are opened by path and every change is made in the SESSION layer:
  * the twin's six identical worker figures per station and its hand-built forklift / tugger fleet are deactivated;
  * at most two workers from the Isaac 6.2 people library stand at the most visible of the twin's work points, each a
    different person, animated with a Reallusion motion (101-joint skeleton, same as the characters) converted to the
    stage's time codes and units (the motions are in cm, the construction characters in m) with its own start offset;
  * in about half of the shots one vehicle (at most two) drives along a hall lane that the camera sees (lanes from the
    twin's traffic model, build/hall/lanes.json): an autonomous stacker (Forklift B) with a pallet of KLT bins, or an
    iw.hub AGV with a load; the path is checked against the station's moving loads.
Rendering follows the twin's scripts/render_process.py (RaytracedLighting, interior exposure) at 1920x1080, 24 fps,
only the ranges the film uses, with a 1 s margin on each side (frame 0 = second max(0, in - 1); see meta.json).

  OMNI_KIT_ACCEPT_EULA=YES $I film/render_hall_clips.py [--test] [shot ...]     shot = <sid>_<in>_<out>
  --test   three frames per shot (start, mid, end) -> build/hall/test[_next]/<shot>_<j>.jpg (plan -> plan[_next].json)
  --next   the twin's staged set; shot windows from film/hall/windows_next.json, per-shot camera offsets and look
           settings from film/hall/overrides.json ("cam": {"up", "fwd", "pitch_down"} in m / deg)
Env: REALISM_VERSION (default v2), HALL_ISO / HALL_AMBIENT (interior exposure and flat fill), HALL_CAST_FIX=0 (keep
the twin's metallic cast primer), HALL_SWEEP="iso:ambient,..." (test only: one mid frame per setting)
Wrap every run in `flock <scratchpad>/gpu.lock` (one Isaac GPU job at a time).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("REALISM_VERSION", "v2")
ARGS = [a for a in sys.argv[1:]]
TEST = "--test" in ARGS
ONLY = [a for a in ARGS if not a.startswith("--")]
RES = (1920, 1080)
FPS = 24
FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(FILM, "film", "hall"))
# which of the twin's process sets to render: the current one, or the twin session's fixed staged set (--next or
# HALL_PROCESS_SET=process_next; film/hall/sets.py)
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
OUT = os.path.join(FILM, "build", "hall")      # lanes.json, plans and test frames; frames of a set: OUT_SET
ISAAC = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/"

# the hall shots of film/edl.py: (sid, in, out) in video seconds of the twin's clip
SHOTS = [("G2", 12, 24), ("G3", 24, 40), ("G3", 48, 56), ("G4", 39, 47), ("G5", 15, 27), ("B7", 0, 9),
         ("B2", 60, 72), ("B4", 39, 48), ("B5", 48, 56), ("H2", 36, 44), ("H3", 52, 60), ("H6", 12, 22),
         ("M1", 15, 27), ("T1", 12, 20)]
WINDOWS_NEXT = os.path.join(FILM, "film", "hall", "windows_next.json")   # film/hall/pick_windows.py
if "--next" in ARGS or os.environ.get("HALL_PROCESS_SET") == "process_next":
    SHOTS = [(w["sid"], *w["new"]) for k, w in json.load(open(WINDOWS_NEXT)).items() if not k.startswith("_")]
MARGIN = 1.0

# people: (name, usd, units scale of the character's skeleton relative to the cm motions, prim scale)
PEOPLE = [
    ("construction_01", "People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd", 0.01, 1.0),
    ("worker_overall", "People/Characters/Reallusion/Worker/Worker.usd", 1.0, 0.01),
    ("construction_05", "People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd", 0.01, 1.0),
    ("construction_02", "People/Characters/original_male_adult_construction_02/male_adult_construction_02.usd", 0.01, 1.0),
]
MOTION_DIR = "People/Characters/Reallusion/Worker/Motions/"
MOTIONS_FOR_POSE = {  # first choice, then alternatives so two workers in one shot never move alike
    "kneel": ["Market_Sales_SortOut_M", "Market_Sales_Assisting_M"],
    "tool": ["Kitchen_CleanTable_M", "StandingDiscussion_LookingDown_M"],
    "reach": ["StandingDiscussion_LookingDown_M", "Kitchen_CleanTable_M"],
    "stand": ["StandingDiscussion_LookingDown_M", "Kitchen_CleanTable_M"],
    "walk": ["StandingDiscussion_LookingDown_M", "Kitchen_CleanTable_M"]}
PERSON_YAW = 90.0      # deg added so the character faces the twin worker's facing direction (set after the test)

# vehicles: (name, usd, forward sign along the asset's x, speed m/s, footprint (L, W), load)
VEHICLES = {
    "stacker": dict(usd="Robots/IsaacSim/ForkliftB/forklift_b.usd", yaw0=0.0, speed=1.8, size=(3.1, 1.2), load="forks",
                    zmax=50.0),     # 2.9 m mast: >= ~70 px tall at 50 m
    "agv": dict(usd="Robots/Idealworks/iwhub/iw_hub_static.usd", yaw0=0.0, speed=1.0, size=(1.5, 0.7), load="top",
                zmax=25.0),         # low (0.25 m + load): only close to the camera
}

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": RES[0], "height": RES[1]})

import carb  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt  # noqa: E402

S = carb.settings.get_settings()
import sets as SETS  # noqa: E402

NEXT = SETS.use_next(ARGS)
SUFFIX = "_next" if NEXT else ""
OUT_SET = os.path.join(FILM, "build", "hall" + SUFFIX)     # rendered frames of this set

# realistic materials of the twin (scripts/realism.py, read only): the material library is edited IN MEMORY before any
# stage is opened, the files on disk stay untouched. Post effects (lens and sensor) run later as a CPU pass
# (film/hall/post_frames.py) with the grain seeded per frame.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("twin_realism", os.path.join(TWIN, "scripts", "realism.py"))
RZ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RZ)
if "--no-realism" not in ARGS:
    RZ.apply_materials()
    __import__("product_looks").apply()      # film-side turbine surface detail (film/product_looks.py)
    if os.environ.get("HALL_CAST_FIX", "1") != "0":
        __import__("hall_looks").cast_iron()     # castings: primer paint, not bare metal (film/hall/hall_looks.py)
ISO_IN, ISO_OUT = float(os.environ.get("HALL_ISO", 240.0)), 60.0     # sweep 2026-10-21: 240 + fill 0.4
AMBIENT = float(os.environ.get("HALL_AMBIENT", 0.3))     # flat fill (RTX scene ambient), 0 = off; 0.3 for the finals (coordinator)
SWEEP = [tuple(map(float, x.split(":"))) for x in os.environ.get("HALL_SWEEP", "").split(",") if x]
LANES = json.load(open(os.path.join(OUT, "lanes.json")))["lanes"]
PLAN_PATH = os.path.join(OUT, f"plan{SUFFIX}.json")
OVERRIDES_PATH = os.path.join(FILM, "film", "hall", f"overrides{SUFFIX}.json")
OVERRIDES = json.load(open(OVERRIDES_PATH)) if os.path.exists(OVERRIDES_PATH) else {}
FINAL_DIR = os.path.join(FILM, "film", "hall", "final" + SUFFIX)    # collision-checked plans (check_plan.py)
FINAL = {f[:-5]: json.load(open(os.path.join(FINAL_DIR, f))) for f in sorted(os.listdir(FINAL_DIR))
         if f.endswith(".json")} if os.path.isdir(FINAL_DIR) and "--replan" not in ARGS else {}


# %% camera geometry


def cam_info(stage, cam, tc):
    prim = stage.GetPrimAtPath(cam)
    M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc)
    c = UsdGeom.Camera(prim)
    f = c.GetFocalLengthAttr().Get(tc)
    ha = c.GetHorizontalApertureAttr().Get(tc)
    return np.array(M).T, f, ha          # column-vector convention


def project(info, p):
    """Normalised image coordinates (-1..1) and depth of a world point; depth <= 0 means behind the camera."""
    M, f, ha = info
    q = np.linalg.inv(M) @ np.r_[p, 1.0]
    z = -q[2]
    if z <= 0.1:
        return None, None, z
    x = f * q[0] / z / (ha / 2)
    y = f * q[1] / z / (ha * RES[1] / RES[0] / 2)
    return x, y, z


DEPTH = None      # distance-to-image-plane buffer of the unmodified scene at mid shot (occlusion test)


def unoccluded(x, y, z, slack=0.8):
    if DEPTH is None:
        return True
    h, w = DEPTH.shape
    u = int((x + 1) / 2 * (w - 1))
    v = int((1 - y) / 2 * (h - 1))
    d = DEPTH[max(0, v - 1):v + 2, max(0, u - 1):u + 2]
    return bool(np.nanmax(d) >= z - slack)


def visible(info, p, zmin=4.0, zmax=95.0, edge=0.92, occlusion=True):
    x, y, z = project(info, p)
    ok = x is not None and abs(x) < edge and abs(y) < edge and zmin < z < zmax
    return ok and (not occlusion or unoccluded(x, y, z))


def film_camera(stage, sid, a, b, ov, tcps):
    """The twin's authored camera, or a film-side copy offset by ov["cam"] = {"up": m, "fwd": m along the view,
    "pitch_down": deg} (railings in the foreground): /World/Film/Cam_<sid>, one matrix sample per frame."""
    src = f"/World/Process/{sid}/Camera"
    c = ov.get("cam")
    if not c:
        return src
    up, fwd, pitch = float(c.get("up", 0.0)), float(c.get("fwd", 0.0)), float(c.get("pitch_down", 0.0))
    path = f"/World/Film/Cam_{sid}"
    cam = UsdGeom.Camera.Define(stage, path)
    if "eye" in c:                       # absolute static camera: {"eye": [x, y, z], "target": [x, y, z], "focal": mm}
        orig = UsdGeom.Camera(stage.GetPrimAtPath(src))
        cam.CreateHorizontalApertureAttr().Set(orig.GetHorizontalApertureAttr().Get())
        cam.CreateVerticalApertureAttr().Set(orig.GetHorizontalApertureAttr().Get() * RES[1] / RES[0])
        cam.CreateFocalLengthAttr().Set(float(c.get("focal", orig.GetFocalLengthAttr().Get())))
        cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 2000.0))
        eye, tgt = Gf.Vec3d(*c["eye"]), Gf.Vec3d(*c["target"])
        V = Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1))
        cam.AddTransformOp().Set(V.GetInverse())
        return path
    orig = UsdGeom.Camera(stage.GetPrimAtPath(src))
    for name in ("FocalLength", "HorizontalAperture", "VerticalAperture", "ClippingRange", "FStop",
                 "FocusDistance"):
        sa, da = getattr(orig, f"Get{name}Attr")(), getattr(cam, f"Create{name}Attr")()
        if sa and sa.HasAuthoredValue():
            ts = sa.GetTimeSamples()
            for t in ts:
                da.Set(sa.Get(t), t)
            if not ts:
                da.Set(sa.Get())
    op = cam.AddTransformOp()
    R = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -pitch))
    for fr in range(int(max(0.0, a - MARGIN - 0.5) * FPS), int((b + MARGIN + 0.5) * FPS) + 1):
        tc = fr / FPS * tcps
        M = UsdGeom.Xformable(stage.GetPrimAtPath(src)).ComputeLocalToWorldTransform(tc)
        fv = -Gf.Vec3d(M[2][0], M[2][1], M[2][2]).GetNormalized()       # camera looks down its -Z
        N = R * M
        N.SetTranslateOnly(M.ExtractTranslation() + Gf.Vec3d(0, 0, up) + fv * fwd)
        op.Set(N, tc)
    return path


# %% workers


def twin_workers(stage, sid, tc):
    """The twin's worker figures: (path, world position, facing yaw deg, pose)."""
    out = []
    root = stage.GetPrimAtPath(f"/World/Process/{sid}/Workers")
    if not root:
        return out
    for w in root.GetChildren():
        pose = "stand"
        for spec in w.GetPrimStack():
            if spec.hasReferences:
                for r in spec.referenceList.GetAddedOrExplicitItems():
                    if "worker_" in r.assetPath:
                        pose = r.assetPath.split("worker_")[1].split(".")[0]
        M = np.array(UsdGeom.Xformable(w).ComputeLocalToWorldTransform(tc)).T
        yaw = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        out.append(dict(path=str(w.GetPath()), p=M[:3, 3].tolist(), yaw=yaw, pose=pose))
    return out


def motion_samples(name, cache={}):
    """(joints, [(rot, trans, scale)] per sample, samples per second) of a Reallusion motion."""
    if name not in cache:
        st = Usd.Stage.Open(ISAAC + MOTION_DIR + name + ".usd")
        an = [UsdSkel.Animation(p) for p in st.Traverse() if p.IsA(UsdSkel.Animation)][0]
        ts = an.GetRotationsAttr().GetTimeSamples() or [0.0]
        frames = [(an.GetRotationsAttr().Get(t), an.GetTranslationsAttr().Get(t), an.GetScalesAttr().Get(t))
                  for t in ts]
        cache[name] = (an.GetJointsAttr().Get(), frames, st.GetTimeCodesPerSecond())
    return cache[name]


def ref_child(stage, parent, usd, name="Asset"):
    """Reference an asset on a child prim, so the asset's own root transform ops never clash with ours."""
    c = UsdGeom.Xform.Define(stage, f"{parent}/{name}")
    c.GetPrim().GetReferences().AddReference(ISAAC + usd)
    return c


def add_person(stage, k, person, motion, pos, yaw, t0, t1, offset_s, tcps):
    name, usd, unit, scale = person
    root = f"/World/Film/People/P{k}"
    x = UsdGeom.Xform.Define(stage, root)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddRotateZOp().Set(float(yaw + PERSON_YAW))
    x.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
    c = ref_child(stage, root, usd, "Character")
    skel = [p for p in Usd.PrimRange(c.GetPrim()) if p.IsA(UsdSkel.Skeleton)][0]
    joints, frames, mfps = motion_samples(motion)
    anim = UsdSkel.Animation.Define(stage, f"/World/Film/Anim/P{k}")
    anim.CreateJointsAttr().Set(joints)
    rot_a, tr_a, sc_a = anim.CreateRotationsAttr(), anim.CreateTranslationsAttr(), anim.CreateScalesAttr()
    n = len(frames)
    for fr in range(int(t0 * FPS), int(t1 * FPS) + 1):
        m = int(round(((fr / FPS - t0) + offset_s) * mfps)) % n
        r, t, s = frames[m]
        tc = fr / FPS * tcps
        rot_a.Set(r, tc)
        tr_a.Set(Vt.Vec3fArray.FromNumpy((np.asarray(t, dtype=np.float32) * unit).astype(np.float32)), tc)
        sc_a.Set(s, tc)
    UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    return dict(person=name, motion=motion, pos=list(map(float, pos)), yaw=float(yaw), offset_s=offset_s)


# %% vehicles


def lane_candidates(info_mid, zmax=95.0):
    """Lanes with their longest camera-visible stretch at the shot's mid time."""
    out = []
    for l in LANES:
        (ax, ay), (bx, by) = l["line"]
        L = math.hypot(bx - ax, by - ay)
        d = np.array([bx - ax, by - ay]) / L
        n = np.array([-d[1], d[0]])
        best, cur = (0, 0), None
        for s in np.arange(0.0, L, 1.0):
            p = np.array([ax, ay]) + d * s + n * l["width"] / 4      # drive on the right-hand half
            vis = visible(info_mid, np.r_[p, 0.6], zmax=zmax) and visible(info_mid, np.r_[p, 1.6], zmax=zmax)
            if vis and cur is None:
                cur = s
            if (not vis or s + 1.0 >= L) and cur is not None:
                if s - cur > best[1] - best[0]:
                    best = (cur, s)
                cur = None
        if best[1] - best[0] >= 6.0:
            p0 = np.array([ax, ay]) + d * best[0] + n * l["width"] / 4
            p1 = np.array([ax, ay]) + d * best[1] + n * l["width"] / 4
            x0, y0, _ = project(info_mid, np.r_[p0, 1.0])
            x1, y1, _ = project(info_mid, np.r_[p1, 1.0])
            screen = math.hypot(x1 - x0, (y1 - y0) * RES[1] / RES[0])     # fraction of the frame width crossed
            out.append(dict(lane=l["id"], a=[ax, ay], d=d.tolist(), n=n.tolist(), width=l["width"], L=L,
                            vis=list(map(float, best)), screen=float(screen)))
    return sorted(out, key=lambda c: -c["screen"])


def obstacle_boxes(stage, sid, tc):
    """World bboxes of the station's process content (loads, inputs, tools) at time tc."""
    bc = UsdGeom.BBoxCache(Usd.TimeCode(tc), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = stage.GetPrimAtPath(f"/World/Process/{sid}")
    boxes = []
    for c in root.GetChildren() if root else []:
        if c.GetName() in ("Camera", "Workers"):
            continue
        for g in c.GetChildren() or [c]:
            r = bc.ComputeWorldBound(g).ComputeAlignedRange()
            if not r.IsEmpty():
                boxes.append((np.array(r.GetMin()), np.array(r.GetMax())))
    return boxes


def vehicle_path(cand, speed, t0, t1, tmid, direction):
    """Constant cruise through the visible stretch, centred on it at mid shot; returns the path parameters
    (pos(t) = base + d * speed * (t - tmid), heading in deg) - the same parameters the collision check replays."""
    d, n = np.array(cand["d"]) * direction, np.array(cand["n"]) * direction
    s_mid = (cand["vis"][0] + cand["vis"][1]) / 2
    base = np.array(cand["a"]) + np.array(cand["d"]) * s_mid + n * cand["width"] / 4
    return dict(base=base.tolist(), d=d.tolist(), speed=float(speed), tmid=float(tmid),
                heading=math.degrees(math.atan2(d[1], d[0])))


def path_pos(path):
    base, d = np.array(path["base"]), np.array(path["d"])
    return lambda t: base + d * path["speed"] * (t - path["tmid"])


def clear(pos, size, heading, boxes, times):
    L, W = size
    hx, hy = abs(math.cos(math.radians(heading))) * L / 2 + abs(math.sin(math.radians(heading))) * W / 2, \
        abs(math.sin(math.radians(heading))) * L / 2 + abs(math.cos(math.radians(heading))) * W / 2
    for t in times:
        p = pos(t)
        for lo, hi in boxes:
            if lo[2] > 3.0:          # loads hanging high on a crane pass over the vehicle
                continue
            if p[0] + hx > lo[0] and p[0] - hx < hi[0] and p[1] + hy > lo[1] and p[1] - hy < hi[1]:
                return False
    return True


def add_vehicle(stage, k, kind, path, t0, t1, tcps):
    v = VEHICLES[kind]
    pos, heading = path_pos(path), path["heading"]
    root = f"/World/Film/Vehicles/V{k}"
    x = UsdGeom.Xform.Define(stage, root)
    ref_child(stage, root, v["usd"], "Body")
    tr = x.AddTranslateOp()
    x.AddRotateZOp().Set(float(heading + v["yaw0"]))
    for fr in range(int(t0 * FPS), int(t1 * FPS) + 1):
        p = pos(fr / FPS)
        tr.Set(Gf.Vec3d(float(p[0]), float(p[1]), 0.0), fr / FPS * tcps)
    # load: a pallet with KLT bins (on the forks, or on the AGV's top plate)
    lx, lz = (-1.55, 0.12) if v["load"] == "forks" else (-0.32, 0.16)
    pal = UsdGeom.Xform.Define(stage, root + "/Pallet")
    ref_child(stage, root + "/Pallet", "Props/Pallet/pallet.usd")
    pal.AddTranslateOp().Set(Gf.Vec3d(lx, 0, lz))
    pal.AddRotateZOp().Set(90.0 if v["load"] == "forks" else 0.0)
    nx, ny, nz = (3, 4, 2) if v["load"] == "forks" else (2, 2, 1)
    for i in range(nx):
        for j in range(ny):
            for h in range(nz):
                b = UsdGeom.Xform.Define(stage, root + f"/Bins/B{i}{j}{h}")
                ref_child(stage, root + f"/Bins/B{i}{j}{h}", "Props/KLT_Bin/small_KLT.usd")
                b.AddTranslateOp().Set(Gf.Vec3d(lx + (i - (nx - 1) / 2) * 0.20, (j - (ny - 1) / 2) * 0.30,
                                                lz + 0.143 + 0.075 + h * 0.15))
    return dict(path, kind=kind)


# %% one shot


def setup(stage, sid, a, b, plan_in):
    """Edit the session layer; returns the plan actually used."""
    tcps = stage.GetTimeCodesPerSecond()
    t0, t1 = max(0.0, a - MARGIN), b + MARGIN
    tmid = (a + b) / 2
    key = f"{sid}_{a}_{b}"
    ov = OVERRIDES.get(key, {})
    cam = CAM
    infos = [cam_info(stage, cam, t * tcps) for t in (a, tmid, b)]
    plan = dict(shot=key, t0=t0, t1=t1, people=[], vehicles=[])

    # hide the twin's clones and its hand-built fleet
    workers = twin_workers(stage, sid, tmid * tcps)
    for w in workers:
        stage.GetPrimAtPath(w["path"]).SetActive(False)
    for path in ov.get("hide", []):                # per-shot: twin prims hidden in this shot only (session layer)
        if stage.GetPrimAtPath(path):
            stage.GetPrimAtPath(path).SetActive(False)
    fleet = stage.GetPrimAtPath("/World/Logistics/Fleet")
    for c in fleet.GetChildren() if fleet else []:
        if c.GetName().startswith(("FL3", "FL16", "Tugger")):
            c.SetActive(False)

    # people: the most visible work points (at most 2)
    scored = []
    for w in workers:
        p = np.array(w["p"])
        hits = [project(i, p + [0, 0, 1.0]) for i in infos]
        if all(h[0] is not None and abs(h[0]) < 0.85 and abs(h[1]) < 0.85 and 3.0 < h[2] < 45.0 for h in hits) and \
                unoccluded(*hits[1]) and unoccluded(*project(infos[1], p + [0, 0, 0.5])):
            scored.append((abs(hits[1][0]) + 0.02 * hits[1][2], w))
    scored.sort(key=lambda s: s[0])
    n_people = ov.get("people", min(2, len(scored)))
    chosen, used = [], []
    for _, w in scored:
        if len(chosen) >= n_people:
            break
        if any(np.linalg.norm(np.array(w["p"][:2]) - np.array(c["p"][:2])) < 2.5 for c in chosen):
            continue                                # keep them apart
        chosen.append(w)
    shot_ix = [s for s in SHOTS].index((sid, a, b))
    plan["candidates"] = dict(people=[dict(p=w["p"], yaw=w["yaw"], pose=w["pose"]) for _, w in scored], vehicles=[])
    final = FINAL.get(key)
    if final is not None:                     # replay the collision-checked plan (film/hall/check_plan.py)
        for k, q in enumerate(final["people"]):
            person = next(pp for pp in PEOPLE if pp[0] == q["person"])
            plan["people"].append(add_person(stage, k, person, q["motion"], q["pos"], q["yaw"], t0, t1,
                                             q["offset_s"], tcps))
        for k, v in enumerate(final["vehicles"]):
            rec = add_vehicle(stage, k, v["kind"], v, t0, t1, tcps)
            rec.update({x: v[x] for x in ("lane", "direction") if x in v})
            plan["vehicles"].append(rec)
        plan["final"] = True
        return plan
    for k, w in enumerate(chosen):
        person = PEOPLE[(2 * shot_ix + k) % len(PEOPLE)]
        options = MOTIONS_FOR_POSE.get(w["pose"], MOTIONS_FOR_POSE["stand"])
        crouch = w["pose"] == "kneel" and (k == 0 and (len(chosen) == 2 or shot_ix % 2 == 0))
        # mix postures: one crouching at the part, the other standing and looking down at it
        motion = options[0] if (w["pose"] != "kneel" or crouch) else "StandingDiscussion_LookingDown_M"
        motion = ov.get("motions", {}).get(str(k), motion)
        plan["people"].append(add_person(stage, k, person, motion, w["p"], w["yaw"], t0, t1,
                                         offset_s=3.7 * (k + 1) + 1.3 * shot_ix, tcps=tcps))

    # vehicles: about half of the shots (alternating), at most 2
    want = ov.get("vehicles", 2 if sid == "B7" else 1)
    if want:
        boxes = obstacle_boxes(stage, sid, tmid * tcps)
        times = np.linspace(t0, t1, 12)
        kinds = ov.get("kinds", ["stacker", "agv"] if shot_ix % 2 == 0 else ["agv", "stacker"])
        pairs = [(kind, c) for kind in kinds for c in lane_candidates(infos[1], VEHICLES[kind]["zmax"])
                 if c["screen"] > 0.15]
        k, used_lanes = 0, set()
        for kind, cand in pairs:
            v = VEHICLES[kind]
            for direction in (1, -1) if (shot_ix // 2) % 2 == 0 else (-1, 1):
                path = vehicle_path(cand, v["speed"], t0, t1, tmid, direction)
                if clear(path_pos(path), v["size"], path["heading"], boxes, times):
                    plan["candidates"]["vehicles"].append(dict(kind=kind, lane=cand["lane"], direction=direction,
                                                               visible=cand["vis"], screen=cand["screen"], **path))
                    if k < want and cand["lane"] not in used_lanes:
                        rec = add_vehicle(stage, k, kind, path, t0, t1, tcps)
                        rec.update(lane=cand["lane"], direction=direction, visible=cand["vis"])
                        plan["vehicles"].append(rec)
                        used_lanes.add(cand["lane"])
                        k += 1
    return plan


def time_lapse(sid, t0, t1):
    """The twin's time-lapse spans (clip seconds; real time = factor x video time) overlapping [t0, t1], clipped."""
    f = os.path.join(TWIN, "scenes", "process_next" if NEXT else "process", f"{sid}_events.json")
    if not os.path.exists(f):
        return []
    spans = json.load(open(f)).get("time_lapse", [])
    return [dict(t0=round(max(t0, s["t0"]), 3), t1=round(min(t1, s["t1"]), 3), factor=s["factor"])
            for s in spans if s["t1"] > t0 and s["t0"] < t1]


CAM = None


def hall_box(stage):
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = bc.ComputeWorldBound(stage.GetPrimAtPath("/World/Site/Hall")).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


def iso_at(stage, cam, tc, box):
    p = UsdGeom.Xformable(stage.GetPrimAtPath(cam)).ComputeLocalToWorldTransform(tc).ExtractTranslation()
    lo, hi = box
    return ISO_IN if lo[0] < p[0] < hi[0] and lo[1] < p[1] < hi[1] and p[2] < hi[2] else ISO_OUT


def main():
    global DEPTH
    S.set("/rtx/rendermode", "RaytracedLighting")
    S.set("/rtx/post/aa/op", 3)
    S.set("/rtx/reflections/enabled", True)
    S.set("/rtx/indirectDiffuse/enabled", True)
    S.set("/rtx/ambientOcclusion/enabled", True)
    tl = omni.timeline.get_timeline_interface()
    plans = json.load(open(PLAN_PATH)) if os.path.exists(PLAN_PATH) else {}
    shots = [s for s in SHOTS if not ONLY or f"{s[0]}_{s[1]}_{s[2]}" in ONLY]
    os.makedirs(os.path.join(OUT, "test" + SUFFIX), exist_ok=True)
    for sid, a, b in shots:
        started = time.time()
        key = f"{sid}_{a}_{b}"
        omni.usd.get_context().open_stage(SETS.process_layer(sid, NEXT))
        for _ in range(40):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        tcps = stage.GetTimeCodesPerSecond()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            app.update()
        global CAM
        CAM = cam = film_camera(stage, sid, a, b, OVERRIDES.get(key, {}), tcps)
        if AMBIENT > 0:
            S.set("/rtx/sceneDb/ambientLightIntensity", AMBIENT)
        rp = rep.create.render_product(cam, RES)
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach(rp)
        dann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        dann.attach(rp)
        tl.set_time_codes_per_second(tcps)
        tl.set_end_time(stage.GetEndTimeCode() / tcps)
        tl.set_current_time((a + b) / 2)
        for _ in range(10):
            app.update()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
        # this annotator returns hectometres here (measured: 0.1011 at a point 10.1 m from the image plane)
        DEPTH = np.asarray(dann.get_data(), dtype=np.float32).reshape(RES[1], RES[0]) * 100.0
        DEPTH = np.where(np.isfinite(DEPTH) & (DEPTH > 0), DEPTH, 1e6)
        dann.detach()
        print("[depth]", key, "min", float(DEPTH.min()), "median", float(np.median(DEPTH)), "max<1e6",
              float(DEPTH[DEPTH < 1e5].max()) if (DEPTH < 1e5).any() else None, flush=True)
        if TEST:
            vis = (np.clip(np.where(DEPTH < 1e5, DEPTH, 120.0) / 120.0, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUT, "test" + SUFFIX, f"{key}_depth.jpg"), vis)
        plan = setup(stage, sid, a, b, plans.get(key))
        plans[key] = plan
        json.dump(plans, open(PLAN_PATH, "w"), indent=1)
        print("[plan]", json.dumps(plan), flush=True)
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            app.update()
        for _ in range(60):
            app.update()
        box = hall_box(stage)
        t0, t1 = plan["t0"], plan["t1"]
        meta = dict(shot=key, fps=FPS, first_second=t0, margin_s=a - t0, in_s=a, out_s=b, camera=cam,
                    iso_interior=ISO_IN, ambient=AMBIENT, realism=os.environ.get("REALISM_VERSION"),
                    time_lapse=time_lapse(sid, t0, t1))
        settings = [(ISO_IN, AMBIENT)]
        if TEST:
            frames = [int(round((a + 0.5) * FPS)), int((a + b) / 2 * FPS), int(round((b - 0.5) * FPS))]
            if SWEEP:
                frames, settings = [int((a + b) / 2 * FPS)], SWEEP
            fdir = os.path.join(OUT, "test" + SUFFIX)
            json.dump(meta, open(os.path.join(fdir, f"{key}_meta.json"), "w"), indent=1)
        else:
            frames = list(range(int(round(t0 * FPS)), int(round(t1 * FPS)) + 1))
            fdir = os.path.join(OUT_SET, key)
            os.makedirs(fdir, exist_ok=True)
            json.dump(meta, open(os.path.join(fdir, "meta.json"), "w"), indent=1)
        jobs = [(i, fr, st) for i, fr in enumerate(frames) for st in settings]
        for i, fr, (iso, amb) in jobs:
            tl.set_current_time(fr / FPS)
            inside = iso_at(stage, cam, fr / FPS * tcps, box) == ISO_IN
            S.set("/rtx/post/tonemap/filmIso", iso if inside else ISO_OUT)
            S.set("/rtx/sceneDb/ambientLightIntensity", amb)
            app.update()
            rep.orchestrator.step(rt_subframes=24 if (TEST or i == 0) else 3, delta_time=0.0, pause_timeline=True)
            img = np.asarray(annot.get_data())[..., :3].astype(np.uint8)
            name = (f"{key}_iso{int(iso)}_amb{amb:g}.jpg" if SWEEP else f"{key}_{i}.jpg") if TEST else f"{i:05d}.jpg"
            cv2.imwrite(os.path.join(fdir, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not TEST and i % (FPS * 4) == 0:
                print(f"[video] {key} {i}/{len(frames)}", flush=True)
        annot.detach()
        rp.destroy()
        print(f"[done] {key} {len(frames)} frames in {time.time() - started:.0f} s", flush=True)
    app.close()


main()
