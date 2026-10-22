"""Animated shots of the robot-only offshore service visit at Windanker T07 (Cuxhaven research twin), for the film.

Loads ~/Documents/CuxhavenSiteTwin/offshore/service_t07.usda READ-ONLY (all edits in the session layer) and reuses
the twin's render helpers and robot posing (scripts/render_lib.py, scripts/pose_fk.py; posing as in
scripts/render_service.py). Rigid prims (drones, the hoisted ANYmal, the hoist line) and two robot arms are animated
per frame; the camera follows eased paths. The rotor is locked (mission step S02), so it does not turn.

  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python film/render_service_anim.py \
      [--test] [--spp N] [--sub N] [shot ...]
Writes build/service/<shot>/%04d.png (1920x1080, 24 fps); encode with film/encode_frames.py.\nRenderer: RTX RaytracedLighting (path tracing of this scene took ~80-95 s per 1080p frame on the RTX 3070).
Only one Isaac GPU job at a time (RTX 3070, 8 GB): stop the GUI first.
"""

import json
import math
import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ARGS = sys.argv[1:]
TEST = "--test" in ARGS


def opt(name, default):
    return int(ARGS[ARGS.index(name) + 1]) if name in ARGS else default


RT = "--pt" not in ARGS                     # RaytracedLighting by default (video); --pt = path tracing (slow here)
SPP, SUB = opt("--spp", 4), opt("--sub", 8 if not RT else 6)
ONLY = [a for i, a in enumerate(ARGS) if not a.startswith("--") and (i == 0 or ARGS[i - 1] not in ("--spp", "--sub"))]
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(FILM, "build", "service")
RES, FPS = (1920, 1080), 24

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting" if RT else "PathTracing",
                     "width": RES[0], "height": RES[1],
                     "extra_args": ["--/crashreporter/enabled=false"]})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

sys.path.insert(0, os.path.join(TWIN, "scripts"))
import pose_fk as FK  # noqa: E402
import render_lib as RL  # noqa: E402
import ocean as OC  # noqa: E402
import realism as RZ  # noqa: E402

OLDSEA = "--oldsea" in ARGS                 # the previous look: flat sea, no realism / haze / post
if not OLDSEA:
    RZ.apply_materials()                     # realistic texture sets on the shared library (in memory), as the ships
    __import__("product_looks").apply()      # film-side turbine surface detail (film/product_looks.py)

if RT:
    import carb

    _s = carb.settings.get_settings()
    _s.set("/rtx/rendermode", "RaytracedLighting")
    _s.set("/rtx/post/aa/op", 3)                 # DLSS / DLAA: needs a few subframes to converge
    _s.set("/rtx/post/dlss/execMode", 2)         # quality
    _s.set("/rtx/indirectDiffuse/enabled", True)
    _s.set("/rtx/ambientOcclusion/enabled", True)
    _s.set("/rtx/reflections/enabled", True)
    _s.set("/rtx/shadows/enabled", True)
else:
    RL.configure(SPP, SUB)
if not OLDSEA:
    RZ.settings(60.0)                        # filmic tone mapping, auto-exposure off, ISO 60: the ships' settings
omni.usd.get_context().open_stage(os.path.join(TWIN, "offshore", "service_t07.usda"))
for _ in range(120):
    app.update()
stage = omni.usd.get_context().get_stage()
while omni.usd.get_context().get_stage_loading_status()[2] > 0:
    app.update()
stage.SetEditTarget(stage.GetSessionLayer())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vessel_swap  # noqa: E402
print("[anim] film vessels swapped in:", vessel_swap.swap(stage), flush=True)   # new SOV + gangway (session layer)
for _ in range(30):
    app.update()
_inst = OC.install if os.environ.get("FILM_FOAM", "film") == "twin" else __import__("sea").install   # film/sea.py foam
SEA = None if OLDSEA else _inst(stage, center=(0.0, 0.0))   # North Sea around T07 (scene origin), 2 km near field
SEA_PATH = "/World/RenderOnly/NorthSea"
info = json.load(open(os.path.join(TWIN, "data", "service_scene.json")))
Mn = np.array(info["nacelle_matrix"])
TP, TOP = 20.0, float(Mn[2, 3])


def nac(p):
    return (Mn @ np.r_[p, 1.0])[:3]


def find(root, pred):
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()) if pred(p)]


# ---------------------------------------------------------------- robot posing (as scripts/render_service.py)
GROUND = {"/World/T07/Robots/AnymalArm": TOP + 3.20, "/World/T07/Robots/SpotArm": TOP + 3.20,
          "/World/T07/Robots/AnymalHoist": TP + 0.3}
POSES, EE = {}, {}


def place_arm(rp, q_arm):
    """Arm on the ANYmal base, posed, gripper on the end effector."""
    base = stage.GetPrimAtPath(f"{rp}/Base/base")
    Wb = FK.world(base) if base and base.IsValid() else FK.world(stage.GetPrimAtPath(f"{rp}/Base"))
    Tm = np.eye(4)
    Tm[:3, 3] = [0.22, 0.0, 0.17]
    FK.set_world(stage.GetPrimAtPath(f"{rp}/Arm"), Wb @ Tm)
    FK.pose(stage, f"{rp}/Arm", q_arm)
    if rp not in EE:
        ee = find(f"{rp}/Arm", lambda p: "end_effector" in p.GetName().lower() or p.GetName() in ("bracelet_link",))
        EE[rp] = ee[0] if ee else None
    if EE[rp] is not None:
        FK.set_world(stage.GetPrimAtPath(f"{rp}/Gripper"), FK.world(EE[rp]))


for rp, ground in GROUND.items():
    r = stage.GetPrimAtPath(rp)
    pose = json.loads(r.GetAttribute("sdt:pose").Get())
    POSES[rp] = pose
    for part, q in pose.items():
        FK.pose(stage, f"{rp}/{part}", q)
    if stage.GetPrimAtPath(f"{rp}/Arm"):
        place_arm(rp, pose.get("Arm", {}))
    feet = [stage.GetPrimAtPath(f"{rp}/{part}") for part in pose if part != "Arm"]
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    zmin = min(bc.ComputeWorldBound(f).ComputeAlignedRange().GetMin()[2] for f in feet)
    M = FK.world(r)
    M[2, 3] += ground - zmin
    FK.set_world(r, M)
    if stage.GetPrimAtPath(f"{rp}/Arm"):
        place_arm(rp, pose.get("Arm", {}))      # arm follows the base after the ground shift

CUT = ["S_B7/Parts/CanopyPanels", "S_B5/Parts/CoolerUnit", "S_B8/Parts/HeliPlatform", "S_B7/Parts/UpperDeck",
       "S_B7/Parts/SwitchgearCabinet", "S_B5/Parts/CanopyFrame", "S_B7/Joints", "S_B8/Joints", "S_B5/Joints",
       "S_B8/Parts/ServiceCrane", "S_B5/Parts/CoolingPumpSkid"]


def cutaway(on):
    for c in CUT:
        p = stage.GetPrimAtPath(f"/World/T07/Nacelle/{c}")
        if not p:
            continue
        for q in ([p] if p.IsA(UsdGeom.Imageable) else [c_ for c_ in p.GetChildren() if c_.IsA(UsdGeom.Imageable)]):
            UsdGeom.Imageable(q).GetVisibilityAttr().Set("invisible" if on else "inherited")
    for p in ("/World/T07/NacelleServiceRoof", "/World/Drones/Cargo"):
        UsdGeom.Imageable(stage.GetPrimAtPath(p)).GetVisibilityAttr().Set("invisible" if on else "inherited")


# ---------------------------------------------------------------- animated prims, restored before every shot
ANIM = ["/World/Drones/Inspection", "/World/Drones/Contact", "/World/Drones/Cargo", "/World/T07/Robots/AnymalHoist",
        "/World/T07/Equipment/NacelleCrane/HookBlock"]
HOME = {p: FK.world(stage.GetPrimAtPath(p)) for p in ANIM}
LINE = UsdGeom.Cylinder(stage.GetPrimAtPath("/World/T07/HoistLine"))
LINE_TOP = 80.1 + 114.2 / 2                                     # crane end of the hoist line (as authored)
LINE_XY = (-2.8433788991949696, 7.731105770561724)
LINE_OP = UsdGeom.Xformable(LINE.GetPrim()).GetOrderedXformOps()[0]


def move(path, M):
    FK.set_world(stage.GetPrimAtPath(path), M)


def translated(path, d):
    M = HOME[path].copy()
    M[:3, 3] += np.asarray(d, float)
    return M


def bbox(path):
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    return bc.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()


HOIST_TOP0 = bbox("/World/T07/Robots/AnymalHoist").GetMax()[2] - HOME["/World/T07/Robots/AnymalHoist"][2, 3]


def hoist_to(z_root, shift=(0.0, 0.0), line_top=LINE_TOP):
    """Hoisted ANYmal with its root at height z_root (shifted in plan by `shift`); the hoist line runs from its top
    up to line_top (the crane hook)."""
    p = "/World/T07/Robots/AnymalHoist"
    M = HOME[p].copy()
    M[2, 3] = z_root
    M[:2, 3] += shift
    move(p, M)
    bottom = z_root + HOIST_TOP0
    LINE.GetHeightAttr().Set(float(line_top - bottom))
    LINE_OP.Set((LINE_XY[0], LINE_XY[1], float((line_top + bottom) / 2)))


def reset():
    for p in ANIM:
        move(p, HOME[p])
    hoist_to(HOME["/World/T07/Robots/AnymalHoist"][2, 3])
    for rp in ("/World/T07/Robots/AnymalArm",):
        place_arm(rp, POSES[rp]["Arm"])
    FK.pose(stage, "/World/T07/Robots/SpotArm/Body", POSES["/World/T07/Robots/SpotArm"]["Body"])
    cutaway(False)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


# ---------------------------------------------------------------- the shots: frame(t in [0,1]) -> (eye, target, focal)
tip = np.array(info["blade_tip"])
hub = FK.world(stage.GetPrimAtPath("/World/T07/Rotor/Hub"))[:3, 3]
mid = np.array(info["blade_mid"])
blade_dir = (mid - hub) / np.linalg.norm(mid - hub)
drone_off = HOME["/World/Drones/Inspection"][:3, 3] - mid
cam_off = Mn[:3, 0] * 7.0 + Mn[:3, 1] * 5.0 + np.array([0, 0, 2.5])
contact0 = HOME["/World/Drones/Contact"][:3, 3]
cargo0 = HOME["/World/Drones/Cargo"][:3, 3]


def shot_approach(t):
    s = ease(t)
    return lerp((300.0, 232.0, 14.0), (128.0, 98.0, 30.0), s), lerp((0.0, 0.0, 104.0), (8.0, 14.0, 46.0), s), 17 + 4 * s


def shot_drone_blade(t):
    s = lerp(34.0, 92.0, ease(t))                       # metres along the blade from the hub
    p = hub + blade_dir * s + drone_off
    M = HOME["/World/Drones/Inspection"].copy()
    M[:3, 3] = p
    move("/World/Drones/Inspection", M)
    b = hub + blade_dir * s                             # the blade point the drone inspects
    away = (p - b) / np.linalg.norm(p - b)              # camera behind the drone, looking past it at the blade
    side = np.cross(away, (0.0, 0.0, 1.0))
    eye = p + away * 4.2 + side * 1.6 + np.array([0, 0, 0.5]) - blade_dir * 1.0
    return eye, b * 0.35 + p * 0.65, 26


def shot_contact_drone(t):
    a = ease(min(t / 0.75, 1.0))                        # approach, then hold on the receptor
    move("/World/Drones/Contact", translated("/World/Drones/Contact", lerp((-2.5, -9.0, -3.0), (0, 0, 0), a)))
    # near tip height (from 5.5 m above, the drone read as hovering over the sea, not touching the blade)
    eye = lerp(tip + np.array([-10.0, -12.0, 1.6]), tip + np.array([-6.0, -7.0, 0.9]), ease(t))
    return eye, lerp(contact0 + np.array([-1.2, -4.5, -1.5]), tip, ease(t)), 30


# the hoist runs at (-2.84, 7.73), about 8 m out from the tower axis: cameras stay on that side of the tower
U = np.array([LINE_XY[0], LINE_XY[1]]) / np.linalg.norm(LINE_XY)        # tower axis -> hoist, horizontal
V = np.array([U[1], -U[0]])


def shot_hoist_lift(t):
    z0 = HOME["/World/T07/Robots/AnymalHoist"][2, 3]
    z = z0 + 7.0 * ease(t)
    hoist_to(z)
    xy = np.array(LINE_XY) + U * 9.0 + V * 4.5
    eye = np.r_[xy, lerp(22.0, 24.5, ease(t))]
    return eye, np.r_[LINE_XY, z + 0.6], 24


def robot_extent(path="/World/T07/Robots/AnymalHoist"):
    """min/max of the hoisted robot's mesh points in the nacelle frame (instance proxies included)."""
    Mi, xc, pts = np.linalg.inv(Mn), UsdGeom.XformCache(), []
    for q in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
        if q.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(q).ComputeVisibility() != "invisible":
            P = np.array(UsdGeom.Mesh(q).GetPointsAttr().Get() or [], float)
            if len(P):
                W = np.array(xc.GetLocalToWorldTransform(q)).T
                pts.append((Mi @ W @ np.c_[P, np.ones(len(P))].T).T[:, :3])
    pts = np.concatenate(pts)
    return pts.min(0), pts.max(0)


HOOK = "/World/T07/Equipment/NacelleCrane/HookBlock"
HOOK_HOME = FK.world(stage.GetPrimAtPath(HOOK))
HOOK_SEAT0 = bbox(HOOK).GetMin()[2] + 0.02                       # hook seat: the sling attaches here
DECK_TOP = TOP + 3.32                                           # lower deck / coaming top (nacelle_hf DECK_Z + 0.12)
LIFT = {}


def hatch_lift():
    """Open the hatch (film/hatch_cut.py) and derive the lift: robot centred in the 1.4 x 0.9 m opening, raised
    until its lowest point clears the deck by 0.15 m; the crane hook rises with it (sling >= 0.15 m)."""
    import hatch_cut
    reset()
    lo, hi = robot_extent()                                     # nacelle frame, robot at home on the TP
    print("[anim] hatch opened:", hatch_cut.open_hatch(stage, Mn, FK), flush=True)
    x0, x1, y0, y1 = hatch_cut.OPEN
    d = np.array([(x0 + x1) / 2 - (lo[0] + hi[0]) / 2, (y0 + y1) / 2 - (lo[1] + hi[1]) / 2, 0.0])
    margin = min(x1 - x0 - (hi[0] - lo[0]), y1 - y0 - (hi[1] - lo[1])) / 2
    assert margin > 0.05, f"robot does not fit the hatch opening (margin {margin:.3f} m)"
    z0 = HOME["/World/T07/Robots/AnymalHoist"][2, 3]
    off_b = lo[2] + TOP - z0                                    # lowest robot point relative to its root
    LIFT.update(shift=(Mn[:3, :3] @ d)[:2], z_end=DECK_TOP + 0.15 - off_b, height=hi[2] - lo[2], margin=margin)
    print(f"[anim] hatch lift: plan shift {d[:2].round(3)} m, side margin {margin:.3f} m, root z end "
          f"{LIFT['z_end']:.3f}, robot height {LIFT['height']:.2f} m", flush=True)


def hook_at(seat):
    M = HOOK_HOME.copy()
    M[2, 3] += seat - HOOK_SEAT0
    move(HOOK, M)


def shot_hoist_hatch(t):
    if not LIFT:
        hatch_lift()
    z = float(lerp(112.0, LIFT["z_end"], ease(t)))
    seat = max(HOOK_SEAT0, z + HOIST_TOP0 + 0.15)
    hook_at(seat)
    hoist_to(z, LIFT["shift"], seat)
    xy = np.array(LINE_XY) + U * 15.0 + V * 6.0
    eye = np.r_[xy, 116.0]
    return eye, np.r_[LINE_XY, lerp(115.0, 133.5, ease(t))], 22


def shot_hatch_check(t):
    """QC only (not in the film): the opening from below (t<0.5) and from above inside the cutaway nacelle."""
    shot_hoist_hatch(0.0)
    z = float(lerp(LIFT["z_end"] - 1.2, LIFT["z_end"], t))
    seat = max(HOOK_SEAT0, z + HOIST_TOP0 + 0.15)
    hook_at(seat)
    hoist_to(z, LIFT["shift"], seat)
    lo, hi = robot_extent()
    print(f"[anim] hatch_check t={t:.2f}: robot (nacelle) {lo.round(3)} .. {hi.round(3)}, hook seat {seat:.3f}", flush=True)
    c = nac((-8.02, -1.88, 3.2))
    if t > 0.75:
        cutaway(True)
        return c + np.array([0.3, 0.4, 4.5]), c, 18
    if t < 0.5:
        return c + np.array([2.5, 2.0, -6.0]), c, 18
    cutaway(True)
    return c + np.array([4.5, 5.5, 5.0]), c + np.array([0, 0, 0.3]), 18


def shot_nacelle_robots(t):
    cutaway(True)
    w = math.sin(2 * math.pi * t)
    qa = dict(POSES["/World/T07/Robots/AnymalArm"]["Arm"])
    qa["joint_1"] += 0.35 * w
    qa["joint_6"] += 0.25 * math.sin(4 * math.pi * t)
    place_arm("/World/T07/Robots/AnymalArm", qa)
    qs = dict(POSES["/World/T07/Robots/SpotArm"]["Body"])
    qs["arm0_sh0"] += 0.30 * w
    qs["arm0_wr1"] += 0.6 * math.sin(2 * math.pi * t + 1.0)
    FK.pose(stage, "/World/T07/Robots/SpotArm/Body", qs)
    ag, sg = info["anymal_goal"], info["spot_goal"]
    e0, t0 = nac((ag[0] + 2.6, ag[1] + 1.6, 3.2 + 1.9)), nac((ag[0] - 0.5, ag[1] - 0.3, 3.2 + 0.6))
    e1, t1 = nac((sg[0] - 2.8, sg[1] + 1.9, 3.2 + 2.0)), nac((sg[0] + 0.3, sg[1] - 0.4, 3.2 + 0.6))
    s = ease(t)
    eye = lerp(e0, e1, s) + np.array([0, 0, 0.6 * math.sin(math.pi * s)])
    return eye, lerp(t0, t1, s), 19


def shot_cargo_drone(t):
    a = ease(min(t / 0.85, 1.0))
    move("/World/Drones/Cargo", translated("/World/Drones/Cargo", lerp((5.0, -3.0, 15.0), (0, 0, 0), a)))
    p = FK.world(stage.GetPrimAtPath("/World/Drones/Cargo"))[:3, 3]
    eye0, tgt0 = nac((-2.0, -9.0, 13.5)), nac((-10.3, -1.5, 11.6))
    eye = lerp(eye0 + np.array([0, 0, 4.0]), eye0, ease(t))
    return eye, lerp(p, tgt0, 0.45), 20


SHOTS = {  # name: (seconds, frame function, cutaway)
    "approach": (7.0, shot_approach),
    "drone_blade": (7.0, shot_drone_blade),
    "contact_drone": (6.0, shot_contact_drone),
    "hoist_lift": (5.0, shot_hoist_lift),
    "hoist_hatch": (6.0, shot_hoist_hatch),
    "nacelle_robots": (8.0, shot_nacelle_robots),
    "cargo_drone": (6.0, shot_cargo_drone),
    "hatch_check": (0.125, shot_hatch_check),
}

cam = RL.Camera(stage, RES, SUB, path="/World/RenderOnly/Cam", far=60000.0)
DEP = None if OLDSEA else RZ.depth_annotator(cam.rp)
T_OFF = {name: 20.0 * k for k, name in enumerate(["approach", "drone_blade", "contact_drone", "hoist_lift",
                                                  "hoist_hatch", "nacelle_robots", "cargo_drone"])}
times = []
try:
    for name, (seconds, fn) in SHOTS.items():
        if (ONLY and name not in ONLY) or (not ONLY and name == "hatch_check"):
            continue
        reset()
        n = int(round(seconds * FPS))
        frames = [0, n // 2, n - 1] if TEST else range(n)
        for i in frames:
            eye, tgt, focal = fn(i / (n - 1))
            if SEA is not None:
                SEA.update(stage, SEA_PATH, T_OFF.get(name, 0.0) + i / FPS)
            t0 = time.time()
            with open(os.devnull, "w") as dn:
                stdout, sys.stdout = sys.stdout, dn
                try:
                    cam.shot(os.path.join(OUT, name, f"{i:04d}.png"), eye, tgt, focal)
                finally:
                    sys.stdout = stdout
            if DEP is not None:                    # haze (depth of this frame), then lens / sensor, grain seed = frame
                png = os.path.join(OUT, name, f"{i:04d}.png")
                RZ.haze(png, np.asarray(DEP.get_data()), dst=png)
                RZ.post(png, seed=i)
            times.append(time.time() - t0)
            if i % 24 == 0 or TEST:
                print(f"[anim] {name} {i + 1}/{n} {times[-1]:.2f}s", flush=True)
finally:
    if times:
        print(f"[anim] frames {len(times)}, mean {np.mean(times):.2f}s, max {np.max(times):.2f}s", flush=True)
    app.close()
