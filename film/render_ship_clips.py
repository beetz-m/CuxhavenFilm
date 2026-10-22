"""Re-render the film's ship shots with the realistic, named vessel models, at 1920x1080 / 24 fps (RTX real-time).

Adapted from the twin's scripts/render_process.py (read-only): same authored clip cameras, same exposure handling
(outdoors ISO 60), RaytracedLighting with indirect diffuse, AO and reflections.
  L_loadout    the twin's scenes/process/L_loadout.usda; RoRo hull + visor swapped for build/vessels (session layer)
  OFF_install  build/scenes/process/OFF_install.usda (film-side: crane on the real aft PORT leg, new Norse Energi)
  OFF_rotor    build/scenes/process/OFF_rotor.usda (film-side, as above)
  rotra_name   NEW: Rotra Vente under way (visor closed, arms folded), tracking camera passing the bow (7 s)
  norse_name   NEW: close pass along Norse Energi's forward hull, jacked up at T07 after the nacelle lift (6 s)
Only the ranges film/edl.py uses are rendered (merged, +1 s margins): build/ships_raw/<sid>_<in>_<out>/%05d.jpg,
frame 0 = clip second <in>; film/post_frames.py then adds the lens/sensor pass into build/ships/.
Realism (unless --plain): the twin's scripts/realism.py apply_materials() textures the shared site material library in
memory before any stage is opened (the vessels' own paints carry derived texture sets), and settings() switches to
filmic tone mapping with auto-exposure off.

  OMNI_KIT_ACCEPT_EULA=YES flock <gpu.lock> ~/Documents/LabAutomationTwin/env_isaacsim/bin/python \
      film/render_ship_clips.py [--test] [name ...]
One Isaac GPU job at a time (RTX 3070, 8 GB).
"""

import math
import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ARGS = sys.argv[1:]
TEST = "--test" in ARGS
PLAIN = "--plain" in ARGS                   # without the realism pass (for before/after comparisons)
NEXT = "--next" in ARGS                     # the twin's staged fixed sources (scenes/process_next) where they exist
ONLY = [a.split("@")[0] for a in ARGS if not a.startswith("--")]
STARTS = {a.split("@")[0]: float(a.split("@")[1]) for a in ARGS if "@" in a}   # e.g. OFF_rotor@119: only that range
FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
OUT = os.path.join(FILM, "build", "ships_raw" if not TEST else ("ships_test_plain" if PLAIN else "ships_test_raw"))
RES, FPS, ISO_OUT, SUBF = (1920, 1080), 24, 60.0, 3

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": RES[0], "height": RES[1]})

import carb  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vessel_swap  # noqa: E402

if not PLAIN:
    sys.path.insert(0, os.path.join(TWIN, "scripts"))
    import realism as RZ  # noqa: E402
    RZ.apply_materials()
    __import__("product_looks").apply()      # film-side turbine surface detail (film/product_looks.py)
S = carb.settings.get_settings()
S.set("/rtx/rendermode", "RaytracedLighting")
S.set("/rtx/post/aa/op", 3)
S.set("/rtx/reflections/enabled", True)
S.set("/rtx/indirectDiffuse/enabled", True)
S.set("/rtx/ambientOcclusion/enabled", True)
S.set("/rtx/shadows/enabled", True)
S.set("/rtx/post/tonemap/filmIso", ISO_OUT)
if not PLAIN:
    RZ.settings(ISO_OUT)
tl = omni.timeline.get_timeline_interface()

CLIPS = {
    "L_loadout": (os.path.join(FILM, "build", "scenes", "process", "L_loadout.usda"), [(0, 31), (41, 53)]),
    "OFF_install": (os.path.join(FILM, "build", "scenes", "process", "OFF_install.usda"), [(0, 74)]),
    "OFF_rotor": (os.path.join(FILM, "build", "scenes", "process", "OFF_rotor.usda"), [(17, 31), (35, 65), (119, 131)]),
}


SEA = {"handle": None}
DEP = {"last": None}
SEA_PATH = "/World/RenderOnly/NorthSea"
CENTRES = ["/World/Install/WTIV", "/World/Port/Vessel", "/World/Film/Rotra"]   # centre the wave field on the action


def replace_sea(stage, centre_prim=None):
    """The realistic North Sea of the twin's scripts/ocean.py (read-only, by path): OC.install(stage, center=(x, y))
    hides the old Sea/Water prims in the session layer and adds wave geometry ~2 km around the centre; sea_update()
    advances it every frame. On by default (verified by the other session 2026-10-21); FILM_OCEAN=0 disables it."""
    SEA["handle"] = None
    if PLAIN or os.environ.get("FILM_OCEAN", "1") != "1":
        return None
    sys.path.insert(0, os.path.join(TWIN, "scripts"))
    import ocean as OC
    c = (0.0, 0.0)
    for p in ([centre_prim] if centre_prim else []) + CENTRES:
        prim = stage.GetPrimAtPath(p) if p else None
        if prim:
            # mean position over the clip (the jack-up arrives from 1.4 km): the 2 km near field covers start and end
            tcs = [0.0, stage.GetEndTimeCode() or 0.0]
            pts = [UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc).ExtractTranslation() for tc in tcs]
            c = (sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts))
            break
    # the old water's own level (offshore 0 m; the Cuxhaven site's harbour water is at -5 m) and, in the harbour, a
    # sheltered sea state (ILLUSTRATIVE: wind sea Hs 0.35 m, little swell) instead of the open North Sea's Hs 1.5 m
    bc = UsdGeom.BBoxCache(0.0, ["default", "render"])
    levels = [bc.ComputeWorldBound(p).ComputeAlignedRange().GetMax()[2] for p in stage.Traverse()
              if p.GetName() in OC.WATER_NAMES and p.IsA(UsdGeom.Imageable)]
    z = float(max(levels)) if levels else 0.0
    harbour = bool(stage.GetPrimAtPath("/World/Port"))
    sea = dict(hs=0.35, tp=3.2, wind_ms=6.0, swell_hs=0.08, swell_tp=8.0) if harbour else None
    # film foam (film/sea.py: coherent whitecaps, soft trails) unless FILM_FOAM=twin
    inst = OC.install if os.environ.get("FILM_FOAM", "film") == "twin" else __import__("sea").install
    SEA["handle"] = inst(stage, center=c, sea=sea)
    if abs(z) > 1e-3:
        UsdGeom.Xformable(stage.GetPrimAtPath(SEA_PATH)).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
    print(f"[ships] ocean installed, centre {c}, level {z:+.2f} m, {'harbour' if harbour else 'open sea'}", flush=True)
    return SEA["handle"]


def sea_update(stage, t):
    if SEA["handle"] is not None:
        SEA["handle"].update(stage, SEA_PATH, float(t))


if NEXT:
    # L_loadout: the film-side copy of the twin's process_next/L_loadout.usda (build_offshore_film.py --next), falling
    # back to the current one; OFF_*: film-side layers generated from the twin's builders (crane on the port leg)
    nxt = os.path.join(FILM, "build", "scenes", "process_next", "L_loadout.usda")
    if os.path.exists(os.path.join(TWIN, "scenes", "process_next", "L_loadout.usda")) and os.path.exists(nxt):
        CLIPS["L_loadout"] = (nxt, CLIPS["L_loadout"][1])
    for sid, (pth, _) in CLIPS.items():
        print(f"[ships] --next: {sid} from {pth}", flush=True)


def open_stage(path):
    omni.usd.get_context().open_stage(path)
    for _ in range(60):
        app.update()
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        app.update()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    return stage


def settle(n=20):
    for _ in range(n):
        app.update()


def grab(annot, steps):
    rep.orchestrator.step(rt_subframes=steps, delta_time=0.0, pause_timeline=True)
    return np.asarray(annot.get_data())[..., :3].astype(np.uint8)


def save(img, path, depth=None):
    """JPEG 95; with a depth map (metres, from RZ.depth_annotator) the twin's RZ.haze() is applied first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if depth is not None and not PLAIN:
        tmp = os.path.join("/dev/shm", f"film_haze_{os.getpid()}.png")
        cv2.imwrite(tmp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        RZ.haze(tmp, depth, dst=tmp)
        img = cv2.cvtColor(cv2.imread(tmp), cv2.COLOR_BGR2RGB)
        os.remove(tmp)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def depth_of(dep):
    return None if dep is None else np.asarray(dep.get_data())


def clip(sid):
    path, ranges = CLIPS[sid]
    stage = open_stage(path)
    print(f"[ships] {sid}: swapped {len(vessel_swap.swap(stage))} vessel references", flush=True)
    replace_sea(stage)
    settle(30)
    rp = rep.create.render_product(f"/World/Process/{sid}/Camera", RES)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)
    dep = None if PLAIN else RZ.depth_annotator(rp)
    tcps = stage.GetTimeCodesPerSecond()
    tl.set_time_codes_per_second(tcps)
    tl.set_end_time(stage.GetEndTimeCode() / tcps)
    t_end = stage.GetEndTimeCode() / tcps
    for a, b in ranges:
        if sid in STARTS and a != STARTS[sid]:
            continue
        b = min(b, t_end)
        d = os.path.join(OUT, f"{sid}_{a}_{int(b)}")
        n = int(round((b - a) * FPS)) + 1
        ks = [0, n // 2, n - 1] if TEST else range(n)
        t0 = time.time()
        for j, k in enumerate(ks):
            tl.set_current_time(a + k / FPS)
            sea_update(stage, a + k / FPS)
            app.update()
            img = grab(annot, 12 if j == 0 else SUBF)
            save(img, os.path.join(d, f"{k:05d}.jpg"), depth_of(dep))
            if k % (FPS * 5) == 0 or TEST:
                print(f"[ships] {sid} {a + k / FPS:.1f}s ({time.time() - t0:.0f}s)", flush=True)
        open(os.path.join(d, "fps.txt"), "w").write(str(FPS))
    annot.detach()
    rp.destroy()


def look(op, cam, eye, tgt, focal):
    op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse())
    cam.GetFocalLengthAttr().Set(float(focal))


def new_cam(stage, path="/World/Film/Cam"):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.3, 80000))
    op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
    rp = rep.create.render_product(path, RES)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)
    DEP["last"] = None if PLAIN else RZ.depth_annotator(rp)
    return cam, op, rp, annot


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def xf(stage, path, ref=None, M=None):
    x = UsdGeom.Xform.Define(stage, path)
    if ref:
        x.GetPrim().GetReferences().AddReference(ref)
    op = x.AddTransformOp()
    if M is not None:
        op.Set(Gf.Matrix4d(*np.asarray(M, float).T.ravel()))
    return op


def T(yaw=0.0, t=(0, 0, 0)):
    a = math.radians(yaw)
    M = np.eye(4)
    M[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    M[:3, 3] = t
    return M


def rotra_name(seconds=7.0):
    """Rotra Vente under way (5 m/s), visor closed, arms folded; the camera overtakes along the starboard side and
    settles on the name at the bow."""
    stage = open_stage(os.path.join(FILM, "build", "scenes", "windanker_film.usda"))
    ves = os.path.join(FILM, "build", "vessels")
    p0, yaw, v = np.array([-5200.0, -3800.0]), 20.0, 5.0
    op_ship = xf(stage, "/World/Film/Rotra")
    xf(stage, "/World/Film/Rotra/Hull", os.path.join(ves, "roro_vessel_closed.usda"), np.eye(4))
    xf(stage, "/World/Film/Rotra/Visor", os.path.join(ves, "roro_visor.usda"), T(0, (54.0, 0, 12.0)))
    op_ship.Set(Gf.Matrix4d(*T(yaw, (*p0, 0.0)).T.ravel()))
    replace_sea(stage, "/World/Film/Rotra")
    settle(40)
    cam, op, rp, annot = new_cam(stage)
    d = os.path.join(OUT, "rotra_name")
    n = int(round(seconds * FPS))
    ks = [0, n // 2, n - 1] if TEST else range(n)
    for j, k in enumerate(ks):
        t = k / FPS
        W = T(yaw, (*(p0 + v * t * np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])), 0.0))
        op_ship.Set(Gf.Matrix4d(*W.T.ravel()))
        u = ease(t / seconds)
        eye_l = np.array([2.0 + 49.0 * u, -60.0 + 18.0 * u, 9.0 - 3.0 * u, 1.0])       # ship frame, starboard side
        tgt_l = np.array([16.0 + 31.0 * u, -10.0, 9.0 - 5.5 * u, 1.0])                # ends on the name (x 41..53)
        look(op, cam, (W @ eye_l)[:3], (W @ tgt_l)[:3], 28.0 + 14.0 * u)
        sea_update(stage, t)
        app.update()
        save(grab(annot, 12 if j == 0 else SUBF), os.path.join(d, f"{k:05d}.jpg"), depth_of(DEP["last"]))
    open(os.path.join(d, "fps.txt"), "w").write(str(FPS))
    annot.detach()
    rp.destroy()
    print("[ships] rotra_name done", flush=True)


def norse_name(seconds=6.0):
    """close pass along Norse Energi's starboard forward hull, jacked up at T07 (OFF_install at 74 s)."""
    stage = open_stage(CLIPS["OFF_install"][0])
    tcps = stage.GetTimeCodesPerSecond()
    tl.set_time_codes_per_second(tcps)
    tl.set_end_time(stage.GetEndTimeCode() / tcps)
    tl.set_current_time(74.0)
    replace_sea(stage)
    settle(20)
    hull = stage.GetPrimAtPath("/World/Install/WTIV/Hull")
    H = np.array(UsdGeom.Xformable(hull).ComputeLocalToWorldTransform(74.0 * tcps)).T     # hull frame (keel) -> world
    cam, op, rp, annot = new_cam(stage)
    d = os.path.join(OUT, "norse_name")
    n = int(round(seconds * FPS))
    ks = [0, n // 2, n - 1] if TEST else range(n)
    for j, k in enumerate(ks):
        u = ease(k / (n - 1))
        eye_l = np.array([22.0 + 46.0 * u, -31.0 - 62.0 + 16.0 * u, 6.0 + 4.0 * u, 1.0])
        tgt_l = np.array([44.0 + 8.0 * u, -31.0, 13.0 + 1.0 * u, 1.0])
        look(op, cam, (H @ eye_l)[:3], (H @ tgt_l)[:3], 24.0 + 10.0 * u)
        sea_update(stage, 74.0 + k / FPS)
        app.update()
        save(grab(annot, 12 if j == 0 else SUBF), os.path.join(d, f"{k:05d}.jpg"), depth_of(DEP["last"]))
    open(os.path.join(d, "fps.txt"), "w").write(str(FPS))
    annot.detach()
    rp.destroy()
    print("[ships] norse_name done", flush=True)


try:
    t0 = time.time()
    for name in ["L_loadout", "OFF_install", "OFF_rotor", "rotra_name", "norse_name"]:
        if ONLY and name not in ONLY:
            continue
        if name in CLIPS:
            clip(name)
        else:
            globals()[name]()
        print(f"[ships] {name} finished, {time.time() - t0:.0f}s total", flush=True)
finally:
    app.close()
