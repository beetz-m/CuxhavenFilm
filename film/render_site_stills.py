"""Film-side re-renders of the twin's site aerials with the realistic sea (the twin's stills show its old tiled water).

Opens ~/Documents/CuxhavenSiteTwin/scenes/cuxhaven_site.usda READ-ONLY (all edits in the session layer), applies the
twin's realism materials, replaces the Elbe water (z = -5 m) by the film's ocean (film/sea.py) in a sheltered estuary
sea state, and renders the same views as the twin's scripts/render_site.py (RTX real-time, 64 subframes, haze from
depth, then the lens/sensor pass). Writes build/stills/<view>.png; film/edl.py S() prefers these over the twin's.

  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python film/render_site_stills.py [view ...]
"""

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
NEXT = "--next" in sys.argv          # the twin's staged clean site (scenes/cuxhaven_site_next.usda, round 7)
FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
OUT = os.path.join(FILM, "build", "stills")
RES, ISO_OUT, SUB = (1920, 1200), 60.0, 64

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": RES[0], "height": RES[1]})

import carb  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

sys.path.insert(0, os.path.join(TWIN, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism as RZ  # noqa: E402
import sea as FS  # noqa: E402

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
RZ.settings(ISO_OUT)

# the twin's views (scripts/render_site.py): name -> (eye, target, focal mm)
VIEWS = {
    "geo_elbe_low": ((-300, 900, 25), (0, 0, 8), 24),
    "aerial_site": ((420, -700, 520), (-330, 260, 0), 26),
    "aerial_port": ((-330, 190, 160), (-640, 470, 0), 24),
    # the milk-run aisle (twin: ops_tugger_trains at eye 5 m, ISO 60, too dark): higher, wider, interior exposure;
    # two candidates, the edit takes ops_tugger_trains
    "ops_tugger_trains": ((54, 38, 8.0), (70, 66, 0.5), 18),
    "ops_tugger_trains_b": ((80, 44, 6.5), (64, 68, 1.0), 20),
}
ISO = {"ops_tugger_trains": 200.0, "ops_tugger_trains_b": 200.0,     # interior views; others ISO_OUT
       "aerial_site": 45.0, "aerial_port": 48.0}                     # high aerials read milky at ISO 60
VIS = {"aerial_site": 45000.0, "aerial_port": 40000.0}             # haze visibility (m); default 25 km
# Elbe estuary off the site: sheltered compared with the open North Sea (ILLUSTRATIVE: wind 7 m/s, Hs 0.6 m)
ESTUARY = dict(hs=0.6, tp=3.8, wind_ms=7.0, swell_hs=0.15, swell_tp=8.0)
SEA_PATH = "/World/RenderOnly/NorthSea"

omni.usd.get_context().open_stage(os.path.join(TWIN, "scenes", "cuxhaven_site_next.usda" if NEXT else "cuxhaven_site.usda"))
for _ in range(60):
    app.update()
while omni.usd.get_context().get_stage_loading_status()[2] > 0:
    app.update()
stage = omni.usd.get_context().get_stage()
stage.SetEditTarget(stage.GetSessionLayer())
sea = FS.install(stage, center=(-350.0, 650.0), sea=ESTUARY)
UsdGeom.Xformable(stage.GetPrimAtPath(SEA_PATH)).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -5.0))   # the Elbe level
sea.update(stage, SEA_PATH, 0.0)
for _ in range(30):
    app.update()

cam = UsdGeom.Camera.Define(stage, "/World/Film/StillCam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.5, 120000))
op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
rp = rep.create.render_product("/World/Film/StillCam", RES)
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach(rp)
dep = RZ.depth_annotator(rp)

os.makedirs(OUT, exist_ok=True)
try:
    for k, (name, (eye, tgt, f)) in enumerate(VIEWS.items()):
        if ONLY and name not in ONLY:
            continue
        op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)),
                                       Gf.Vec3d(0, 0, 1)).GetInverse())
        cam.GetFocalLengthAttr().Set(float(f))
        S.set("/rtx/post/tonemap/filmIso", ISO.get(name, ISO_OUT))
        rep.orchestrator.step(rt_subframes=SUB, delta_time=0.0, pause_timeline=True)
        img = np.asarray(annot.get_data())[..., :3].astype(np.uint8)
        path = os.path.join(OUT, f"{name}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)     # pre-haze copy + depth, to retune without the GPU
        cv2.imwrite(os.path.join(OUT, "raw", f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(OUT, "raw", f"{name}_depth.npy"), np.asarray(dep.get_data(), np.float32))
        if name not in ("ops_tugger_trains", "ops_tugger_trains_b"):    # haze only outdoors
            RZ.haze(path, np.asarray(dep.get_data()), visibility_m=VIS.get(name, 25000.0), dst=path)
        RZ.post(path, dst=path, seed=k)
        print(f"[stills] {path}", flush=True)
finally:
    app.close()
