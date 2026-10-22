"""Test stills of the new vessel models on the twin's Windanker sea (offshore/windanker.usda, opened read-only; the
vessels are placed in the session layer, away from the turbines).

  OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python film/vessels/preview_vessels.py \
      [--pt] [view ...]
Writes build/vessels/preview/<view>.jpg (1600x900). --pt: path traced (slow); default RTX real-time.
"""

import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PT = "--pt" in sys.argv
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
FILM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
VES = os.path.join(FILM, "build", "vessels")
OUT = os.path.join(VES, "preview")
RES = (1600, 900)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": RES[0], "height": RES[1]})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

sys.path.insert(0, os.path.join(FILM, "film", "vessels"))
S = carb.settings.get_settings()


def T(yaw=0.0, t=(0, 0, 0), ry=0.0):
    a, b = math.radians(yaw), math.radians(ry)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry
    M[:3, 3] = t
    return M


def place(stage, path, asset, M):
    x = UsdGeom.Xform.Define(stage, path)
    if asset:
        x.GetPrim().GetReferences().AddReference(os.path.join(VES, asset))
    x.AddTransformOp().Set(Gf.Matrix4d(*np.asarray(M, float).T.ravel()))
    return x


def build(stage):
    import build_vessels as BV
    # Rotra Vente under way, visor closed
    RVp = np.array([3000.0, -3000.0, 0.0])
    place(stage, "/World/Film/Rotra", None, T(0.0, RVp))
    place(stage, "/World/Film/Rotra/Hull", "roro_vessel.usda", np.eye(4))
    place(stage, "/World/Film/Rotra/Visor", "roro_visor.usda", T(0, (BV.RV["x_visor"], 0, BV.RV["z_visor_top"])))
    # Rotra Vente with the visor open (as at the quay)
    RVo = np.array([3000.0, -3400.0, 0.0])
    place(stage, "/World/Film/RotraOpen", None, T(0.0, RVo))
    place(stage, "/World/Film/RotraOpen/Hull", "roro_vessel.usda", np.eye(4))
    place(stage, "/World/Film/RotraOpen/Visor", "roro_visor.usda", BV.visor_open_matrix())
    # Norse Energi in transit: keel at -draft, legs up, boom resting forward
    WT = BV.WT
    NEp = np.array([3600.0, -3000.0, 0.0])
    keel = -WT["draft"]
    place(stage, "/World/Film/NE", None, T(0.0, NEp))
    place(stage, "/World/Film/NE/Hull", "wtiv_hull.usda", T(0, (0, 0, keel)))
    for i, (lx, ly) in enumerate(WT["legs"]):
        place(stage, f"/World/Film/NE/Leg{i}", "wtiv_leg.usda", T(0, (lx, ly, keel - 1.0)))
    cx, cy = WT["legs"][WT["crane_leg"]]
    place(stage, "/World/Film/NE/Crane", "wtiv_crane.usda", T(WT["rest_slew"], (cx, cy, keel + WT["D"] + WT["pedestal_h"])))
    place(stage, "/World/Film/NE/Crane/Boom", "wtiv_boom.usda", T(0, WT["boom_foot"], ry=-WT["rest_luff"]))
    # SOV with the gangway stowed
    SVp = np.array([3000.0, -2650.0, 0.0])
    place(stage, "/World/Film/SOV", None, T(0.0, SVp + [0, 0, -BV.SOV["draft"]]))
    place(stage, "/World/Film/SOV/Hull", "sov.usda", np.eye(4))
    gx, gy = BV.SOV["gangway"]
    place(stage, "/World/Film/SOV/Gangway", "w2w_gangway.usda",
          T(180.0, (gx, gy, BV.SOV["D"] + BV.SOV["pedestal_h"] + 0.6)))
    return RVp, RVo, NEp, SVp


VIEWS = {}


def views(RVp, RVo, NEp, SVp):
    v = {}
    r = RVp
    v["rotra_bow_stbd"] = (r + [120, -95, 18], r + [20, 0, 8], 35)
    v["rotra_side_stbd"] = (r + [0, -190, 14], r + [0, 0, 9], 30)
    v["rotra_stern_port"] = (r + [-130, 90, 30], r + [-30, 0, 10], 35)
    v["rotra_name"] = (r + [62, -26, 5], r + [57, 0, 3.5], 35)
    v["rotra_open"] = (RVo + [115, -70, 25], RVo + [40, 0, 12], 30)
    n = NEp
    v["ne_bow_stbd"] = (n + [230, -170, 70], n + [0, 0, 30], 28)
    v["ne_stern_port"] = (n + [-210, 170, 90], n + [0, 0, 30], 28)
    v["ne_side_stbd"] = (n + [20, -260, 22], n + [10, 0, 30], 26)
    v["ne_name"] = (n + [85, -70, 12], n + [48, -31, 9], 40)
    s = SVp
    v["sov_bow_stbd"] = (s + [80, -60, 18], s + [5, 0, 6], 32)
    v["sov_side_port"] = (s + [-10, 120, 12], s + [0, 0, 6], 30)
    return v


def main():
    omni.usd.get_context().open_stage(os.path.join(TWIN, "offshore", "windanker.usda"))
    for _ in range(60):
        app.update()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    pos = build(stage)
    for _ in range(60):
        app.update()
    if PT:
        S.set("/rtx/rendermode", "PathTracing")
        S.set("/rtx/pathtracing/spp", 16)
        S.set("/rtx/pathtracing/totalSpp", 256)
        S.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        sub = 16
    else:
        S.set("/rtx/rendermode", "RaytracedLighting")
        S.set("/rtx/post/aa/op", 3)
        S.set("/rtx/reflections/enabled", True)
        S.set("/rtx/indirectDiffuse/enabled", True)
        S.set("/rtx/ambientOcclusion/enabled", True)
        sub = 8
    cam = UsdGeom.Camera.Define(stage, "/World/Film/Cam")
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0 * RES[1] / RES[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.3, 60000))
    op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
    rp = rep.create.render_product("/World/Film/Cam", RES)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)
    os.makedirs(OUT, exist_ok=True)
    for name, (eye, tgt, f) in views(*pos).items():
        if ONLY and name not in ONLY:
            continue
        op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse())
        cam.GetFocalLengthAttr().Set(float(f))
        rep.orchestrator.step(rt_subframes=sub, delta_time=0.0, pause_timeline=True)
        rep.orchestrator.step(rt_subframes=sub, delta_time=0.0, pause_timeline=True)
        img = np.asarray(annot.get_data())[..., :3].astype(np.uint8)
        Image.fromarray(img).save(os.path.join(OUT, f"{name}.jpg"), quality=92)
        print("[preview]", name, flush=True)


main()
app.close()
