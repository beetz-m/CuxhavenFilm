"""Diagnose the white dashes on the open sea: single frames of OFF_install with material variants (no clip renders).

  A  as rendered (film foam, roughness 0.035 + foam, detail bump 0.35)
  B  foam off (albedo without foam): what is left is sky/sun reflection on wave facets
  C  rougher water (roughness 0.08 constant) and weaker capillary detail (detail bump 0.15), foam on
  D  B + C
Writes build/sea_test/<t>_<variant>.jpg and build/sea_test/sheet.jpg (rows: times, columns: A-D).

  OMNI_KIT_ACCEPT_EULA=YES flock <gpu.lock> ~/Documents/LabAutomationTwin/env_isaacsim/bin/python film/sea_test.py
"""

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
sys.argv = [sys.argv[0], "OFF_install@-1"]           # render_ship_clips: no clip range matches -> no clip renders
FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(FILM, "film", "render_ship_clips.py")).read().split("\ntry:\n    t0 = time.time()")[0])

import ocean as OC  # noqa: E402  (on sys.path via render_ship_clips)
import sea as FS  # noqa: E402
from pxr import Sdf, UsdShade  # noqa: E402

OUTD = os.path.join(FILM, "build", "sea_test")
os.makedirs(OUTD, exist_ok=True)
TIMES = [5.0, 13.0, 45.0]
VARS = os.environ.get("SEA_VARS", "ABCD")

stage = open_stage(CLIPS["OFF_install"][0])
vessel_swap.swap(stage)
o = replace_sea(stage)
# albedo variants, written next to the test frames (never over build/ocean)
OC.OUT, keep = OUTD, OC.OUT
FOAMS = {"B": (FS.FilmOcean.WATER, 1.0), "E": ((0.50, 0.52, 0.52), 0.40), "F": ((0.45, 0.47, 0.47), 0.22)}
for v_, (col, gain) in FOAMS.items():
    FS.FilmOcean.FOAM, FS.FilmOcean.COV_GAIN = col, gain
    o._textures(0.0)
    os.rename(os.path.join(OUTD, "sea_albedo.png"), os.path.join(OUTD, f"sea_albedo_{v_}.png"))
OC.OUT = keep
settle(30)
sh = next(UsdShade.Shader(p) for p in stage.Traverse()
          if str(p.GetPath()).startswith(SEA_PATH) and p.GetTypeName() == "Shader")
alb0 = sh.GetInput("diffuse_texture").Get()
rp = rep.create.render_product(f"/World/Process/OFF_install/Camera", RES)
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach(rp)
dep = RZ.depth_annotator(rp)
tcps = stage.GetTimeCodesPerSecond()
tl.set_time_codes_per_second(tcps)
tl.set_end_time(stage.GetEndTimeCode() / tcps)


def variant(v):
    rough = v in "CD"
    alb = "B" if v in "BD" else v if v in FOAMS else None
    sh.GetInput("diffuse_texture").Set(Sdf.AssetPath(os.path.join(OUTD, f"sea_albedo_{alb}.png")) if alb else alb0)
    sh.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(0.0 if rough else 1.0)
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.08)
    sh.CreateInput("detail_bump_factor", Sdf.ValueTypeNames.Float).Set(0.15 if rough else 0.35)


try:
    from PIL import Image, ImageDraw
    for t in TIMES:
        tl.set_current_time(t)
        sea_update(stage, t)
        for v in VARS:
            variant(v)
            settle(5)
            img = grab(annot, 24)
            save(img, os.path.join(OUTD, f"{t:04.1f}_{v}.jpg"), depth_of(dep))
            print(f"[sea_test] {t:.1f}s {v}", flush=True)
    W, H = 960, 540
    sheet = Image.new("RGB", (len(VARS) * W, len(TIMES) * H))
    for i, t in enumerate(TIMES):
        for j, v in enumerate(VARS):
            im = Image.open(os.path.join(OUTD, f"{t:04.1f}_{v}.jpg")).resize((W, H))
            ImageDraw.Draw(im).text((12, 10), f"{v}  t={t:g}s", fill=(255, 255, 0))
            sheet.paste(im, (j * W, i * H))
    sheet.save(os.path.join(OUTD, "sheet.jpg"), quality=88)
    print(os.path.join(OUTD, "sheet.jpg"))
finally:
    app.close()
