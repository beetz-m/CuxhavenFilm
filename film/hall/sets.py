"""Which of the twin's process stages the hall renderer and the collision check open (pxr only; shared by both).

  current set:  scenes/process/<sid>.usda (sublayers scenes/cuxhaven_site.usda)
  next set:     scenes/process_next/<sid>.usda when the twin session has staged it (it sublayers
                scenes/cuxhaven_site_next.usda); for a station not staged yet, a film-side copy of scenes/process/<sid>.usda
                whose site sublayer is remapped to cuxhaven_site_next.usda and whose asset paths are made absolute
                (build/hall/next_layers/<sid>.usda). The twin's files are never written.
Select with --next on the command line or HALL_PROCESS_SET=process_next.
"""

import os

from pxr import Sdf, UsdUtils

TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
FILM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENES = os.path.join(TWIN, "scenes")


def use_next(argv=()):
    return "--next" in argv or os.environ.get("HALL_PROCESS_SET", "process") == "process_next"


def site_layer(next_):
    return os.path.join(SCENES, "cuxhaven_site_next.usda" if next_ else "cuxhaven_site.usda")


def process_layer(sid, next_):
    """Path of the stage to open for station clip `sid`."""
    if not next_:
        return os.path.join(SCENES, "process", f"{sid}.usda")
    staged = os.path.join(SCENES, "process_next", f"{sid}.usda")
    if os.path.exists(staged):
        return staged
    src = os.path.join(SCENES, "process", f"{sid}.usda")
    dst = os.path.join(FILM, "build", "hall", "next_layers", f"{sid}.usda")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    lay = Sdf.Layer.FindOrOpen(src)
    out = Sdf.Layer.CreateNew(dst)
    out.TransferContent(lay)
    base = os.path.dirname(src)

    def absolute(p):
        if not p or "://" in p or os.path.isabs(p):
            return p
        a = os.path.normpath(os.path.join(base, p))
        if os.path.basename(a) == "cuxhaven_site.usda":
            return site_layer(True)
        return a

    UsdUtils.ModifyAssetPaths(out, absolute)
    out.subLayerPaths = [absolute(p) for p in lay.subLayerPaths]
    out.Save()
    return dst
