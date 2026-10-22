"""Film-side offshore scene and installation clips with the REAL Norse Energi crane side (aft PORT leg) and the new
vessel models - generated from the twin's builders (CuxhavenSiteTwin read-only: their source is loaded by path,
patched in memory, and every output goes to this project).

Patches (each must match exactly once, or the build fails):
  scenes/build_offshore_scene.py  outputs -> build/scenes/windanker_film.usda; twin asset paths absolute; the WTIV
                                  parts reference build/vessels/wtiv_*.usda; crane on leg 2 (aft port), the jack-up
                                  on the turbine's other side (s_ang +75 deg), boom rest slewed -8 deg
  scenes/build_offshore_clip.py   -> build/scenes/process/OFF_install.usda; camera side mirrored to keep the shots
  scenes/build_offshore_rotor.py  -> build/scenes/process/OFF_rotor.usda; camera side mirrored
Also: build/scenes/collisions/<sid>.json from the twin's swept collision checker (python-fcl), run on these layers.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python film/scenes/build_offshore_film.py
     ~/Documents/CuxhavenSiteTwin/.venv_coll/bin/python film/scenes/build_offshore_film.py --check
"""

import math
import os
import sys
import types

import numpy as np

FILM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
SC = os.path.join(TWIN, "scenes")
OUT = os.path.join(FILM, "build", "scenes")
PROC = os.path.join(OUT, "process")
VES = os.path.join(FILM, "build", "vessels") + "/"
CRANE_LEG = 2


def load_patched(name, path, repl):
    src = open(path).read()
    for old, new in repl:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{os.path.basename(path)}: patch target found {n}x: {old!r}")
        src = src.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = path                                     # keep the twin's ROOT for its own imports
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def abs_t(rel):
    return os.path.join(TWIN, rel) + ("/" if rel.endswith("/") else "")


def build():
    os.makedirs(PROC, exist_ok=True)
    for d in ("", "product", "site", os.path.join("assets", "offshore"), "scenes"):
        sys.path.insert(0, os.path.join(TWIN, d))
    import build_offshore as BO
    BO.WT["crane_leg"] = CRANE_LEG
    BO.WT["rest_slew"] = -8.0
    windanker = os.path.join(OUT, "windanker_film.usda")
    SO = load_patched("build_offshore_scene", os.path.join(SC, "build_offshore_scene.py"), [
        ('OUT = os.path.join(ROOT, "offshore", "windanker.usda")', f'OUT = {windanker!r}'),
        ('OFF = "../assets/offshore/usd/"', f'OFF = {abs_t("assets/offshore/usd/")!r}\nVES = {VES!r}'),
        ('NAC = "../assets/nacelle/usd/"', f'NAC = {abs_t("assets/nacelle/usd/")!r}'),
        ('STATES = "../product/usd/states/"', f'STATES = {abs_t("product/usd/states/")!r}'),
        ('"../geo/usd/tex/ocean_normal.png"', repr(abs_t("geo/usd/tex/ocean_normal.png"))),
        ('"../assets/sky/sky_nosun.hdr"', repr(abs_t("assets/sky/sky_nosun.hdr"))),
        ('"../assets/vehicles/usd/rope_unit.usda"', repr(abs_t("assets/vehicles/usd/rope_unit.usda"))),
        ('"../assets/vehicles/usd/hook_heavy.usda"', repr(abs_t("assets/vehicles/usd/hook_heavy.usda"))),
        ('"../assets/vehicles/usd/nacelle_yoke.usda"', repr(abs_t("assets/vehicles/usd/nacelle_yoke.usda"))),
        ('OFF + "wtiv_hull.usda"', 'VES + "wtiv_hull.usda"'),
        ('OFF + "wtiv_leg.usda"', 'VES + "wtiv_leg.usda"'),
        ('OFF + "wtiv_crane.usda"', 'VES + "wtiv_crane.usda"'),
        ('OFF + "wtiv_boom.usda"', 'VES + "wtiv_boom.usda"'),
        ("rad, s_ang = 55.0, -75.0", "rad, s_ang = 55.0, 75.0"),
        # blade rack 2.2 m inboard: with the crane on the port aft leg the blades must rise clear of its pedestal
        ("return Wv @ T(None, (36.0, 8.6, deck + 4.6 + j * 7.5))", "return Wv @ T(None, (36.0, 6.4, deck + 4.6 + j * 7.5))"),
    ])
    rel = os.path.relpath(windanker, PROC)
    load_patched("build_offshore_clip", os.path.join(SC, "build_offshore_clip.py"), [
        ('OUT = os.path.join(ROOT, "scenes", "process", f"{SID}.usda")', f'OUT = os.path.join({PROC!r}, f"{{SID}}.usda")'),
        ('subLayerPaths.append("../../offshore/windanker.usda")', f'subLayerPaths.append({rel!r})'),
        ("    n = np.array([-d[1], d[0]])\n    def P(v):", "    n = -np.array([-d[1], d[0]])\n    def P(v):"),
        # pick the middle deck nacelle (x -6, 50 m from the port crane): boom 77.6 deg instead of 84.7 deg at x -30
        ("slot_v = np.array([-30.0, SO.DECK_NAC_Y])", "slot_v = np.array([-6.0, SO.DECK_NAC_Y])"),
        ('st.OverridePrim("/World/Install/Deck/Nacelle2").SetActive(False)',
         'st.OverridePrim("/World/Install/Deck/Nacelle1").SetActive(False)'),
        ('if p.GetName() != "Nacelle2"]', 'if p.GetName() != "Nacelle1"]'),
    ]).main()
    load_patched("build_offshore_rotor", os.path.join(SC, "build_offshore_rotor.py"), [
        ('OUT = os.path.join(ROOT, "scenes", "process", f"{SID}.usda")', f'OUT = os.path.join({PROC!r}, f"{{SID}}.usda")'),
        ('subLayerPaths.append("../../offshore/windanker.usda")', f'subLayerPaths.append({rel!r})'),
        ('"../../product/usd/states/H6_hub_released.usda"', repr(abs_t("product/usd/states/H6_hub_released.usda"))),
        ('"../../assets/offshore/usd/blade_115.usda"', repr(abs_t("assets/offshore/usd/blade_115.usda"))),
        ('"../../assets/vehicles/usd/lifting_beam.usda"', repr(abs_t("assets/vehicles/usd/lifting_beam.usda"))),
        ("    d = away\n    n = np.array([-d[1], d[0]])", "    d = away\n    n = -np.array([-d[1], d[0]])"),
        # turn the rotor only after the hook has left the hub (it swept the mounted blade through boom and hook) ...
        ("rot_keys += [(t, theta), (t + 5.0, target_th)]", "rot_keys += [(t + 6.5, theta), (t + 11.5, target_th)]"),
        # ... and swing the empty hook back pulled in towards the crane, clear of the tower top and the nacelle
        ("go(cog_deck + [0, 0, RIG_BLADE], 8.0, rise=deck + 40.0)",
         "go(cog_deck + [0, 0, RIG_BLADE], 8.0, rise=tower_top + 30.0)"),
    ]).main()
    del SO
    for path in (windanker, os.path.join(PROC, "OFF_install.usda"), os.path.join(PROC, "OFF_rotor.usda")):
        add_luff_ropes(path)
    build_loadout()


# %% luffing ropes: A-frame sheaves -> boom-head sheaves, re-aimed at every time sample of the boom


def _gm(M):
    from pxr import Gf
    return Gf.Matrix4d(*np.asarray(M, float).T.ravel())


def rope_matrix(a, b):
    """unit +z segment -> segment a..b (crane frame)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    z = b - a
    L = np.linalg.norm(z)
    z /= L
    x = np.cross([0, 1.0, 0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z * L, a
    return M


def add_luff_ropes(path):
    """Author /World/Install/WTIV/Crane/Luff{P,S} in the layer: two rope groups whose ends follow the boom's luff
    (read back from the boom's own transform at every time sample, default included)."""
    from pxr import Usd, UsdGeom
    sys.path.insert(0, os.path.join(FILM, "film", "vessels"))
    import build_vessels as BV
    st = Usd.Stage.Open(path)
    boom = st.GetPrimAtPath("/World/Install/WTIV/Crane/Boom")
    if not boom:
        raise SystemExit(f"{path}: no boom")
    xb = UsdGeom.Xformable(boom)
    times = sorted({t for op in xb.GetOrderedXformOps() for t in op.GetTimeSamples()})
    ops = {}
    for side, sy in (("P", 1.0), ("S", -1.0)):
        x = UsdGeom.Xform.Define(st, f"/World/Install/WTIV/Crane/Luff{side}")
        x.GetPrim().GetReferences().ClearReferences()
        x.GetPrim().GetReferences().AddReference(os.path.join(VES, "wtiv_luff_rope.usda"))
        x.ClearXformOpOrder()
        ops[side] = (x.AddTransformOp(), sy)
    af = lambda sy: np.array([-13.0, sy * 10.5, BV.AF_Z])  # noqa: E731
    luffs = []

    def put(tc):
        Mb = np.array(xb.GetLocalTransformation(tc)).T               # boom frame -> crane frame
        lf = math.degrees(math.atan2(Mb[2, 0], Mb[0, 0]))
        luffs.append(lf)
        for op, sy in ops.values():
            tip = (Mb @ np.array([BV.LUFF_ATT[0], sy * BV.LUFF_ATT[1], BV.LUFF_ATT[2], 1.0]))[:3]
            M = rope_matrix(af(sy), tip)
            if tc == Usd.TimeCode.Default():
                op.Set(_gm(M))
            else:
                op.Set(_gm(M), tc)
    put(Usd.TimeCode.Default())
    for t in times:
        put(Usd.TimeCode(t))
    st.GetRootLayer().Save()
    working = [l for l in luffs if l > 5.0]
    print(f"  luffing ropes in {os.path.basename(path)}: {len(times)} samples; boom luff working range "
          f"{min(working) if working else 0:.1f}..{max(luffs):.1f} deg, rest {min(luffs):.1f} deg")


# %% load-out: the twin's L_loadout layer copied film-side, with the Rotra Vente models referenced


def build_loadout(next_=False):
    """film-side L_loadout; next_: from the twin's staged scenes/process_next/L_loadout.usda when it exists
    (written to build/scenes/process_next/), else from scenes/process."""
    from pxr import Sdf
    src_dir = os.path.join(SC, "process_next") if next_ and os.path.exists(
        os.path.join(SC, "process_next", "L_loadout.usda")) else os.path.join(SC, "process")
    src = Sdf.Layer.FindOrOpen(os.path.join(src_dir, "L_loadout.usda"))
    out_dir = os.path.join(OUT, "process_next") if next_ else PROC
    os.makedirs(out_dir, exist_ok=True)
    dst_path = os.path.join(out_dir, "L_loadout.usda")
    print(f"  L_loadout source: {src_dir}")
    if os.path.exists(dst_path):
        os.remove(dst_path)
    dst = Sdf.Layer.CreateNew(dst_path)
    dst.TransferContent(src)
    dst.subLayerPaths[:] = [os.path.join(src_dir, sl) if not os.path.isabs(sl) else sl
                            for sl in src.subLayerPaths]
    dst.subLayerPaths[:] = [os.path.normpath(sl) for sl in dst.subLayerPaths]
    from pxr import UsdUtils
    base = src_dir
    UsdUtils.ModifyAssetPaths(dst, lambda a: os.path.normpath(os.path.join(base, a))
                              if a and not os.path.isabs(a) and not a.startswith(("omniverse:", "http")) else a)
    for prim, asset in (("/World/Port/Vessel/Hull", "roro_vessel.usda"), ("/World/Port/Vessel/Visor", "roro_visor.usda")):
        spec = Sdf.CreatePrimInLayer(dst, prim)
        if spec.specifier != Sdf.SpecifierDef:
            spec.specifier = Sdf.SpecifierOver
        spec.referenceList.explicitItems = [Sdf.Reference(os.path.join(VES, asset))]
    dst.Save()
    print(f"  wrote {dst_path} (sublayers {list(dst.subLayerPaths)})")


def check():
    os.environ["COLL_PROC_DIR"] = PROC                         # the checker's own switches (absolute paths)
    os.environ["COLL_OUT"] = os.path.join(OUT, "collisions")
    CC = load_patched("collision_check", os.path.join(TWIN, "scripts", "collision_check.py"), [])
    os.makedirs(os.path.join(OUT, "collisions"), exist_ok=True)
    for sid in ([a for a in sys.argv[1:] if not a.startswith("--") and a != "0.5"] or
                ("L_loadout", "OFF_install", "OFF_rotor")):
        CC.check(sid)


if __name__ == "__main__":
    if "--next" in sys.argv:
        build_loadout(next_=True)
        for sid in ("OFF_install", "OFF_rotor"):
            if os.path.exists(os.path.join(SC, "process_next", f"{sid}.usda")):
                print(f"  NOTE: the twin staged {sid} in process_next; the film-side {sid} is generated from the "
                      "builders (crane on the port leg) - rebuild without --next after the builders change")
    elif "--check" in sys.argv:
        sys.argv = [sys.argv[0], "--dt", "0.5"]
        check()
    else:
        build()
