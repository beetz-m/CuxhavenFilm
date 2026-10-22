"""Film-side surface detail for the plain turbine materials (nacelle canopy, tower, transition piece, monopile, blades).

These five materials of the twin's shared library (assets/materials/site_materials.usda) are flat constant colours;
the twin's realism pass has no texture set for them. Here they get procedural texture sets at real scale (the mesh
UVs are metres; on the tower and blades v runs along the height / length):
  NacelleWhite  GRP canopy, RAL 9016-like: orange-peel gelcoat normal, gentle mottling and roughness variation
  TowerGrey     painted steel, RAL 7035-like: circumferential can welds every 3.56 m, a flange joint every 28.5 m
                (bolt row and a darker gasket line), faint rain streaks, mottling
  TPYellow      RAL 1023-like coating, weathered: chips, grime streaks, a darker splash band
  MonopileGrey  dark coating with mottling
  BladeWhite    gelcoat with chordwise smoothness, faint lengthwise streaks
Applied in memory exactly like the twin's realism.apply_materials() (the library layer is edited in the registry,
never saved). Values are ILLUSTRATIVE (appearance only).

    import product_looks as PL
    PL.apply()            # after RZ.apply_materials(), before opening the stage
"""

from __future__ import annotations

import os

import numpy as np

FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
LIB = os.path.join(TWIN, "assets", "materials", "site_materials.usda")
OUT = os.path.join(FILM, "build", "textures")
_KEEP = []


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


PA = {"v": 1.0}          # texel aspect (u size / v size) of the set being built: noise isotropic in metres


def _noise(shape, sigma, seed, aniso=(1.0, 1.0)):
    """Periodic smooth noise (Gaussian-filtered white noise), unit std; sigma in (v) px, aniso = (rows, cols)
    frequency scale (< 1 stretches features along that axis), corrected by the texel aspect PA."""
    aniso = (aniso[0], aniso[1] * PA["v"])
    h, w = shape
    r = np.random.default_rng(seed).standard_normal(shape)
    fy = np.fft.fftfreq(h)[:, None] * aniso[0]
    fx = np.fft.fftfreq(w)[None, :] * aniso[1]
    n = np.real(np.fft.ifft2(np.fft.fft2(r) * np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))))
    return (n - n.mean()) / (n.std() + 1e-9)


def _normal(height, strength):
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5 * strength
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 0.5 * strength
    n = np.stack([-gx, gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def _save(name, alb_lin, rough, height, strength):
    from PIL import Image
    os.makedirs(OUT, exist_ok=True)
    flip = lambda a: a[::-1]                      # noqa: E731  image row 0 = v max
    paths = {k: os.path.join(OUT, f"{name}_{k}.png") for k in ("albedo", "rough", "normal")}
    Image.fromarray((flip(_srgb(alb_lin)) * 255).astype(np.uint8)).save(paths["albedo"])
    Image.fromarray((flip(np.clip(rough, 0, 1)) * 255).astype(np.uint8)).save(paths["rough"])
    Image.fromarray(flip(_normal(height, strength))).save(paths["normal"])
    return paths


# ------------------------------------------------------------------------------------------------- texture sets
def nacelle_white():
    N, tile = 1024, 4.0                                       # 4 mm / px; GRP panels 2.0 m x 1.33 m
    u = (np.arange(N) + 0.5) / N * tile
    base = np.array([0.74, 0.75, 0.74])
    mott = 0.05 * _noise((N, N), 60, 1) + 0.02 * _noise((N, N), 12, 2)
    grime = np.clip(_noise((N, N), 30, 3, aniso=(0.12, 1.0)), 0, None) * 0.05     # run-off below joints
    du = np.minimum(u % 2.0, 2.0 - u % 2.0)                                       # vertical joints every 2.0 m
    dv = np.minimum(u % 1.3333, 1.3333 - u % 1.3333)                              # horizontal joints every 1.33 m
    seam = np.maximum(np.exp(-(du / 0.006) ** 2)[None, :], np.exp(-(dv / 0.006) ** 2)[:, None])
    shadow = np.maximum(np.exp(-(du / 0.03) ** 2)[None, :], np.exp(-(dv / 0.03) ** 2)[:, None])
    k = (1 + mott - grime) * (1 - 0.35 * seam) * (1 - 0.05 * shadow)
    alb = base[None, None] * k[..., None]
    peel = _noise((N, N), 1.6, 4) * 0.6 + _noise((N, N), 5, 5) * 0.4               # gelcoat orange peel
    rough = 0.30 + 0.08 * _noise((N, N), 50, 6) + 0.03 * peel + 0.2 * seam + 0.06 * grime / 0.05
    height = 0.25 * peel - 3.0 * seam
    return _save("nacelle_white", alb, rough, height, 0.6), (tile, tile)


def tower_grey():
    H, W, tile_v, tile_u = 4096, 256, 28.5, 4.0              # v: 7 mm / px along the height
    v = (np.arange(H) + 0.5) / H * tile_v
    base = np.array([0.58, 0.59, 0.57])
    mott = 0.045 * _noise((H, W), 40, 11) + 0.015 * _noise((H, W), 8, 12)
    streak = np.clip(_noise((H, W), 6, 13, aniso=(0.04, 1.0)), 0, None) * 0.035    # rain streaks (long in v)
    height = np.zeros((H, W))
    shade = np.ones(H)
    can = 3.5625                                               # 8 cans per section
    for z in np.arange(0, tile_v, can):
        d = np.abs(v - z)
        weld = np.exp(-(d / 0.015) ** 2)
        height += (weld * 2.0)[:, None]
        shade -= 0.10 * weld
    d = np.minimum(np.abs(v - 0.0), np.abs(v - tile_v))      # flange joint at the tile edge
    flange = (d < 0.05).astype(float)
    shade -= 0.25 * np.exp(-(d / 0.008) ** 2)                  # gasket line
    height += (flange * 2.0 + np.exp(-(d / 0.05) ** 2))[:, None]
    bolts = np.zeros((H, W))
    rows = np.where((d > 0.05) & (d < 0.09))[0]
    cols = (np.arange(W) % 32) < 6
    bolts[np.ix_(rows, np.where(cols)[0])] = 1.0
    height += bolts * 1.5
    alb = base[None, None] * (shade[:, None] * (1 + mott - streak) - 0.10 * bolts)[..., None]
    rough = 0.34 + 0.05 * _noise((H, W), 30, 14) + 0.06 * streak / 0.035
    return _save("tower_grey", alb, rough, height, 0.9), (tile_u, tile_v)


def tp_yellow():
    H, W, tile_v, tile_u = 2048, 1024, 16.0, 8.0
    v = (np.arange(H) + 0.5) / H * tile_v
    base = np.array([0.80, 0.52, 0.02])
    mott = 0.05 * _noise((H, W), 50, 21) + 0.02 * _noise((H, W), 6, 22)
    run = np.clip(_noise((H, W), 4, 23, aniso=(0.05, 1.0)), 0, None) * 0.06       # rust / grime run-off
    chips = (_noise((H, W), 1.2, 24) > 2.9).astype(float)                          # small paint chips
    splash = np.clip(1 - v / 3.0, 0, 1)[:, None] * 0.25                             # splash band at the tile base
    k = np.clip(1 + mott - run - splash, 0.3, 1.2)
    alb = base[None, None] * k[..., None]
    rust = np.array([0.18, 0.08, 0.03])
    alb = alb * (1 - chips[..., None]) + rust[None, None] * chips[..., None]
    rough = 0.45 + 0.08 * _noise((H, W), 30, 25) + 0.25 * chips + 0.1 * splash
    return _save("tp_yellow", alb, rough, -chips * 0.8 + 0.2 * _noise((H, W), 3, 26), 0.6), (tile_u, tile_v)


def monopile_grey():
    N, tile = 1024, 8.0
    base = np.array([0.20, 0.21, 0.22])
    k = 1 + 0.07 * _noise((N, N), 40, 31) - np.clip(_noise((N, N), 5, 32, aniso=(0.05, 1)), 0, None) * 0.05
    alb = base[None, None] * k[..., None]
    rough = 0.55 + 0.08 * _noise((N, N), 30, 33)
    return _save("monopile_grey", alb, rough, 0.3 * _noise((N, N), 3, 34), 0.5), (tile, tile)


def blade_white():
    H, W, tile_v, tile_u = 4096, 256, 32.0, 4.0
    base = np.array([0.78, 0.79, 0.77])
    k = 1 + 0.015 * _noise((H, W), 60, 41) - np.clip(_noise((H, W), 4, 42, aniso=(0.03, 1)), 0, None) * 0.012
    alb = base[None, None] * k[..., None]
    rough = 0.22 + 0.04 * _noise((H, W), 40, 43)
    return _save("blade_white", alb, rough, 0.3 * _noise((H, W), 2, 44), 0.25), (tile_u, tile_v)


SETS = {"NacelleWhite": nacelle_white, "TowerGrey": tower_grey, "TPYellow": tp_yellow,
        "MonopileGrey": monopile_grey, "BladeWhite": blade_white}


def build(force=False):
    """Texture sets (cached in build/textures); returns {material: (paths, (tile_u_m, tile_v_m))}."""
    import json
    meta = os.path.join(OUT, "product_looks.json")
    if os.path.exists(meta) and not force and os.path.getmtime(meta) > os.path.getmtime(__file__):
        return {k: (v[0], tuple(v[1])) for k, v in json.load(open(meta)).items()}
    out = {}
    for k, f in SETS.items():
        PA["v"] = {"TowerGrey": (4.0 / 256) / (28.5 / 4096), "BladeWhite": (4.0 / 256) / (32.0 / 4096)}.get(k, 1.0)
        out[k] = f()
    PA["v"] = 1.0
    json.dump(out, open(meta, "w"), indent=1)
    return out


def apply(verbose=True):
    """Edit the shared material library in memory (never saved): the five plain materials get the texture sets."""
    from pxr import Sdf
    sets = build()
    layer = Sdf.Layer.FindOrOpen(LIB)
    _KEEP.append(layer)
    done = []
    with Sdf.ChangeBlock():
        for name, (p, (tu, tv)) in sets.items():
            sh = layer.GetPrimAtPath(f"/Looks/{name}/Shader")
            if sh is None:
                continue

            def put(attr, tname, value, raw=False):
                a = sh.attributes.get(f"inputs:{attr}")
                if a is None:
                    a = Sdf.AttributeSpec(sh, f"inputs:{attr}", tname)
                a.default = value
                if raw:
                    a.SetInfo("colorSpace", "raw")
            put("diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(p["albedo"]))
            put("diffuse_tint", Sdf.ValueTypeNames.Color3f, (1.0, 1.0, 1.0))
            put("reflectionroughness_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(p["rough"]), raw=True)
            put("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 1.0)
            put("normalmap_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(p["normal"]), raw=True)
            put("bump_factor", Sdf.ValueTypeNames.Float, 1.0)
            put("texture_scale", Sdf.ValueTypeNames.Float2, (1.0 / tu, 1.0 / tv))
            done.append(name)
    if verbose:
        print(f"[looks] {len(done)} turbine materials textured (in memory): {done}", flush=True)
    return done


if __name__ == "__main__":
    from PIL import Image
    sets = build(force=True)
    tiles = []
    for name, (p, _) in sets.items():
        im = Image.open(p["albedo"]).convert("RGB")
        tiles.append(im.resize((256, 512)))
    sheet = Image.new("RGB", (256 * len(tiles), 512))
    for i, t in enumerate(tiles):
        sheet.paste(t, (256 * i, 0))
    sheet.save(os.path.join(OUT, "product_looks_preview.png"))
    print(os.path.join(OUT, "product_looks_preview.png"))
