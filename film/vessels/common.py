"""Shared pieces for the film's vessel models: the twin's mesh kit (loaded read-only by path) pointed at a film-side
material library, a lofted-hull generator, raised lettering from font outlines, railings and window bands.

The film-side library build/vessels/materials.usda sublayers the twin's site library (unchanged) and adds vessel
paints and textures; the vessel assets bind through it.
"""

from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

FILM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
OUT = os.path.join(FILM, "build", "vessels")
MATLIB = os.path.join(OUT, "materials.usda")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KIT = _load("cux_kit", os.path.join(TWIN, "assets", "common", "kit.py"))
meshkit = KIT.meshkit
Kit, new_asset, sdt = KIT.Kit, KIT.new_asset, KIT.sdt
SM = KIT.SM
LM = SM.LM
meshkit.MATLIB = MATLIB

# vessel paints (OmniPBR: linear diffuse, roughness, metallic); colours READ-FROM-PHOTO (reference/vessels)
PBR = {
    "RotraBlue": ((0.012, 0.10, 0.42), 0.38, 0.0),        # hull, approx sRGB (24, 86, 170)
    "RotraBlueDark": ((0.005, 0.035, 0.15), 0.55, 0.0),   # recessed grid bays of the hold bulwark
    "Antifouling": ((0.022, 0.022, 0.025), 0.62, 0.0),    # black boot-top / bottom
    "RotraOrange": ((0.86, 0.24, 0.02), 0.40, 0.0),       # wheelhouse roof band, visor arms
    "BridgeGlass": ((0.012, 0.016, 0.022), 0.04, 0.0),    # wheelhouse / accommodation windows
    "RoofDark": ((0.10, 0.105, 0.11), 0.55, 0.0),         # collapsible hold roof
    "DemeGreen": ((0.10, 0.56, 0.05), 0.36, 0.0),         # Norse Energi hull, approx sRGB (70, 190, 60)
    "DeckGreen": ((0.06, 0.26, 0.05), 0.60, 0.0),         # aft working deck paint
    "LetterWhite": ((0.90, 0.90, 0.90), 0.35, 0.0),       # ship names, hull marks
    "SovBlue": ((0.018, 0.06, 0.20), 0.38, 0.0),          # generic SOV hull
    "SovBoot": ((0.28, 0.07, 0.04), 0.60, 0.0),           # SOV boot-top (red-brown)
    "HeliDeck": ((0.05, 0.09, 0.07), 0.70, 0.0),          # helideck surface
    "RescueOrange": ((0.95, 0.38, 0.04), 0.45, 0.0),      # lifeboats / rescue boats
    "Rubber": ((0.03, 0.03, 0.03), 0.85, 0.0),            # fenders
    "LegBlack": ((0.035, 0.038, 0.042), 0.45, 0.25),      # lattice legs
    "BoomRed": ((0.62, 0.035, 0.03), 0.40, 0.0),          # crane boom chords, hook block
}
TEXTURED = {  # name: (png, base colour (linear), roughness, tile m)
    "HoldWall": ("hold_wall.png", (0.62, 0.64, 0.66), 0.45, 2.0),        # light grey corrugated panels
    "DeckPlate": ("deck_plate.png", (0.46, 0.47, 0.48), 0.70, 6.0),     # grey steel deck with weld seams
    "GratingYellow": ("grating_yellow.png", (0.85, 0.62, 0.04), 0.65, 1.0),
    "GratingGreen": ("grating_green.png", (0.10, 0.45, 0.06), 0.65, 1.0),
}
for _n, (_p, _c, _r, _t) in TEXTURED.items():
    meshkit.TILE_M[_n] = _t


def _save_png(name, rgb):
    from PIL import Image
    os.makedirs(os.path.join(OUT, "textures"), exist_ok=True)
    arr = (np.clip(rgb, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, "textures", name))


def make_textures():
    rng = np.random.default_rng(3)
    n = 512
    yy, xx = np.mgrid[0:n, 0:n] / n
    # corrugated wall: 4 trapezoidal ribs per 2 m (u horizontal), a seam at the tile edge, faint streaks
    ph = (xx * 4) % 1.0
    shade = np.where(ph < 0.10, 0.80, np.where(ph < 0.18, 1.06, np.where(ph < 0.60, 1.0, np.where(ph < 0.68, 0.88, 0.97))))
    shade = shade * (1 + 0.015 * rng.normal(size=(1, n)).repeat(n, 0)) * (0.97 + 0.03 * yy)
    base = np.array(TEXTURED["HoldWall"][1])
    _save_png("hold_wall.png", shade[..., None] * base)
    # deck plate: welded plates (2 x 6 m tile -> seams), low-frequency wear
    d = 1.0 + 0.04 * LM._tileable_lowfreq(rng, n, 10) + 0.02 * rng.normal(size=(n, n))
    d[:, :2] *= 0.75
    d[:2, :] *= 0.75
    d[n // 3:n // 3 + 2, :] *= 0.8
    d[2 * n // 3:2 * n // 3 + 2, :] *= 0.8
    _save_png("deck_plate.png", d[..., None] * np.array(TEXTURED["DeckPlate"][1]))
    # grating: 8 x 8 bars per metre, dark gaps
    g = ((xx * 32) % 1.0 < 0.28) | ((yy * 32) % 1.0 < 0.28)
    for name in ("GratingYellow", "GratingGreen"):
        col = np.array(TEXTURED[name][1])
        _save_png(TEXTURED[name][0], np.where(g[..., None], col, col * 0.18))


# realistic texture sets (another session's 2048 px tileable PBR maps, read-only by path): albedo derived per paint
# (source pattern x our colour) into build/vessels/textures, roughness and normal maps used as they are
REAL_DIR = os.path.join(TWIN, "assets", "materials")
REAL_SRC = {"RotraBlue": "HullBlue", "RotraBlueDark": "HullBlue", "Antifouling": "HullBlack", "RotraOrange": "PaintOrange",
            "DemeGreen": "HullGreen", "DeckGreen": "HullGreen", "LegBlack": "LegDark", "BoomRed": "RailingRed",
            "SovBlue": "HullBlue", "SovBoot": "HullRed", "RescueOrange": "PaintOrange", "RoofDark": "SteelPaintDark",
            "LetterWhite": "ShipWhite"}


def real_spec():
    import json
    p = os.path.join(REAL_DIR, "realistic.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def derive_albedo(name, src, colour, size=1024):
    """our paint colour carrying the source map's grime / scratch pattern (luminance ratio to its mean)."""
    from PIL import Image
    spec = real_spec()[src]
    a = np.asarray(Image.open(os.path.join(REAL_DIR, spec["albedo"])).convert("RGB").resize((size, size), Image.LANCZOS),
                   np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ratio = lum / max(1e-4, float(np.median(lum)))
    ratio = np.clip(ratio, 0.35, 2.2)
    out = np.clip(ratio[..., None] * np.asarray(colour, np.float32), 0, 1)
    fn = f"{name}_albedo.png"
    _save_png(fn, out)
    return fn, spec


def build_materials():
    """build/vessels/materials.usda: the twin's site library (sublayer, unchanged) + vessel paints."""
    make_textures()
    if os.path.exists(MATLIB):
        os.remove(MATLIB)
    st = Usd.Stage.CreateNew(MATLIB)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    st.SetMetadata("comment", "Generated by film/vessels/common.py - edit the script, not this file.")
    st.GetRootLayer().subLayerPaths.append(os.path.relpath(os.path.join(TWIN, "assets", "materials",
                                                                        "site_materials.usda"), OUT))
    F, C3, A = LM.F, LM.C3, LM.A
    F2 = Sdf.ValueTypeNames.Float2
    real = real_spec()
    for name, (diff, rough, metal) in PBR.items():
        _, sh = LM._shader(st, f"/Looks/{name}", "OmniPBR")
        LM._set(sh, "diffuse_color_constant", C3, Gf.Vec3f(*diff))
        LM._set(sh, "reflection_roughness_constant", F, rough)
        LM._set(sh, "metallic_constant", F, metal)
        src = REAL_SRC.get(name)
        if src and src in real:                  # realistic paint: grime, orange peel, scratches (mesh UVs in metres)
            fn, spec = derive_albedo(name, src, diff)
            sc = 1.0 / spec["tile_m"]
            LM._set(sh, "diffuse_texture", A, Sdf.AssetPath(f"./textures/{fn}"))
            LM._set(sh, "reflectionroughness_texture", A, Sdf.AssetPath(os.path.join(REAL_DIR, spec["rough"])), raw=True)
            LM._set(sh, "reflection_roughness_texture_influence", F, 1.0)
            LM._set(sh, "normalmap_texture", A, Sdf.AssetPath(os.path.join(REAL_DIR, spec["normal"])), raw=True)
            LM._set(sh, "bump_factor", F, float(spec["bump"]))
            LM._set(sh, "texture_scale", F2, Gf.Vec2f(sc, sc))
    for name, (png, col, rough, _t) in TEXTURED.items():
        _, sh = LM._shader(st, f"/Looks/{name}", "OmniPBR")
        LM._set(sh, "diffuse_texture", A, Sdf.AssetPath(f"./textures/{png}"))
        LM._set(sh, "reflection_roughness_constant", F, rough)
        LM._set(sh, "metallic_constant", F, 0.0)
        if "SteelPaintGrey" in real and name in ("DeckPlate", "HoldWall"):
            spec = real["SteelPaintGrey"]
            LM._set(sh, "normalmap_texture", A, Sdf.AssetPath(os.path.join(REAL_DIR, spec["normal"])), raw=True)
            LM._set(sh, "bump_factor", F, 0.6)
            LM._set(sh, "detail_normalmap_texture", A, Sdf.AssetPath(os.path.join(REAL_DIR, spec["normal"])), raw=True)
            LM._set(sh, "detail_texture_scale", F2, Gf.Vec2f(_t / spec["tile_m"], _t / spec["tile_m"]))
    st.SetDefaultPrim(st.GetPrimAtPath("/Looks"))
    st.GetRootLayer().Save()
    return len(PBR) + len(TEXTURED)


def write(name, k, sem, comment="Generated by film/vessels/build_vessels.py - edit the script, not this file."):
    st, root = new_asset(os.path.join(OUT, f"{name}.usda"), "Asset", comment)
    n = k.write(st, root)
    sdt(st.GetPrimAtPath(root), **sem)
    st.GetRootLayer().Save()
    print(f"  {name}: {n} triangles")
    return n


# %% geometry helpers


def _grid_mesh(k, mats, G, flip=False):
    """Quad grid G[j][i] (rows j, columns i) -> triangles with smooth vertex normals; mats[j] = material of band j."""
    G = np.asarray(G, float)
    nj, ni = G.shape[:2]
    du = np.gradient(G, axis=1)
    dv = np.gradient(G, axis=0)
    N = np.cross(du, dv)
    N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-12
    if flip:
        N = -N
    for j in range(nj - 1):
        P = np.concatenate([G[j], G[j + 1]])
        Nn = np.concatenate([N[j], N[j + 1]])
        T = []
        for i in range(ni - 1):
            a, b, c, d = i, i + 1, ni + i + 1, ni + i
            T += [[a, b, c], [a, c, d]] if not flip else [[a, c, b], [a, d, c]]
        if not T:
            continue
        # drop degenerate triangles (collapsed stem points)
        T = np.array(T)
        A = np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1)
        T = T[A > 1e-9]
        if len(T):
            k.add(mats[j], P, Nn, T)


def outward_flip(side):
    """flip flag for an outward-facing hull grid on side -1 (starboard) / +1 (port)."""
    return side > 0


def loft_side(k, zlev, xa, xb, hb, mats, nu=90, side=-1, spacing="cos"):
    """One side shell of a lofted hull: for each waterline z in zlev, x runs from xa(z) to xb(z) and y = side*hb(x, z).
    mats[j] = material between zlev[j] and zlev[j+1]. Returns the grid (for closures)."""
    us = np.linspace(0, 1, nu)
    if spacing == "cos":
        us = 0.5 - 0.5 * np.cos(np.pi * us)
    G = []
    for z in zlev:
        a, b = xa(z), xb(z)
        row = []
        for u in us:
            x = a + (b - a) * u
            row.append((x, side * hb(x, z), z))
        G.append(row)
    G = np.array(G)
    # du x dv = x x z = -y: outward on the starboard side as is, flipped on the port side
    _grid_mesh(k, mats, G, flip=outward_flip(side))
    return G


def plate_from_outline(k, mat, outline, z, down=False, holes=()):
    """Horizontal plate from a closed outline (earcut, optional holes)."""
    import mapbox_earcut as ec
    rings = [np.asarray(outline, float)[:, :2]] + [np.asarray(h, float)[:, :2] for h in holes]
    V = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = ec.triangulate_float64(V, ends).reshape(-1, 3)
    P = np.c_[V, np.full(len(V), z)]
    # earcut keeps the input winding; make the normal face up (or down)
    if len(tri):
        n0 = np.cross(P[tri[0, 1]] - P[tri[0, 0]], P[tri[0, 2]] - P[tri[0, 0]])
        if (n0[2] < 0) != down:
            tri = tri[:, ::-1]
    k.add(mat, P, np.tile([0, 0, -1.0 if down else 1.0], (len(P), 1)), tri)


def strip(k, mat, A, B):
    """Quad strip between two polylines A[i], B[i] (both sides rendered: two windings)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    P = np.concatenate([A, B])
    n = len(A)
    T = []
    for i in range(n - 1):
        T += [[i, i + 1, n + i + 1], [i, n + i + 1, n + i]]
    T = np.array(T)
    Nf = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    nrm = Nf.mean(0)
    nrm /= np.linalg.norm(nrm) + 1e-12
    k.add(mat, P, np.tile(nrm, (len(P), 1)), T)
    k.add(mat, P, np.tile(-nrm, (len(P), 1)), T[:, ::-1])


def rail(k, pts, h=1.1, mat="SafetyYellow", r=0.035, post=1.6, mid=True):
    """Guard rail along a polyline on a deck: top + mid rail, posts every `post` metres."""
    p = np.asarray(pts, float)
    top = p + [0, 0, h]
    k.tube(mat, top, r, seg=6)
    if mid:
        k.tube(mat, p + [0, 0, h * 0.5], r * 0.7, seg=5)
    seglen = np.linalg.norm(np.diff(p, axis=0), axis=1)
    for (a, b), L in zip(zip(p[:-1], p[1:]), seglen):
        nposts = max(1, int(L / post))
        for t in np.linspace(0, 1, nposts + 1)[:-1]:
            q = a + (b - a) * t
            k.cyl(mat, q, q + [0, 0, h], r * 0.8, seg=5, caps=False)
    k.cyl(mat, p[-1], p[-1] + [0, 0, h], r * 0.8, seg=5, caps=False)


def window_band(k, a, b, z0, h, normal, pitch=1.3, frame="ShipWhite", out=0.03, mullion=0.12):
    """Dark glass band from a to b (xy points on a wall), bottom z0, height h, with mullions every `pitch` m."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    nrm = np.asarray(normal, float)
    nrm = nrm / np.linalg.norm(nrm)
    d = b - a
    L = np.linalg.norm(d)
    t = d / L
    c = (a + b) / 2 + nrm[:2] * out
    k.quad("BridgeGlass", (c[0], c[1], z0 + h / 2), (nrm[0], nrm[1], 0.0), (0, 0, 1), L, h)
    for s in np.arange(0.0, L + 1e-6, pitch):
        q = a + t * s + nrm[:2] * (out + 0.02)
        k.box(frame, (q[0], q[1], z0 + h / 2), (abs(t[0]) * mullion + abs(nrm[0]) * 0.06 + 1e-3,
                                                abs(t[1]) * mullion + abs(nrm[1]) * 0.06 + 1e-3, h))


def text_polygons(text, height, font, italic=0.0, spacing=1.0):
    """Glyph polygons (shapely) of `text`, cap height ~ `height` m, baseline at y = 0, starting at x = 0."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from shapely.geometry import Polygon
    from shapely.ops import unary_union  # noqa: F401
    fp = FontProperties(fname=font)
    cap = TextPath((0, 0), "H", size=1.0, prop=fp).get_extents().height
    scale = height / cap
    polys = []
    x = 0.0
    bar = TextPath((0, 0), "|", size=1.0, prop=fp).get_extents().width
    for ch in text:
        if ch == " ":
            x += 0.30 * spacing
            continue
        tp = TextPath((0, 0), ch, size=1.0, prop=fp)
        adv = TextPath((0, 0), ch + "|", size=1.0, prop=fp).get_extents().width - bar
        geom = None
        for ring in tp.to_polygons():
            if len(ring) < 3:
                continue
            p = Polygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            geom = p if geom is None else geom.symmetric_difference(p)
        if geom is not None and not geom.is_empty:
            from shapely import affinity
            geom = affinity.affine_transform(geom, [scale, italic * scale, 0, scale, x * scale, 0])
            polys.append(geom)
        x += adv * spacing
    total = x * scale
    return polys, total


def letters(k, polys, mapping, mat="LetterWhite"):
    """Triangulate glyph polygons and map 2D (u, v) to 3D with mapping(u, v) -> (point, normal)."""
    import mapbox_earcut as ec
    for g in polys:
        geoms = getattr(g, "geoms", [g])
        for p in geoms:
            if p.is_empty or p.area < 1e-6:
                continue
            rings = [np.asarray(p.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in p.interiors]
            V = np.concatenate(rings)
            ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
            tri = ec.triangulate_float64(V, ends).reshape(-1, 3)
            P, N = zip(*[mapping(u, v) for u, v in V])
            P, N = np.array(P), np.array(N)
            if len(tri) == 0:
                continue
            n0 = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
            flip = (n0 * N[tri[:, 0]]).sum(1) < 0
            tri[flip] = tri[flip][:, ::-1]
            k.add(mat, P, N, tri)


def hull_text(k, text, hb, side, x_start, z_base, height, font, italic=0.0, out=0.035, spacing=1.0,
              towards_bow=True, mat="LetterWhite"):
    """Name on a lofted hull side. side -1 = starboard, +1 = port; text reads left to right for a viewer outside
    (starboard: towards +x; port: towards -x). x_start is the aft end (starboard) / fore end (port) of the text."""
    polys, total = text_polygons(text, height, font, italic, spacing)
    sgn_u = 1.0 if side < 0 else -1.0

    def mapping(u, v):
        x = x_start + sgn_u * u
        z = z_base + v
        y = side * hb(x, z)
        dx = 0.05
        dy_dx = side * (hb(x + dx, z) - hb(x - dx, z)) / (2 * dx)
        dy_dz = side * (hb(x, z + dx) - hb(x, z - dx)) / (2 * dx)
        n = np.array([-dy_dx, 1.0, -dy_dz]) * side
        n /= np.linalg.norm(n)
        return np.array([x, y, z]) + n * out, n
    letters(k, polys, mapping, mat)
    return total


def plane_text(k, text, origin, u_dir, v_dir, height, font, italic=0.0, out=0.03, center=True, mat="LetterWhite"):
    """Text on a plane: origin (baseline start, or centre if center), u_dir reading direction, v_dir up."""
    polys, total = text_polygons(text, height, font, italic)
    o, u, v = (np.asarray(a, float) for a in (origin, u_dir, v_dir))
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    n = np.cross(u, v)
    if center:
        o = o - u * total / 2
    letters(k, polys, lambda a, b: (o + u * a + v * b + n * out, n), mat)
    return total


FONT_SANS = "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf"
FONT_ITALIC = "/usr/share/fonts/opentype/inter/InterDisplay-BoldItalic.otf"
FONT_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"


def rounded_rect(x0, x1, y0, y1, r, n=8):
    """CCW outline of a rectangle with rounded corners."""
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return pts


def wall_ring(k, mat, outline, z0, z1, smooth=True):
    """Vertical walls along a closed outline with smooth (vertex-averaged) normals."""
    P2 = np.asarray(outline, float)
    n = len(P2)
    en = []
    for i in range(n):
        d = P2[(i + 1) % n] - P2[i]
        e = np.array([d[1], -d[0]])
        en.append(e / (np.linalg.norm(e) + 1e-12))
    en = np.array(en)
    for i in range(n):
        a, b = P2[i], P2[(i + 1) % n]
        if smooth:
            na = en[i - 1] + en[i]
            nb = en[i] + en[(i + 1) % n]
            if np.dot(en[i - 1], en[i]) < 0.7:
                na = en[i]
            if np.dot(en[i], en[(i + 1) % n]) < 0.7:
                nb = en[i]
        else:
            na = nb = en[i]
        na = na / (np.linalg.norm(na) + 1e-12)
        nb = nb / (np.linalg.norm(nb) + 1e-12)
        P = np.array([[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]])
        N = np.array([[*na, 0], [*nb, 0], [*nb, 0], [*na, 0]])
        k.add(mat, P, N, np.array([[0, 1, 2], [0, 2, 3]]))


def lifeboat(k, c, L=8.5, B=3.2, H=3.0, yaw=0.0, mat="RescueOrange"):
    """Enclosed lifeboat: rounded hull (orange) with a white canopy band and windows."""
    R = np.array([[math.cos(math.radians(yaw)), -math.sin(math.radians(yaw)), 0],
                  [math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0], [0, 0, 1]])
    c = np.asarray(c, float)
    prof = [(-L / 2, 0.0), (-L / 2 + 0.4, -0.6), (L / 2 - 1.2, -0.7), (L / 2, 0.2)]
    for i, (sx, z0, z1, w) in enumerate(((0, 0, H * 0.45, B), (0, H * 0.45, H * 0.9, B * 0.92))):
        out = rounded_rect(-L / 2 + i * 0.3, L / 2 - i * 0.3, -w / 2, w / 2, w * 0.45, 6)
        ring = [tuple((R @ np.array([x, y, 0]))[:2] + c[:2]) for x, y in out]
        wall_ring(k, mat if i == 0 else "ShipWhite" if False else mat, ring, c[2] + z0, c[2] + z1)
        if i == 1:
            plate_from_outline(k, mat, ring, c[2] + z1)
        if i == 0:
            plate_from_outline(k, mat, ring, c[2] + z0, down=True)
    del prof
    # window band
    for sgn in (-1, 1):
        a = c + R @ np.array([-L / 4, sgn * (B * 0.46 + 0.02), H * 0.62])
        b = c + R @ np.array([L / 4, sgn * (B * 0.46 + 0.02), H * 0.62])
        k.tube("BridgeGlass", np.array([a, b]), 0.18, seg=6)
