"""Film-side hoist hatch for T07 (session layer only, the twin stays read-only).

The twin authors the HoistHatch (coaming + two hinged leaves, opening 1.4 x 0.9 m, nacelle_hf) but the canopy shell,
lower deck, rear frame and the hatch's own shaft plate are closed meshes under it, so the hoisted ANYmal used to rise
through solid floor. open_hatch() cuts the opening (nacelle frame rectangle) out of those meshes by clipping their
triangles in plan view, and swings both leaves open (95 deg, the joint limit).
"""

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

HATCH = "/World/T07/Equipment/HoistHatch"
MESH_ROOTS = ["/World/T07/Nacelle/S_B7/Parts/CanopyPanels/CanopyPanels_000",
              "/World/T07/Nacelle/S_B3/Parts/LowerDeck/LowerDeck_000",
              "/World/T07/Nacelle/S_B3/Parts/RearFrame/RearFrame_000",
              f"{HATCH}/Coaming/Visual/SteelPaintDark"]
OPEN = (-8.72, -7.32, -2.33, -1.43)          # opening x0, x1, y0, y1 in the nacelle frame (hatch centre -8.02 / -1.88)
ZBAND = (0.8, 3.45)                          # nacelle-frame heights of the floor stack (shell 1.0 ... deck 3.32)


def _clip(poly, axis, value, keep_less):
    """Sutherland-Hodgman: keep the part of poly (list of (xyz, attrs)) with coord[axis] <= / >= value."""
    out = []
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        da = (a[0][axis] - value) * (1 if keep_less else -1)
        db = (b[0][axis] - value) * (1 if keep_less else -1)
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append((a[0] + t * (b[0] - a[0]), [x + t * (y - x) for x, y in zip(a[1], b[1])]))
    return out


def _outside(poly):
    """poly minus the opening rectangle, as up to four convex pieces."""
    x0, x1, y0, y1 = OPEN
    pieces = [_clip(poly, 0, x0, True), _clip(poly, 0, x1, False)]
    mid = _clip(_clip(poly, 0, x0, False), 0, x1, True)
    pieces += [_clip(mid, 1, y0, True), _clip(mid, 1, y1, False)]
    return [p for p in pieces if len(p) >= 3]


def _cut_mesh(prim, Mi):
    m = UsdGeom.Mesh(prim)
    P = np.array(m.GetPointsAttr().Get(), float)
    fc = np.array(m.GetFaceVertexCountsAttr().Get())
    fi = np.array(m.GetFaceVertexIndicesAttr().Get())
    assert (fc == 3).all(), prim.GetPath()
    W = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    Pn = (Mi @ W @ np.c_[P, np.ones(len(P))].T).T[:, :3]          # nacelle frame
    N = np.array(m.GetNormalsAttr().Get(), float)
    st_pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    ST = np.array(st_pv.Get(), float)
    assert len(N) == len(P) == len(ST), prim.GetPath()             # vertex interpolation throughout
    x0, x1, y0, y1 = OPEN
    tris = fi.reshape(-1, 3)
    lo, hi = Pn[tris].min(1), Pn[tris].max(1)
    hit = ((hi[:, 0] > x0) & (lo[:, 0] < x1) & (hi[:, 1] > y0) & (lo[:, 1] < y1)
           & (hi[:, 2] > ZBAND[0]) & (lo[:, 2] < ZBAND[1]))
    if not hit.any():
        return 0
    newP, newN, newST, newT = list(P), list(N), list(ST), [t for t, h in zip(tris, hit) if not h]
    for t in tris[hit]:
        poly = [(Pn[v], [*P[v], *N[v], *ST[v]]) for v in t]
        for piece in _outside(poly):
            ids = []
            for _, a in piece:
                newP.append(a[0:3])
                newN.append(a[3:6])
                newST.append(a[6:8])
                ids.append(len(newP) - 1)
            newT += [[ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1)]
    newT = np.array(newT, int)
    nn = np.asarray(newN)
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
    m.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in newP]))
    m.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in nn]))
    st_pv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, p)) for p in newST]))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(newT)))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(v) for v in newT.ravel()]))
    m.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, np.min(newP, 0))), Gf.Vec3f(*map(float, np.max(newP, 0)))]))
    return int(hit.sum())


def open_hatch(stage, Mn, FK):
    """Cut the opening and open both leaves; returns {mesh path: triangles cut}. Edit target must be the session."""
    Mi = np.linalg.inv(Mn)
    cut = {}
    for root in MESH_ROOTS:
        r = stage.GetPrimAtPath(root)
        assert r, root
        if r.IsInstance():
            r.SetInstanceable(False)
        for p in Usd.PrimRange(r):
            if p.IsA(UsdGeom.Mesh):
                n = _cut_mesh(p, Mi)
                if n:
                    cut[str(p.GetPath())] = n
    assert cut, "hatch cut found no floor triangles"
    FK.pose(stage, HATCH, {"LeafN_Hinge": math.radians(95.0), "LeafS_Hinge": math.radians(-95.0)})
    return cut
