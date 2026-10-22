"""Realistic, named vessel models for the Cuxhaven film, built to drop into the twin's animated scenes.

Same asset names, origins, axes, scale and articulation split as the twin's assets (CuxhavenSiteTwin read-only):
  roro_vessel, roro_visor     assets/equipment/build_equipment.py: x to the bow, z = 0 waterline, origin midship;
                              hold deck +1.7 m; visor asset origin = its top aft edge (pivot), closed at
                              (x_visor, 0, z_visor_top), open by RV['visor_open']
  wtiv_hull, wtiv_leg,        assets/offshore/build_offshore.py: hull origin at the keel, centre; x to the bow,
  wtiv_crane, wtiv_boom       y to port; legs at WT['legs']; leg origin at the spudcan tip, axis +z; crane origin on
                              the crane-leg axis at the pedestal top; boom origin at the foot pin, along +x, luff about -y
  sov, w2w_gangway            assets/offshore/build_service.py: SOV origin at the keel below midship, x to the bow,
                              waterline z = draft; gangway origin at the slewing pivot on the pedestal top, along +x

Vessels (reference/vessels/vessels.json, photos in reference/vessels/; local reference only):
  Rotra Vente  RoRo nacelle carrier (Amasus/deugro for Siemens Gamesa), IMO 9805568; 141.5 x 21 m, draught 7.1 m,
               air draught 35 m (SOURCED, ship-technology); the model keeps the twin's 141.0 m so the hold slots,
               bow ramp and quay alignment stay exact. Name 'ROTRA VENTE' on both bow sides and the stern.
  Norse Energi DEME jack-up (GustoMSC NG20000X), installed all 21 Windanker turbines 2026 (SOURCED, OE Resources);
               151.3 x 62 m, depth 12 m, 4 x 124 m legs, 3,200 t leg-encircling crane around the AFT PORT leg
               (SOURCED + photos). Name 'NORSE ENERGI' (white italic) forward on both sides and on the stern.
  SOV          generic Ulstein SX175-type service operation vessel - no SOV is published for Windanker, so it is
               unnamed and carries no operator livery (ILLUSTRATIVE).
No company logos are reproduced; ship names are plain text.

Run: ~/Documents/LabAutomationTwin/env_cad/bin/python film/vessels/build_vessels.py [--crane-leg 2|3]
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import Kit  # noqa: E402

CRANE_LEG = 2 if "--crane-leg" not in sys.argv else int(sys.argv[sys.argv.index("--crane-leg") + 1])

# ======================================================================================================== Rotra Vente
# twin frame values (assets/equipment/build_equipment.py RV) kept exactly; B and the form from the sources/photos
RV = dict(L=141.0, B=21.0, T=5.0, deck=1.7, hull_top=6.0, wall_top=20.0, x_visor=54.0, z_visor_top=12.0,
          x_hinge=66.0, hold=(-46.0, 52.0), visor_open=dict(x=56.0, z=27.0, ang=-40.0))
RB2, RBILGE = RV["B"] / 2, 1.6
KEEL_R = -RV["T"]


def rv_xb(z):
    """stem: raked forward with height"""
    return 67.2 + 0.20 * (z - KEEL_R)


def rv_xa(z):
    """transom stern at -L/2 above z = 1; below it the run rises towards the transom"""
    if z >= 1.0:
        return -RV["L"] / 2
    return -RV["L"] / 2 + 15.0 * ((1.0 - z) / 6.0) ** 1.6


def rv_full(x, z):
    w = RB2
    if x > 33.0:
        t = min(1.0, max(0.0, (x - 33.0) / (rv_xb(z) - 33.0)))
        p = 1.5 + 1.6 * min(1.0, max(0.0, (z - KEEL_R) / 17.0))      # fuller (flared) higher up
        w = RB2 * max(0.0, 1.0 - t ** p) ** 0.55
    if x < -48.0 and z < 2.0:
        s = min(1.0, max(0.0, (-48.0 - x) / max(1e-6, -48.0 - rv_xa(z))))
        w *= 1.0 - 0.55 * s ** 1.8 * min(1.0, max(0.0, (2.0 - z) / 7.0))
    return w


def rv_hb(x, z):
    w = rv_full(x, z)
    if z < KEEL_R + RBILGE:
        dz = KEEL_R + RBILGE - z
        w = w - RBILGE + math.sqrt(max(0.0, RBILGE ** 2 - dz ** 2))
    return max(0.0, w)


def rv_split_x(z):
    """the visor's aft boundary: everything above the hold deck forward of x_visor lifts with the visor (the SPMT
    enters the bow turned ~17 deg, sweeping +-8.6 m: no fixed hull sides may stand there - collision check)."""
    if z <= RV["deck"] + 1e-4:
        return rv_xb(z)
    return RV["x_visor"]


Z_LOW = [KEEL_R, KEEL_R + 0.03, KEEL_R + 0.15, KEEL_R + 0.4, KEEL_R + 0.75, KEEL_R + 1.2, KEEL_R + RBILGE,
         -2.5, -1.2, 0.0, 0.45, 1.0, RV["deck"]]
Z_SIDE = [RV["deck"] + 0.002, 2.4, 3.2, 4.0, 4.8, 5.4, RV["hull_top"]]


def _mats_rv(zl):
    return ["Antifouling" if z1 <= 0.45 + 1e-6 else "RotraBlue" for z1 in zl[1:]]


def visor_open_matrix():
    o = RV["visor_open"]
    a = math.radians(o["ang"])
    M = np.eye(4)
    M[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    M[:3, 3] = (o["x"], 0.0, o["z"])
    return M


def visor_closed_matrix():
    M = np.eye(4)
    M[:3, 3] = (RV["x_visor"], 0.0, RV["z_visor_top"])
    return M


def visor_arms(k, Mv, closed):
    """the two orange Z-shaped hydraulic arms between the hull top and the visor sides, for a visor pose Mv."""
    for sgn in (-1, 1):
        yv = rv_hb(58.0, 8.0) + 0.5
        head = (Mv @ np.array([4.0, sgn * yv, -4.0, 1.0]))[:3]
        foot = np.array([46.5, sgn * (RB2 - 0.9), RV["hull_top"]])
        knee = (np.array([50.5, sgn * (RB2 - 0.6), RV["hull_top"] + 1.4]) if closed else
                np.array([47.5, sgn * (RB2 - 0.5), head[2] + 2.5]))
        pts = [foot, foot + (knee - foot) * 0.5, knee, knee + (head - knee) * 0.5 + [0, 0, 0.3 if closed else 0.8], head]
        k.tube("RotraOrange", np.array(pts), 0.75, seg=14, caps=True)
        k.box("RotraOrange", tuple(foot + [0, 0, 0.8]), (2.8, 1.6, 1.6))
        if not closed:
            k.cyl("Galvanized", foot + [0.6, 0, 1.6], knee + [0.6, 0, -2.0], 0.22, seg=8)      # hydraulic cylinder


def rotra_vente(closed=False):
    """closed=False: arms in the open-visor pose (the twin's quay scenes); True: arms folded (under way)."""
    k = Kit()
    L, B = RV["L"], RV["B"]
    x0h, x1h = RV["hold"]
    # ---- fixed hull: keel to the hull top aft, to the visor split forward
    zl = Z_LOW + Z_SIDE

    def x_end(z):
        return rv_split_x(z) if z > RV["deck"] else rv_xb(z)
    grids = {}
    for side in (-1, 1):
        grids[side] = C.loft_side(k, zl, rv_xa, x_end, rv_hb, _mats_rv(zl), nu=110, side=side)
    # transom (flat stern) between the two sides, from the run up to the hull top
    for j in range(len(zl) - 1):
        a, b = grids[-1][j, 0], grids[-1][j + 1, 0]
        c, d = grids[1][j + 1, 0], grids[1][j, 0]
        mat = "Antifouling" if zl[j + 1] <= 0.45 + 1e-6 else "RotraBlue"
        P = np.array([a, b, c, d])
        k.add(mat, P, np.tile([-1.0, 0, 0], (4, 1)), np.array([[0, 1, 2], [0, 2, 3]]))
    # flat bottom
    xs = np.linspace(rv_xa(KEEL_R), rv_xb(KEEL_R), 80)
    out = [(x, -rv_hb(x, KEEL_R)) for x in xs] + [(x, rv_hb(x, KEEL_R)) for x in xs[::-1]]
    out = [p for i, p in enumerate(out) if i == 0 or abs(p[1] - out[i - 1][1]) + abs(p[0] - out[i - 1][0]) > 1e-6]
    C.plate_from_outline(k, "Antifouling", out, KEEL_R, down=True)
    # ---- inner skin of the hold (wing tanks 1.0 m), hold deck, gunwale cap, inner bow sides
    def hb_in(x, z):
        return max(0.15, rv_hb(x, z) - 1.0) if x < 38.0 else max(0.1, rv_hb(x, z) - 0.25)   # thin at the bow (SPMT sweep)
    zin = [RV["deck"], 3.0, 4.5, RV["hull_top"]]
    for side in (-1, 1):
        G = C.loft_side(Kit(), zin, lambda z: x0h, lambda z: RV["x_visor"], hb_in, ["SteelPaintGrey"] * 3, nu=60,
                        side=side)
        C._grid_mesh(k, ["SteelPaintGrey"] * (len(zin) - 1), G, flip=not C.outward_flip(side))   # facing inwards
        # cap between outer shell and inner skin along the top edge (hull top aft, split line forward)
        top_o = [(x, side * rv_hb(x, RV["hull_top"]), RV["hull_top"]) for x in np.linspace(x0h, RV["x_visor"], 40)]
        top_i = [(x, side * hb_in(x, RV["hull_top"]), RV["hull_top"]) for x in np.linspace(x0h, RV["x_visor"], 40)]
        C.strip(k, "RotraBlue", top_o, top_i)
        zz = np.linspace(RV["deck"], RV["hull_top"], 6)
        so = [(rv_split_x(z), side * rv_hb(rv_split_x(z), z), z) for z in zz]
        si = [(rv_split_x(z), side * hb_in(rv_split_x(z), z), z) for z in zz]
        C.strip(k, "RotraBlue", so, si)
    deck_x = np.linspace(x0h, rv_xb(RV["deck"]) - 0.3, 60)
    deck = [(x, -hb_in(x, RV["deck"])) for x in deck_x] + [(x, hb_in(x, RV["deck"])) for x in deck_x[::-1]]
    C.plate_from_outline(k, "DeckPlate", deck, RV["deck"] + 0.02)
    for y in (-3.6, 3.6):                                                      # SPMT lanes (as the twin)
        k.box("LineYellow", ((x0h + RV["x_hinge"]) / 2, y, RV["deck"] + 0.05), (RV["x_hinge"] - x0h - 2, 0.3, 0.02))
    for x in np.arange(x0h + 10, RV["x_hinge"] - 4, 22.0):
        k.box("LineYellow", (x, 0, RV["deck"] + 0.05), (0.3, B - 4.5, 0.02))
    # ---- hull side grid (raised bulwark frames with dark bays, 2 rows) along the hold - the ship's signature
    for side in (-1, 1):
        y = side * (RB2 + 0.005)
        for x in np.arange(-44.0, 30.0, 2.4):
            for z0, z1 in ((2.3, 3.9), (4.1, 5.7)):
                k.quad("RotraBlueDark", (x + 1.2, y, (z0 + z1) / 2), (0, side, 0), (0, 0, 1), 2.0, z1 - z0)
        for z in (2.15, 4.0, 5.85):
            k.box("RotraBlue", (-7.0, side * (RB2 + 0.08), z), (74.0, 0.16, 0.2))
        for x in np.arange(-44.0, 30.01, 2.4):
            k.box("RotraBlue", (x, side * (RB2 + 0.08), 4.0), (0.2, 0.16, 3.9))
    # ---- hold walls: light grey corrugated panels above the hull, seams every 5 m, dark roof coaming
    xw0, xw1 = x0h, 40.0
    for side in (-1, 1):
        yw = side * (RB2 - 0.55)
        k.box("HoldWall", ((xw0 + xw1) / 2, yw, (RV["hull_top"] + RV["wall_top"]) / 2),
              (xw1 - xw0, 0.5, RV["wall_top"] - RV["hull_top"]))
        for x in np.arange(xw0 + 5.0, xw1, 5.0):
            k.box("SteelPaintGrey", (x, yw + side * 0.28, (RV["hull_top"] + RV["wall_top"]) / 2),
                  (0.25, 0.08, RV["wall_top"] - RV["hull_top"]))
        k.box("RoofDark", ((xw0 + xw1) / 2, yw, RV["wall_top"] + 0.5), (xw1 - xw0, 1.2, 1.0))
        k.box("RoofDark", (xw1 + 0.3, yw, (RV["hull_top"] + RV["wall_top"]) / 2), (0.8, 1.2, RV["wall_top"] - RV["hull_top"]))
        C.rail(k, [(xw0, yw, RV["wall_top"] + 1.0), (xw1, yw, RV["wall_top"] + 1.0)], h=1.0, mat="RailingRed", post=3.0)
    # collapsible roof sections stacked at the aft end of the hold (open for loading)
    for i in range(4):
        k.box("RoofDark", (x0h + 5.0, 0, RV["wall_top"] + 1.3 + i * 0.75), (9.2 - i * 0.3, B - 1.6, 0.6))
    for sy in (-1, 1):
        k.box("RoofDark", (x0h + 5.0, sy * (RB2 - 1.2), RV["wall_top"] + 2.2), (9.6, 0.4, 3.2))
    # forward portal (hold gate frame) at the bow opening, as the twin (clears the nacelle on its SPMT)
    for sy in (-1, 1):
        k.box("SteelPaintGrey", (x1h + 0.6, sy * 7.0, (RV["deck"] + 20.0) / 2), (1.2, 3.0, 20.0 - RV["deck"]))
    k.box("SteelPaintGrey", (x1h + 0.6, 0, 20.0 - 1.5), (1.2, 17.0, 3.0))
    # ---- aft body: poop deck, machinery casing, accommodation tower, wheelhouse with wings, mast
    xs_ = np.linspace(-L / 2, x0h, 30)
    poop = [(x, -rv_hb(x, RV["hull_top"])) for x in xs_] + [(x, rv_hb(x, RV["hull_top"])) for x in xs_[::-1]]
    C.plate_from_outline(k, "DeckPlate", poop, RV["hull_top"] + 0.01)
    C.rail(k, [(-L / 2 + 0.3, -RB2 + 0.3, RV["hull_top"]), (-L / 2 + 0.3, RB2 - 0.3, RV["hull_top"])], mat="ShipWhite")
    for sy in (-1, 1):
        C.rail(k, [(-L / 2 + 0.3, sy * (RB2 - 0.3), RV["hull_top"]), (x0h, sy * (RB2 - 0.3), RV["hull_top"])],
               mat="ShipWhite")
    k.box("ShipWhite", (-53.0, 0, RV["hull_top"] + 2.2), (13.0, B - 3.0, 4.4))            # machinery casing
    xa0, xa1, hw = -68.5, -54.0, 8.0
    decks = [RV["hull_top"] + 2.8 * i for i in range(6)]                                    # 7.5 .. 21.5
    for z0, z1 in zip(decks[:-1], decks[1:]):
        k.box("ShipWhite", ((xa0 + xa1) / 2, 0, (z0 + z1) / 2), (xa1 - xa0, 2 * hw, z1 - z0 - 0.02))
        k.box("ShipWhite", ((xa0 + xa1) / 2, 0, z1 - 0.06), (xa1 - xa0 + 0.6, 2 * hw + 0.6, 0.12))   # deck edge
        for sy in (-1, 1):                                                               # cabin windows
            for x in np.arange(xa0 + 1.6, xa1 - 1.0, 2.6):
                k.quad("BridgeGlass", (x, sy * (hw + 0.02), z0 + 1.55), (0, sy, 0), (0, 0, 1), 1.1, 0.9)
        for y in np.arange(-hw + 1.6, hw - 1.0, 2.6):
            k.quad("BridgeGlass", (xa1 + 0.02, y, z0 + 1.55), (1, 0, 0), (0, 0, 1), 1.1, 0.9)
    zb0, zb1 = decks[-1], decks[-1] + 3.2                                                  # wheelhouse 21.5..24.7
    k.box("ShipWhite", (xa1 - 4.5, 0, (zb0 + zb1) / 2), (9.0, B + 1.0, zb1 - zb0))        # full-width with wings
    k.box("RotraOrange", (xa1 - 4.5, 0, zb1 + 0.3), (9.4, B + 1.4, 0.6))                  # orange roof band
    C.window_band(k, (xa1, -B / 2 - 0.4), (xa1, B / 2 + 0.4), zb0 + 0.9, 1.9, (1, 0, 0), pitch=1.4)
    for sy in (-1, 1):
        C.window_band(k, (xa1 - 8.6, sy * (B / 2 + 0.5)), (xa1 - 0.4, sy * (B / 2 + 0.5)), zb0 + 0.9, 1.9, (0, sy, 0),
                      pitch=1.4)
    C.window_band(k, (xa1 - 9.0, -B / 2), (xa1 - 9.0, B / 2), zb0 + 0.9, 1.9, (-1, 0, 0), pitch=1.6)
    # mast with radar scanners, lights; exhaust pipes
    mx = xa1 - 5.0
    k.cyl("ShipWhite", (mx, 0, zb1 + 0.6), (mx, 0, 33.0), 0.28, seg=10)
    k.box("ShipWhite", (mx, 0, 29.0), (0.4, 6.0, 0.3))                                     # yard
    k.box("RailingRed", (mx, 0, 30.3), (0.3, 4.4, 0.25))                                   # radar scanner
    k.box("SteelPaintDark", (mx + 1.2, 0, 27.4), (0.25, 3.6, 0.22))
    k.sphere("ShipWhite", (mx - 1.0, 1.4, 26.2), 0.5)
    k.sphere("ShipWhite", (mx - 1.0, -1.4, 26.2), 0.5)
    for sy in (-0.9, 0.9):
        k.cyl("SteelPaintGrey", (xa0 + 2.0, sy, zb0), (xa0 + 2.0, sy, zb1 + 2.8), 0.35, seg=12)
        k.cyl("SteelPaintDark", (xa0 + 2.0, sy, zb1 + 2.8), (xa0 + 2.0, sy, zb1 + 3.0), 0.38, seg=12)
    # rescue boat on its davit (port side, boat deck), railings on the decks
    C.lifeboat(k, (-62.0, hw + 2.2, decks[1] + 0.4), L=6.0, B=2.4, H=1.6)
    k.box("SteelPaintGrey", (-62.0, hw + 0.8, decks[2] + 0.2), (0.4, 2.6, 0.4))
    for z in decks[1:-1]:
        for sy in (-1, 1):
            C.rail(k, [(xa0 - 0.2, sy * (hw + 0.3), z), (xa1 + 0.2, sy * (hw + 0.3), z)], h=1.0, mat="ShipWhite", mid=False,
                   r=0.03, post=2.0)
    # mooring: bollards and winches on the poop and forecastle
    for sy in (-1, 1):
        for x in (-69.0, -66.0):
            k.cyl("SteelPaintDark", (x, sy * 8.6, RV["hull_top"]), (x, sy * 8.6, RV["hull_top"] + 0.7), 0.25, seg=10)
        k.box("SteelPaintGrey", (-67.5, sy * 4.0, RV["hull_top"] + 0.6), (2.0, 2.6, 1.2))
    # draught marks (white) at the stern and on the fixed forefoot
    for x, side in ((-69.2, -1), (-69.2, 1), (63.0, -1), (63.0, 1)):
        for z in (-1.0, 0.0, 1.0):
            C.hull_text(k, f"{int(z + RV['T'] + 1)}M", rv_hb, side, x, z - 0.15, 0.22, C.FONT_BOLD, out=0.03)
    # ---- visor arms (orange), in the pose of this variant
    visor_arms(k, visor_closed_matrix() if closed else visor_open_matrix(), closed)
    # ---- names: both bow sides on the fixed forefoot (reads correctly from each side), stern with home port
    for side in (-1, 1):
        C.hull_text(k, "ROTRA VENTE", rv_hb, side, 41.0 if side < 0 else 52.6, 2.4, 1.1, C.FONT_SANS, spacing=1.0)
    C.plane_text(k, "ROTRA VENTE", (-L / 2, 0, 5.2), (0, -1, 0), (0, 0, 1), 0.9, C.FONT_SANS, out=0.035)
    C.plane_text(k, "WERKENDAM", (-L / 2, 0, 3.9), (0, -1, 0), (0, 0, 1), 0.55, C.FONT_SANS, out=0.035)
    C.write("roro_vessel_closed" if closed else "roro_vessel", k, dict(
        partId="RoRoVessel", canonicalName="Rotra Vente (RoRo nacelle carrier)", vesselName="ROTRA VENTE",
        imo="9805568 (SOURCED-VERIFY)", length_m=L, beam_m=B, holdDeck_m=RV["deck"],
        source="ship-technology (141.5 x 21 m, draught 7.1 m, air draught 35 m); hlpfi 2024 (14 MW load-out); "
               "reference/vessels/rotra_vente photos (form, livery, name)",
        geometryStatus="B SOURCED; L kept at the twin's 141.0 m (sourced 141.5) for slot/ramp alignment; form, "
                       "heights, details READ-FROM-PHOTO / ILLUSTRATIVE; home port SOURCED-VERIFY",
        visorArms="folded (visor closed)" if closed else "open-visor pose (quay)"))
    if closed:
        return
    # ======================================== visor: forecastle above the split, pivot (asset origin) at its top aft edge
    kv = Kit()
    piv = np.array([RV["x_visor"], 0.0, RV["z_visor_top"]])
    zv = [RV["deck"], 2.4, 3.2, 4.0, 4.8, 5.4, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.6, RV["z_visor_top"]]
    vgrid = {}
    for side in (-1, 1):
        G = C.loft_side(Kit(), zv, lambda z: RV["x_visor"], rv_xb, rv_hb, ["RotraBlue"] * (len(zv) - 1), nu=50, side=side)
        G = G - piv
        C._grid_mesh(kv, ["RotraBlue"] * (len(zv) - 1), G, flip=C.outward_flip(side))
        Gi = G.copy()
        Gi[..., 1] *= np.clip(1 - 0.35 / np.maximum(np.abs(Gi[..., 1]), 0.4), 0.0, 1.0)
        C._grid_mesh(kv, ["SteelPaintGrey"] * (len(zv) - 1), Gi, flip=not C.outward_flip(side))  # inside
        vgrid[side] = G
    C.strip(kv, "SteelPaintGrey", vgrid[-1][:, 0], vgrid[1][:, 0])      # aft face + sloped underside (both sides)
    top = [tuple(p[:2]) for p in vgrid[-1][-1]] + [tuple(p[:2]) for p in vgrid[1][-1][::-1]]
    top = [p for i, p in enumerate(top) if i == 0 or np.hypot(p[0] - top[i - 1][0], p[1] - top[i - 1][1]) > 1e-6]
    C.plate_from_outline(kv, "DeckPlate", top, 0.0)
    rail_pts = [(x - piv[0], s * (rv_hb(x, RV["z_visor_top"]) - 0.4), 0.0) for s in (-1,) for x in
                np.linspace(RV["x_visor"] + 0.5, rv_xb(RV["z_visor_top"]) - 1.5, 14)]
    rail_pts += [(x - piv[0], (rv_hb(x, RV["z_visor_top"]) - 0.4), 0.0) for x in
                 np.linspace(rv_xb(RV["z_visor_top"]) - 1.5, RV["x_visor"] + 0.5, 14)]
    C.rail(kv, rail_pts, h=1.1, mat="RailingRed", post=2.0)
    for sy in (-1, 1):                                                                       # bollards, windlass
        kv.cyl("SteelPaintDark", (4.0, sy * 6.0, 0.0), (4.0, sy * 6.0, 0.7), 0.28, seg=10)
        kv.box("SteelPaintGrey", (8.0, sy * 3.2, 0.6), (2.4, 2.2, 1.2))
    kv.cyl("ShipWhite", (11.5, 0, 0.0), (11.5, 0, 6.0), 0.18, seg=8)                         # foremast
    kv.sphere("ShipWhite", (11.5, 0, 6.1), 0.25)
    for side in (-1, 1):                                                                     # hawse pockets
        x = 64.0
        y = side * (rv_hb(x, 9.5) + 0.04)
        kv.quad("Antifouling", (x - piv[0], y, 9.5 - piv[2]), (0, side, 0), (0, 0, 1), 1.6, 1.0)
    C.write("roro_visor", kv, dict(partId="RoRoBowVisor", canonicalName="Rotra Vente bow visor",
                                   pivot="top aft edge, rotate about y (negative = open)",
                                   geometryStatus="movable bow SOURCED; form READ-FROM-PHOTO"))


# ======================================================================================================= Norse Energi
# legs, crane geometry and frames from the twin's WT (scenes place the parts with it); L, B, leg length SOURCED (DEME)
WT = dict(L=151.3, B=62.0, D=12.0, draft=6.5, legs=[(44.0, 21.5), (44.0, -21.5), (-50.0, 21.5), (-50.0, -21.5)],
          leg_len=124.0, leg_w=12.0, crane_leg=CRANE_LEG, pedestal_h=26.0, boom_len=150.0, boom_foot=(14.0, 0.0, 9.0),
          crane_t=3200.0, rest_slew=(-8.0 if CRANE_LEG == 2 else 8.0), rest_luff=0.0)
AF_Z = 49.0                       # A-frame head (luffing sheaves) above the pedestal top, crane frame
LUFF_ATT = (150.0 - 1.0, 2.6, 2.4)  # luffing-rope attachment at the boom head (boom frame, +-y)


def _truss(k, a, b, w, n_bays, chord_mat, brace_mat, chord_r=0.5, brace_r=0.2, flat_z=False):
    """square lattice member from a to b: 4 chords on a w x w square, X bracing per bay, horizontals."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = (b - a) / np.linalg.norm(b - a)
    up = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(ax, up)
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    cs = [c * w / 2 for c in ((u + v), (u - v), (-u - v), (-u + v))]
    for c in cs:
        k.cyl(chord_mat, a + c, b + c, chord_r, seg=10, caps=False)
    for i in range(n_bays):
        p0 = a + (b - a) * i / n_bays
        p1 = a + (b - a) * (i + 1) / n_bays
        for j in range(4):
            c0, c1 = cs[j], cs[(j + 1) % 4]
            k.cyl(brace_mat, p0 + c0, p1 + c1, brace_r, seg=6, caps=False)
            k.cyl(brace_mat, p0 + c1, p1 + c0, brace_r, seg=6, caps=False)
            k.cyl(brace_mat, p1 + c0, p1 + c1, brace_r * 1.1, seg=6, caps=False)


def _ne_outline():
    L, B = WT["L"], WT["B"]
    pts = []
    rb, rs = 16.0, 3.0
    for cx, cy, a0, r in ((L / 2 - rb, -B / 2 + rb, -90, rb), (L / 2 - rb, B / 2 - rb, 0, rb),
                          (-L / 2 + rs, B / 2 - rs, 90, rs), (-L / 2 + rs, -B / 2 + rs, 180, rs)):
        n = 14 if r == rb else 5
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return pts


def norse_energi():
    L, B, D = WT["L"], WT["B"], WT["D"]
    k = Kit()
    plan = _ne_outline()
    wl = WT["draft"] + 0.8
    C.wall_ring(k, "Antifouling", plan, 0.0, wl)
    C.wall_ring(k, "DemeGreen", plan, wl, D)
    k.tube("DemeGreen", np.array([(x, y, D + 0.25) for x, y in plan + plan[:1]]), 0.25, seg=8)   # deck edge
    ww = WT["leg_w"] / 2 + 1.0
    wells = [[(x - ww, y - ww), (x + ww, y - ww), (x + ww, y + ww), (x - ww, y + ww)] for x, y in WT["legs"]]
    C.plate_from_outline(k, "Antifouling", plan, 0.0, down=True, holes=[w[::-1] for w in wells])
    C.plate_from_outline(k, "DeckPlate", plan, D, holes=[w[::-1] for w in wells])
    for w_ in wells:
        C.wall_ring(k, "SteelPaintGrey", w_[::-1], 0.0, D, smooth=False)                    # well walls (inward)
    # aft working deck: green paint with round sea-fastening grillages (photos ne_3, ne_5)
    aft = [(-L / 2 + 0.6, -B / 2 + 0.6), (-36.0, -B / 2 + 0.6), (-36.0, B / 2 - 0.6), (-L / 2 + 0.6, B / 2 - 0.6)]
    hole = [w for (x, y), w in zip(WT["legs"], wells) if x < -40]
    C.plate_from_outline(k, "DeckGreen", aft, D + 0.03, holes=[w[::-1] for w in hole])
    for cx_, cy_ in ((-64.0, 0.0), (-42.0, 0.0)):
        a = np.linspace(0, 2 * math.pi, 49)
        for r in (4.5, 3.2):
            k.tube("SafetyYellow", np.c_[cx_ + r * np.cos(a), cy_ + r * np.sin(a), np.full(49, D + 0.35)], 0.22, seg=6)
    for y in np.arange(-B / 2 + 8, B / 2 - 7, 8.0):                                       # cargo deck grillage lines
        k.box("LineYellow", (2.0, y, D + 0.06), (66.0, 0.25, 0.04))
    # side rails along the deck edge (gaps at the forward legs / accommodation)
    for sy in (-1, 1):
        C.rail(k, [(-L / 2 + 2, sy * (B / 2 - 0.6), D), (28.0, sy * (B / 2 - 0.6), D)], mat="SafetyYellow", post=2.5)
    # hull marks: tug circles, draught marks
    for sy in (-1, 1):
        for x in np.arange(-60.0, 50.0, 22.0):
            a = np.linspace(0, 2 * math.pi, 25)
            k.tube("LetterWhite", np.c_[x + 0.5 * np.cos(a), np.full(25, sy * (B / 2 + 0.05)), 10.0 + 0.5 * np.sin(a)],
                   0.06, seg=4)
    # jacking houses (white, open frame on top) around the three non-crane legs
    for i, (x, y) in enumerate(WT["legs"]):
        if i == WT["crane_leg"]:
            continue
        w = WT["leg_w"] + 3.0
        for sx, sy, lx, ly in ((w / 2, 0, 1.2, w), (-w / 2, 0, 1.2, w), (0, w / 2, w, 1.2), (0, -w / 2, w, 1.2)):
            k.box("ShipWhite", (x + sx, y + sy, D + 6.5), (lx, ly, 13.0))
            k.box("SteelPaintDark", (x + sx * 1.01, y + sy * 1.01, D + 10.5), (max(lx * 0.4, 0.05) if lx < 2 else lx * 0.6,
                                                                              max(ly * 0.4, 0.05) if ly < 2 else ly * 0.6, 0.9))
        for ex in (-1, 1):
            for ey in (-1, 1):
                k.box("ShipWhite", (x + ex * w / 2, y + ey * w / 2, D + 10.0), (1.6, 1.6, 20.0))
        for z in (D + 15.5, D + 19.5):
            for sx, sy, lx, ly in ((w / 2, 0, 1.0, w), (-w / 2, 0, 1.0, w), (0, w / 2, w, 1.0), (0, -w / 2, w, 1.0)):
                k.box("ShipWhite", (x + sx, y + sy, z), (lx, ly, 0.9))
        for ex in (-1, 1):                                                                   # diagonal braces
            k.cyl("ShipWhite", (x + ex * w / 2, y - w / 2, D + 13.0), (x + ex * w / 2, y + w / 2, D + 19.5), 0.25, seg=6)
        sq = [(x - w / 2 - 0.8, y - w / 2 - 0.8), (x + w / 2 + 0.8, y - w / 2 - 0.8), (x + w / 2 + 0.8, y + w / 2 + 0.8),
              (x - w / 2 - 0.8, y + w / 2 + 0.8), (x - w / 2 - 0.8, y - w / 2 - 0.8)]
        C.rail(k, [(px, py, D + 20.0) for px, py in sq], mat="SafetyYellow", post=2.5)
        k.box("DemeGreen", (x, y - w / 2 - 0.3, D + 1.2), (2.0, 0.2, 2.2))                   # door
    # ---- accommodation forward (between the forward legs), bridge, lifeboats, helideck
    bow0 = L / 2 - 16.0
    xa0, xa1, hw = 38.0, L / 2 - 5.5, 13.0                 # aft face clear of the blade-rack roots (x 37.5)
    decks = [D + 3.1 * i for i in range(8)]                                                  # 7 decks
    for z0, z1 in zip(decks[:-1], decks[1:]):
        k.box("ShipWhite", ((xa0 + xa1) / 2, 0, (z0 + z1) / 2), (xa1 - xa0, 2 * hw, z1 - z0 - 0.02))
        k.box("ShipWhite", ((xa0 + xa1) / 2, 0, z1 - 0.08), (xa1 - xa0 + 0.8, 2 * hw + 0.8, 0.16))
        for sy in (-1, 1):
            C.window_band(k, (xa0 + 2, sy * hw), (xa1 - 1.5, sy * hw), z0 + 1.0, 1.1, (0, sy, 0), pitch=2.2)
        C.window_band(k, (xa1, -hw + 1.5), (xa1, hw - 1.5), z0 + 1.0, 1.1, (1, 0, 0), pitch=2.2)
    zb0, zb1 = decks[-1], decks[-1] + 3.6
    k.box("ShipWhite", (xa1 - 6.0, 0, (zb0 + zb1) / 2), (12.0, 2 * hw + 12.0, zb1 - zb0))      # wheelhouse + wings
    C.window_band(k, (xa1, -hw - 6.0), (xa1, hw + 6.0), zb0 + 1.0, 2.2, (1, 0, 0), pitch=1.5)
    for sy in (-1, 1):
        C.window_band(k, (xa1 - 11.5, sy * (hw + 6.0)), (xa1 - 0.5, sy * (hw + 6.0)), zb0 + 1.0, 2.2, (0, sy, 0), pitch=1.5)
    k.box("ShipWhite", (xa1 - 6.0, 0, zb1 + 0.2), (12.6, 2 * hw + 12.6, 0.4))
    mx = xa1 - 8.0
    # mast: kept below the resting boom (boom bottom chord ~D+33.6 over the bridge)
    k.cyl("ShipWhite", (mx, -6.0, zb1), (mx, -6.0, zb1 + 5.0), 0.35, seg=10)
    k.box("ShipWhite", (mx, -6.0, zb1 + 3.8), (0.5, 6.0, 0.35))
    k.box("SteelPaintDark", (mx, -6.0, zb1 + 4.8), (0.35, 4.6, 0.3))                         # radar
    k.sphere("ShipWhite", (mx - 1.5, -9.0, zb1 + 2.5), 0.7)
    k.sphere("ShipWhite", (mx - 1.5, -3.0, zb1 + 2.5), 0.7)
    for z in decks[1:-1]:
        for sy in (-1, 1):
            C.rail(k, [(xa0 - 0.3, sy * (hw + 0.4), z), (xa1 + 0.3, sy * (hw + 0.4), z)], h=1.0, mat="ShipWhite",
                   mid=False, r=0.03, post=2.2)
    for x in (57.5, 66.0):                                         # lifeboats, starboard (forward of leg 1's lattice)
        C.lifeboat(k, (x, -hw - 2.4, decks[2] + 0.2), L=8.5, B=3.2, H=2.8)
        for dx in (-3.0, 3.0):
            k.box("ShipWhite", (x + dx, -hw - 1.2, decks[3] - 0.2), (0.5, 3.0, 0.5))
    C.lifeboat(k, (58.0, hw + 2.2, decks[2] + 0.4), L=6.0, B=2.4, H=1.6)                   # rescue boat, port
    # helideck: octagon cantilevered over the bow, forward starboard (photo ne_3)
    hx, hy, hz = L / 2 + 2.0, -13.0, zb1 + 2.5
    oc = [(hx + 12.5 * math.cos(math.radians(22.5 + 45 * i)), hy + 12.5 * math.sin(math.radians(22.5 + 45 * i)))
          for i in range(8)]
    C.plate_from_outline(k, "HeliDeck", oc, hz + 0.4)
    C.plate_from_outline(k, "SteelPaintGrey", oc, hz, down=True)
    C.wall_ring(k, "SteelPaintGrey", oc, hz, hz + 0.4, smooth=False)
    a = np.linspace(0, 2 * math.pi, 65)
    k.tube("SafetyYellow", np.c_[hx + 6.0 * np.cos(a), hy + 6.0 * np.sin(a), np.full(65, hz + 0.44)], 0.35, seg=4)
    k.tube("LetterWhite", np.array([(x, y, hz + 0.43) for x, y in oc + oc[:1]]) * [0.97, 0.97, 1] +
           [hx * 0.03, hy * 0.03, 0], 0.15, seg=4)
    C.plane_text(k, "H", (hx, hy - 1.8, hz + 0.45), (0, 1, 0), (-1, 0, 0), 3.6, C.FONT_BOLD, out=0.0)
    for (ex, ey) in ((xa1 - 1.0, -hw + 1.0), (xa1 - 1.0, -2.0), (xa1 - 6.0, -hw + 1.0)):
        k.cyl("ShipWhite", (ex, ey, zb1), (hx - 4.0, hy + (ey - hy) * 0.4, hz), 0.45, seg=8)
    for (ex, ey) in ((xa1, -hw), (xa1, -1.0)):
        k.cyl("ShipWhite", (ex, ey, decks[3]), (hx + 6.0, hy + (ey - hy) * 0.3, hz), 0.4, seg=8)
    # ---- crane pedestal (white) around the crane leg, slew-bearing seat, door, platform ring
    cx, cy = WT["legs"][WT["crane_leg"]]
    k.cyl("ShipWhite", (cx, cy, D), (cx, cy, D + WT["pedestal_h"] - 0.8), 11.5, seg=56, caps=False)
    rin = WT["leg_w"] / 2 * math.sqrt(2) + 1.0
    k.lathe("ShipWhite", [(rin, D + WT["pedestal_h"] - 0.8), (13.0, D + WT["pedestal_h"] - 0.8),
                          (13.0, D + WT["pedestal_h"] - 0.02), (rin, D + WT["pedestal_h"] - 0.02),
                          (rin, D + WT["pedestal_h"] - 0.8)], base=(cx, cy, 0), seg=56, smooth=False)
    # platform ring high on the pedestal, above the 3-high blade rack (top blade ~D+21) beside the port aft leg
    k.lathe("ShipWhite", [(11.5, D + 22.4), (13.0, D + 22.4), (13.0, D + 22.8), (11.5, D + 22.8), (11.5, D + 22.4)],
            base=(cx, cy, 0), seg=56, smooth=False)
    a = np.linspace(0, 2 * math.pi, 57)
    k.tube("SafetyYellow", np.c_[cx + 12.9 * np.cos(a), cy + 12.9 * np.sin(a), np.full(57, D + 23.9)], 0.04, seg=4)
    door_dir = np.array([1.0, 0.0])
    k.box("SteelPaintDark", (cx + 11.45 * door_dir[0], cy, D + 1.4), (0.3, 1.6, 2.4))
    for z in (D + 4.0, D + 8.0, D + 18.0, D + 22.0):                                         # small windows/vents
        k.box("SteelPaintDark", (cx + 11.47, cy, z), (0.2, 1.0, 0.6))
    # boom rest frame on the accommodation roof (the boom lies forward, over the accommodation, in transit)
    fx, _, fz = WT["boom_foot"]
    th = math.radians(WT["rest_slew"])
    foot = np.array([cx + fx * math.cos(th), cy + fx * math.sin(th)])
    xr = xa0 + 3.0
    t = (xr - foot[0]) / math.cos(th)
    yr = foot[1] + t * math.sin(th)
    zr = D + WT["pedestal_h"] + fz - (9.0 - 5.0 * t / WT["boom_len"]) / 4 - 0.4
    k.box("ShipWhite", (xr, yr, zr - 0.6), (2.0, 11.0, 1.2))
    for sy in (-4.5, 4.5):
        k.cyl("ShipWhite", (xr, yr + sy, decks[-1]), (xr, yr + sy, zr - 1.2), 0.5, seg=8)
    # ---- stern: two cantilevered grating platforms (port / starboard), green frames, struts to the transom
    for sy in (-1, 1):
        y0, y1 = sy * 6.0, sy * 28.0
        xo, xi = -L / 2 - 14.0, -L / 2 + 1.0
        pl = [(xo, min(y0, y1)), (xi, min(y0, y1)), (xi, max(y0, y1)), (xo, max(y0, y1))]
        C.plate_from_outline(k, "GratingYellow", pl, D + 0.25)
        C.plate_from_outline(k, "GratingYellow", pl, D + 0.2, down=True)
        C.wall_ring(k, "DemeGreen", pl, D - 0.6, D + 0.25, smooth=False)
        for yy in np.linspace(min(y0, y1) + 1, max(y0, y1) - 1, 4):
            k.cyl("DemeGreen", (-L / 2, yy, D - 7.0), (xo + 2.0, yy, D - 0.6), 0.45, seg=8)
        C.rail(k, [(xi, y0, D + 0.25), (xo, y0, D + 0.25), (xo, y1, D + 0.25), (xi, y1, D + 0.25)], mat="SafetyYellow",
               post=2.5)
    # auxiliary crane (white lattice) near the forward port leg
    ax_, ay_ = 26.0, 26.0
    k.cyl("ShipWhite", (ax_, ay_, D), (ax_, ay_, D + 6.0), 1.4, seg=20)
    k.box("ShipWhite", (ax_ - 1.0, ay_, D + 7.2), (4.0, 3.0, 2.4))
    _truss(k, (ax_, ay_, D + 8.0), (ax_ - 26.0, ay_ - 3.0, D + 16.0), 1.6, 8, "ShipWhite", "ShipWhite", 0.15, 0.07)
    # ---- names: forward hull both sides (white italic), stern
    hb_side = lambda x, z: B / 2 if x < bow0 else math.sqrt(max(0.0, 16.0 ** 2 - (x - bow0) ** 2)) + B / 2 - 16.0  # noqa: E731
    for side in (-1, 1):
        C.hull_text(k, "NORSE ENERGI", hb_side, side, 41.0 if side < 0 else 56.5, 8.6, 1.5, C.FONT_ITALIC,
                    italic=0.18, out=0.04)
    C.plane_text(k, "NORSE ENERGI", (-L / 2, 0, 9.0), (0, -1, 0), (0, 0, 1), 1.3, C.FONT_ITALIC, italic=0.18, out=0.04)
    C.write("wtiv_hull", k, dict(
        partId="WTIVHull", canonicalName="Norse Energi (DEME jack-up WTIV, GustoMSC NG20000X)", vesselName="NORSE ENERGI",
        imo="1035296 (SOURCED-VERIFY)", length_m=L, beam_m=B, hullDepth_m=D, transitDraft_m=WT["draft"],
        legs=[f"{x},{y}" for x, y in WT["legs"]], craneLeg=WT["crane_leg"], craneCapacity_t=WT["crane_t"],
        source="DEME (151.3 x 62 m, depth 12 m, 4 x 124 m legs, 3,200 t crane); OE Resources 2026 (Windanker); "
               "reference/vessels/norse_energi photos (layout, livery, name)",
        geometryStatus="L, B, D, legs, crane capacity SOURCED; crane around the aft port leg READ-FROM-PHOTO; "
                       "leg positions, heights, details ILLUSTRATIVE"))
    # ---- leg: square lattice, black, red/white top bands, spudcan
    kl = Kit()
    w = WT["leg_w"]
    top = WT["leg_len"]
    _truss(kl, (0, 0, 5.0), (0, 0, top - 9.0), w, 14, "LegBlack", "LegBlack", chord_r=0.6, brace_r=0.24)
    for i, mat in enumerate(("BoomRed", "ShipWhite", "BoomRed")):
        _truss(kl, (0, 0, top - 9.0 + 3.0 * i), (0, 0, top - 6.0 + 3.0 * i), w, 1, mat, mat, chord_r=0.62, brace_r=0.25)
    kl.box("BoomRed", (0, 0, top + 0.2), (w + 1.2, w + 1.2, 0.4))
    for ex in (-1, 1):                                                                        # rack teeth (chords)
        for ey in (-1, 1):
            kl.cyl("LegBlack", (ex * w / 2, ey * w / 2, 5.0), (ex * w / 2, ey * w / 2, top - 9.0), 0.75, seg=8, caps=False)
    kl.lathe("LegBlack", [(0.0, 0.0), (7.0, 0.8), (7.0, 2.0), (3.5, 5.0), (0.0, 5.0)], seg=24, smooth=False)   # spudcan
    C.write("wtiv_leg", kl, dict(partId="WTIVLeg", canonicalName="Norse Energi leg", length_m=top,
                                 geometryStatus="length SOURCED (124 m); lattice and colours READ-FROM-PHOTO"))
    # ---- slewing crane house: origin on the leg axis at the pedestal top (as the twin)
    kc = Kit()
    rin = WT["leg_w"] / 2 * math.sqrt(2) + 1.0
    kc.lathe("ShipWhite", [(rin, 0.0), (13.5, 0.0), (13.5, 2.0), (rin, 2.0), (rin, 0.0)], seg=56, smooth=False)
    kc.box("ShipWhite", (-17.5, 0, 7.0), (14.0, 20.0, 10.0))                                 # machinery house
    for y in np.arange(-8.0, 8.1, 2.0):
        kc.box("SteelPaintGrey", (-24.55, y, 7.0), (0.1, 1.2, 7.0))                          # louvres
    kc.box("ShipWhite", (12.5, 10.0, 5.0), (7.0, 5.0, 5.0))                                   # operator cab
    C.window_band(kc, (16.0, 7.6), (16.0, 12.4), 4.0, 2.4, (1, 0, 0), pitch=1.2)
    # tall white A-frame (gantry) behind the leg; its head is 22 m wide so the two luffing-rope groups run past the
    # leg on both sides (every part stays outside the leg's swept radius: corner chords 9.25 m)
    for sy in (-1, 1):
        kc.cyl("ShipWhite", (-12.0, sy * 10.0, 2.0), (-13.0, sy * 10.5, AF_Z - 1.5), 0.9, seg=10)   # front legs
        kc.cyl("ShipWhite", (-25.0, sy * 9.0, 2.0), (-13.0, sy * 10.5, AF_Z - 1.5), 0.9, seg=10)   # back stays
        kc.cyl("ShipWhite", (-12.4, sy * 10.2, 24.0), (-24.4, sy * 9.4, 13.0), 0.45, seg=8)       # brace
        kc.cyl("SteelPaintDark", (-13.0, sy * 9.4, AF_Z), (-13.0, sy * 11.6, AF_Z), 1.1, seg=16)  # sheave blocks
    kc.box("ShipWhite", (-13.0, 0, AF_Z - 0.4), (3.4, 23.0, 2.4))                             # A-frame head beam
    # boom heel: two brackets in front of the leg carry the boom-foot pins at (14, +-3.6, 9) - the boom is hinged
    # on the crane house, never free in the air
    for sy in (-1, 1):
        kc.box("ShipWhite", (13.2, sy * 3.6, 5.6), (7.4, 1.6, 7.0))
        kc.cyl("SteelPaintDark", (14.0, sy * 2.6, 9.0), (14.0, sy * 4.6, 9.0), 0.9, seg=14)       # foot pins
    kc.box("ShipWhite", (12.0, 0, 2.6), (5.0, 9.0, 1.2))
    C.write("wtiv_crane", kc, dict(partId="WTIVCrane", canonicalName="Norse Energi 3,200 t leg-encircling crane",
                                   capacity_t=WT["crane_t"], type="leg-encircling",
                                   geometryStatus="capacity SOURCED; geometry READ-FROM-PHOTO / ILLUSTRATIVE"))
    # ---- boom: tapered lattice, red chords and white lacing, red head with sheaves (origin = foot pin)
    kb = Kit()
    Lb = WT["boom_len"]
    nb = 14
    for i in range(nb):
        x0, x1 = Lb * i / nb, Lb * (i + 1) / nb
        w0, w1 = 9.0 - 5.0 * i / nb, 9.0 - 5.0 * (i + 1) / nb
        for sy in (-1, 1):
            for sz in (-1, 1):
                kb.cyl("BoomRed", (x0, sy * w0 / 2, sz * w0 / 4), (x1, sy * w1 / 2, sz * w1 / 4), 0.42, seg=8, caps=False)
            kb.cyl("ShipWhite", (x0, sy * w0 / 2, -w0 / 4), (x1, sy * w1 / 2, w1 / 4), 0.17, seg=6, caps=False)
            kb.cyl("ShipWhite", (x0, sy * w0 / 2, w0 / 4), (x1, sy * w1 / 2, -w1 / 4), 0.17, seg=6, caps=False)
        for sz in (-1, 1):
            kb.cyl("ShipWhite", (x0, -w0 / 2, sz * w0 / 4), (x1, w1 / 2, sz * w1 / 4), 0.15, seg=6, caps=False)
        kb.cyl("ShipWhite", (x1, -w1 / 2, 0), (x1, w1 / 2, 0), 0.2, seg=6, caps=False)
    kb.box("BoomRed", (Lb + 2.0, 0, -1.0), (5.0, 4.6, 4.2))                                   # boom head
    kb.cyl("SteelPaintDark", (Lb + 3.0, -2.0, -2.2), (Lb + 3.0, 2.0, -2.2), 1.7, seg=16)
    kb.box("BoomRed", (Lb - 6.0, 0, 2.6), (10.0, 2.2, 1.6))                                   # auxiliary jib head
    for sy in (-1, 1):                                                                         # luffing-rope sheaves
        kb.cyl("SteelPaintDark", (LUFF_ATT[0], sy * (LUFF_ATT[1] - 0.5), LUFF_ATT[2]),
               (LUFF_ATT[0], sy * (LUFF_ATT[1] + 0.5), LUFF_ATT[2]), 0.9, seg=14)
    C.write("wtiv_boom", kb, dict(partId="WTIVBoom", canonicalName="Norse Energi main boom", length_m=Lb,
                                  geometryStatus="colours READ-FROM-PHOTO; length ILLUSTRATIVE (twin)"))


# =========================================================================================================== the SOV
SOV = dict(L=85.0, B=19.5, D=8.5, draft=6.0, gangway=(4.0, -6.5), pedestal_h=12.0)      # twin values (frame)
SB2 = SOV["B"] / 2


def sov_xb(z):
    return 40.6 + 2.0 * math.sin(math.pi * min(1.0, max(0.0, z / 16.0)))                     # X-bow-like bulge


def sov_xa(z):
    return -SOV["L"] / 2 if z >= 5.0 else -SOV["L"] / 2 + 9.0 * ((5.0 - z) / 5.0) ** 1.5


def sov_hb(x, z):
    w = SB2
    if x > 16.0:
        t = min(1.0, max(0.0, (x - 16.0) / (sov_xb(z) - 16.0)))
        p = 1.4 + 1.3 * min(1.0, max(0.0, z / 15.0))
        w = SB2 * max(0.0, 1.0 - t ** p) ** 0.6
    if x < -32.0 and z < 5.0:
        s = min(1.0, max(0.0, (-32.0 - x) / max(1e-6, -32.0 - sov_xa(z))))
        w *= 1.0 - 0.5 * s ** 1.8 * (5.0 - z) / 5.0
    r = 1.4
    if z < r:
        w = w - r + math.sqrt(max(0.0, r ** 2 - (r - z) ** 2))
    return max(0.0, w)


def sov():
    k = Kit()
    L, D = SOV["L"], SOV["D"]
    boot = SOV["draft"] + 0.4
    za = [0.0, 0.03, 0.15, 0.4, 0.75, 1.4, 3.0, 4.6, SOV["draft"], boot, 7.3, D]
    mats = ["SovBoot" if z1 <= boot + 1e-6 else "SovBlue" for z1 in za[1:]]
    G = {s: C.loft_side(k, za, sov_xa, sov_xb, sov_hb, mats, nu=90, side=s) for s in (-1, 1)}
    for j in range(len(za) - 1):                                                             # transom
        P = np.array([G[-1][j, 0], G[-1][j + 1, 0], G[1][j + 1, 0], G[1][j, 0]])
        k.add(mats[j], P, np.tile([-1.0, 0, 0], (4, 1)), np.array([[0, 1, 2], [0, 2, 3]]))
    xs = np.linspace(sov_xa(0.0), sov_xb(0.0), 60)
    bot = [(x, -sov_hb(x, 0.0)) for x in xs] + [(x, sov_hb(x, 0.0)) for x in xs[::-1]]
    C.plate_from_outline(k, "SovBoot", bot, 0.0, down=True)
    # forecastle: the blue hull rises two decks above the main deck forward (photos sov_1..4)
    x_fc = 6.0
    zf = [D, 10.0, 11.5, 13.0, 14.6]
    Gf = {s: C.loft_side(k, zf, lambda z: x_fc, sov_xb, sov_hb, ["SovBlue"] * 4, nu=50, side=s) for s in (-1, 1)}
    C.strip(k, "ShipWhite", Gf[-1][:, 0], Gf[1][:, 0])                                     # aft face of the forecastle
    topf = [tuple(p[:2]) for p in Gf[-1][-1]] + [tuple(p[:2]) for p in Gf[1][-1][::-1]]
    C.plate_from_outline(k, "DeckPlate", topf, 14.6)
    # main deck aft of the forecastle, bulwark, rails
    xs = np.linspace(sov_xa(D), x_fc + 0.2, 30)
    mdeck = [(x, -sov_hb(x, D)) for x in xs] + [(x, sov_hb(x, D)) for x in xs[::-1]]
    C.plate_from_outline(k, "DeckPlate", mdeck, D)
    for side in (-1, 1):
        pts = [(x, side * (sov_hb(x, D) - 0.15), D) for x in np.linspace(sov_xa(D) + 0.3, x_fc, 20)]
        C.rail(k, pts, h=1.1, mat="SafetyYellow", post=1.8)
    # superstructure (white) on the forecastle: two decks + bridge wrapping forward, helideck above
    decks = [14.6, 17.4, 20.2]
    xs0, xs1 = 8.0, 34.0
    for i, (z0, z1) in enumerate(zip(decks[:-1], decks[1:])):
        x0_, x1_ = xs0 + 1.0 * i, xs1 - 1.5 * i
        hw = sov_hb(x1_, 14.0) - 0.6 - 0.5 * i
        k.box("ShipWhite", ((x0_ + x1_) / 2, 0, (z0 + z1) / 2), (x1_ - x0_, 2 * hw, z1 - z0 - 0.02))
        for sy in (-1, 1):
            C.window_band(k, (x0_ + 1.5, sy * hw), (x1_ - 1.0, sy * hw), z0 + 0.9, 1.0, (0, sy, 0), pitch=2.0)
        C.window_band(k, (x1_, -hw + 0.8), (x1_, hw - 0.8), z0 + 0.9, 1.0, (1, 0, 0), pitch=2.0)
    zb0, zb1 = decks[-1], decks[-1] + 3.0
    bx0, bx1 = 18.0, 31.0
    k.box("ShipWhite", ((bx0 + bx1) / 2, 0, (zb0 + zb1) / 2), (bx1 - bx0, SOV["B"] + 1.0, zb1 - zb0))
    C.window_band(k, (bx1, -SB2 - 0.4), (bx1, SB2 + 0.4), zb0 + 0.8, 1.9, (1, 0, 0), pitch=1.3)
    for sy in (-1, 1):
        C.window_band(k, (bx0 + 0.5, sy * (SB2 + 0.5)), (bx1 - 0.3, sy * (SB2 + 0.5)), zb0 + 0.8, 1.9, (0, sy, 0), pitch=1.3)
    C.window_band(k, (bx0, -SB2), (bx0, SB2), zb0 + 0.8, 1.9, (-1, 0, 0), pitch=1.6)
    # helideck over the forward superstructure (photo sov_4), on white legs
    hx, hz = 31.0, zb1 + 2.4
    oc = [(hx + 10.0 * math.cos(math.radians(22.5 + 45 * i)), 10.0 * math.sin(math.radians(22.5 + 45 * i))) for i in range(8)]
    C.plate_from_outline(k, "HeliDeck", oc, hz + 0.35)
    C.plate_from_outline(k, "SteelPaintGrey", oc, hz, down=True)
    C.wall_ring(k, "ShipWhite", oc, hz, hz + 0.35, smooth=False)
    a = np.linspace(0, 2 * math.pi, 65)
    k.tube("SafetyYellow", np.c_[hx + 5.0 * np.cos(a), 5.0 * np.sin(a), np.full(65, hz + 0.39)], 0.3, seg=4)
    C.plane_text(k, "H", (hx, -1.4, hz + 0.4), (0, 1, 0), (-1, 0, 0), 2.8, C.FONT_BOLD, out=0.0)
    for (ex, ey) in ((bx1 - 1.0, -6.0), (bx1 - 1.0, 6.0), (bx0 + 6.0, -6.0), (bx0 + 6.0, 6.0)):
        k.cyl("ShipWhite", (ex, ey, zb1), (ex + (hx - ex) * 0.3, ey * 0.8, hz), 0.3, seg=8)
    # mast with radars and domes (aft end of the bridge)
    mx = bx0 + 1.5
    k.cyl("ShipWhite", (mx, 0, zb1), (mx, 0, zb1 + 9.0), 0.3, seg=10)
    k.box("ShipWhite", (mx, 0, zb1 + 6.0), (0.4, 6.0, 0.3))
    k.box("SteelPaintDark", (mx, 0, zb1 + 7.5), (0.3, 4.4, 0.25))
    for sy in (-1.8, 1.8):
        k.sphere("ShipWhite", (mx - 1.0, sy, zb1 + 4.0), 0.55)
    # gangway tower (motion-compensated access system; the gangway asset sits on top) - twin position
    gx, gy = SOV["gangway"]
    k.cyl("ShipWhite", (gx, gy, D), (gx, gy, D + SOV["pedestal_h"]), 1.6, seg=28)
    k.cyl("SteelPaintGrey", (gx, gy, D + SOV["pedestal_h"]), (gx, gy, D + SOV["pedestal_h"] + 0.6), 2.2, seg=28)
    for z in np.arange(D + 1.5, D + SOV["pedestal_h"], 2.0):                                 # access ladder cage
        k.box("SafetyYellow", (gx - 1.7, gy, z), (0.08, 0.9, 0.08))
    k.box("SafetyYellow", (gx, gy, D + SOV["pedestal_h"] + 0.9), (5.0, 5.0, 0.1))
    # aft deck: 3D-compensated crane (port), containers, rescue boat
    k.cyl("CraneYellow", (-8.0, 6.5, D), (-8.0, 6.5, D + 4.0), 1.0, seg=16)
    k.box("CraneYellow", (-8.0, 6.5, D + 4.6), (3.0, 2.4, 1.6))
    _truss(k, (-8.0, 6.5, D + 5.4), (-24.0, 4.0, D + 11.0), 1.0, 6, "CraneYellow", "CraneYellow", 0.12, 0.06)
    for i, (x, y, c) in enumerate(((-22.0, -4.5, "PaintBlue"), (-22.0, -1.8, "SteelPaintGrey"), (-28.5, -4.5, "RailingRed"))):
        k.box(c, (x, y, D + 1.3), (6.06, 2.44, 2.59))
    C.lifeboat(k, (10.0, SB2 + 1.4, 15.0), L=6.0, B=2.4, H=1.6)
    for sy in (-1, 1):                                                                       # fenders aft
        for x in (-38.0, -30.0):
            k.cyl("Rubber", (x, sy * (sov_hb(x, D - 1.5) + 0.2), D - 2.2), (x, sy * (sov_hb(x, D - 1.5) + 0.2), D - 0.6),
                  0.45, seg=10)
    C.write("sov", k, dict(partId="ServiceOperationVessel", canonicalName="Service operation vessel (generic SX175 type)",
                           vesselName="", L_m=L, B_m=SOV["B"], depth_m=D, pedestalTop_m=D + SOV["pedestal_h"] + 0.6,
                           source="reference/vessels/sov_generic_sx175 (Ulstein SX175 photos; look only)",
                           geometryStatus="ILLUSTRATIVE (no SOV is published for Windanker; unnamed, no livery)"))


def w2w_gangway():
    """motion-compensated telescopic gangway: slewing head, box-girder outer section, telescopic inner section,
    landing tip (origin at the slewing pivot on the pedestal top, along +x)."""
    k = Kit()
    Lg, w = 24.0, 1.3
    k.cyl("SteelPaintGrey", (0, 0, 0), (0, 0, 0.8), 1.8, seg=24)
    k.box("SteelPaintGrey", (0.6, 0, 1.2), (2.8, 2.6, 1.0))
    for sy in (-1, 1):
        k.cyl("Galvanized", (-0.6, sy * 0.9, 0.9), (3.5, sy * 0.9, 0.2), 0.16, seg=8)          # luffing cylinders
    for x0, x1, ww, mat in ((0.8, 15.0, w + 0.3, "ShipWhite"), (14.0, Lg, w, "SteelPaintWhite")):
        k.box(mat, ((x0 + x1) / 2, 0, 1.25), (x1 - x0, ww, 0.14))
        for sy in (-1, 1):
            k.box(mat, ((x0 + x1) / 2, sy * ww / 2, 0.95), (x1 - x0, 0.1, 0.7))
            k.cyl("SafetyYellow", (x0, sy * ww / 2, 2.35), (x1, sy * ww / 2, 2.35), 0.04, seg=6)
            k.cyl("SafetyYellow", (x0, sy * ww / 2, 1.8), (x1, sy * ww / 2, 1.8), 0.025, seg=6)
            for x in np.arange(x0, x1 + 0.01, 1.5):
                k.cyl("SafetyYellow", (x, sy * ww / 2, 1.3), (x, sy * ww / 2, 2.35), 0.025, seg=6, caps=False)
    k.box("RescueOrange", (Lg + 0.25, 0, 1.0), (0.5, w + 0.4, 0.35))                        # landing tip (bumper)
    C.write("w2w_gangway", k, dict(partId="MotionCompensatedGangway", length_m=Lg, geometryStatus="ILLUSTRATIVE"))


def luff_rope():
    """one luffing-rope group: 4 parallel falls, unit length along +z (scaled per frame between the A-frame sheave and
    the boom-head sheave)."""
    k = Kit()
    for dx, dy in ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)):
        k.cyl("SteelRope", (dx, dy, 0.0), (dx, dy, 1.0), 0.045, seg=6, caps=False)
    C.write("wtiv_luff_rope", k, dict(partId="WTIVLuffingRope", geometryStatus="ILLUSTRATIVE (4 falls per side)"))


def main():
    print("materials:", C.build_materials())
    luff_rope()
    rotra_vente()
    rotra_vente(closed=True)
    norse_energi()
    sov()
    w2w_gangway()


if __name__ == "__main__":
    main()
