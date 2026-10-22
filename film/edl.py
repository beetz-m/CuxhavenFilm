"""Edit decision list of the Cuxhaven film: what is shown, in which order, for how long, with which titles.

Sources (read-only, by path): the Cuxhaven twin's process renders (clean 12 fps frames, no caption bar), its physics
episodes (clean frames), its path-traced stills, and this project's new service-robot animation (build/service/).
Captions quote the Cuxhaven register via titles.value() and show its status labels.
"""

from __future__ import annotations

import json
import os

import imageio_ffmpeg

from titles import card, value

TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERIOR = "eq=contrast=1.14:brightness=0.05:saturation=1.28:gamma=1.22,unsharp=5:5:0.5"
OUTDOOR = "eq=contrast=1.06:saturation=1.12"
GREY_ON_GREY = "eq=contrast=1.26:brightness=0.05:saturation=1.35:gamma=1.2,unsharp=5:5:0.6"


def P(sid, a, b, **kw):
    """A process-render segment from its clean frames (12 fps)."""
    return dict(name=f"{sid} {a}-{b}", seq=f"{TWIN}/renders/process/frames/{sid}", src_fps=12, **{"in": a, "out": b},
                grade=kw.pop("grade", INTERIOR), **kw)


# grades for the film's own hall renders (brighter than the twin's clips): softer versions of the clip grades
HALL_GRADE = {INTERIOR: "eq=contrast=1.06:saturation=1.12",
              GREY_ON_GREY: "eq=contrast=1.12:saturation=1.18:gamma=1.04"}


def HALL(sid, a, b, trim=None, **kw):
    """A hall shot re-rendered for the film (film/render_hall_clips.py: at most two varied, animated workers,
    occasional vehicles on collision-checked paths, realistic materials, post effects), 24 fps frames in
    build/hall_post/<sid>_<a>_<b>/ with a margin before `a` (meta.json); falls back to the twin's clip.
    trim=(t0, t1): play only clip seconds t0..t1 of the rendered frames (within a - margin .. b + margin; no
    re-render, a and b stay the frame-directory key)."""
    t0, t1 = trim or (a, b)
    d = os.path.join(ROOT, "build", "hall_post", f"{sid}_{a}_{b}")
    meta = os.path.join(d, "meta.json")
    if not os.path.exists(meta):
        return P(sid, t0, t1, **kw)
    meta = json.load(open(meta))
    m = meta["margin_s"]
    first, last = a - m, meta.get("out_s", b) + (meta.get("margin_s", 1.0))
    assert first - 1e-6 <= t0 < t1 <= last + 1e-6, (sid, a, b, trim, "outside the rendered frames")
    grade = kw.pop("grade", INTERIOR)
    # the twin's time-lapse spans (clip seconds; real time = factor x video time) shifted into this sequence;
    # only spans overlapping the played range (assemble.py further drops overlaps under 0.8 s)
    tl = [(round(s["factor"]), m + s["t0"] - a, m + s["t1"] - a) for s in meta.get("time_lapse", ())
          if s["t1"] > t0 and s["t0"] < t1]
    return dict(name=f"{sid} {t0:g}-{t1:g} film", seq=d, src_fps=24, **{"in": m + t0 - a, "out": m + t1 - a},
                grade=HALL_GRADE.get(grade, grade), timelapse=tl, **kw)


SHIP_GRADE = "eq=contrast=1.03:saturation=1.05"   # the realism pass already sets tone and saturation


def SHIP(seq, a, b, **kw):
    """A ship shot from build/ships/<seq> (24 fps, film-side realistic named vessels; in/out relative to the
    sequence start)."""
    return dict(name=f"{seq} {a}-{b}", seq=os.path.join(ROOT, "build", "ships", seq), src_fps=24,
                **{"in": a, "out": b}, grade=kw.pop("grade", SHIP_GRADE), **kw)


def E(name, a, b, **kw):
    """A physics-episode segment from its clean frames."""
    d = f"{TWIN}/renders/episodes/{name}"
    n = len([f for f in os.listdir(d) if f.endswith(".png")])
    secs = imageio_ffmpeg.count_frames_and_secs(f"{TWIN}/renders/episodes/{name}.mp4")[1]
    return dict(name=f"{name} {a}-{b}", seq=d, pattern="frame_%05d.png", src_fps=round(n / secs),
                **{"in": a, "out": b}, grade=kw.pop("grade", "eq=contrast=1.05:saturation=1.1"), **kw)


def S(path, dur, zoom=(1.0, 1.1), pan=((0.5, 0.5), (0.5, 0.5)), **kw):
    """A still with a slow camera move; a film-side re-render in build/stills/ (film/render_site_stills.py: new sea)
    replaces the twin's still of the same name."""
    own = os.path.join(ROOT, "build", "stills", os.path.basename(path))
    src = own if os.path.exists(own) else f"{TWIN}/renders/{path}"
    return dict(name=os.path.basename(path), still=src, dur=dur, zoom=zoom, pan=pan, **kw)


def SVC(shot, fallback, **kw):
    """A shot of the new service animation, or a still move when it has not been rendered."""
    path = os.path.join(ROOT, "build", "service", f"{shot}.mp4")
    if os.path.exists(path):
        secs = imageio_ffmpeg.count_frames_and_secs(path)[1]
        return dict(name=f"service {shot}", clip=path, **{"in": 0.0, "out": round(secs - 0.05, 2)}, **kw)
    return S(fallback, 6.0, **kw)


def shots():
    hall_l, hall_w, hall_h = value("hall_length"), value("hall_width"), value("hall_height")
    cmin, cmax = value("crane_capacity_range")
    v = [
        # ---------------------------------------------------------------- opening
        dict(name="title", card=lambda: card(["Cuxhaven:", "building offshore wind"],
                                              "From a nacelle factory on the Elbe to a turbine at sea",
                                              "A research digital twin reconstructed from public sources.\n"
                                              "Not Siemens Gamesa material; no internal data."), dur=6.0, xfade=0.0),
        # ---------------------------------------------------------------- 1 the site
        S("site/geo_northsea.png", 7.0, (1.0, 1.18), ((0.5, 0.5), (0.55, 0.52)),
          chapter=("01", "The site", "Cuxhaven, where the Elbe meets the North Sea")),
        S("site/geo_elbe_low.png", 5.5, (1.05, 1.15), ((0.45, 0.55), (0.55, 0.5))),
        S("site/aerial_site.png", 7.0, (1.0, 1.14), ((0.5, 0.5), (0.52, 0.47)),
          fact=(f"{hall_l:.0f} × {hall_w:.0f} m hall, {hall_h:.0f} m high", ("hall_length", "hall_width", "hall_height"),
                "OpenStreetMap, press"), fact_at=1.0),
        S("site/aerial_port.png", 5.5, (1.08, 1.0), ((0.55, 0.5), (0.48, 0.5))),
        S("site/cut_hall_overview.png", 7.0, (1.0, 1.12), ((0.5, 0.55), (0.5, 0.5)),
          fact=(f"About {value('output_per_day')} nacelle per working day", ("output_per_day",), "IFR case study"),
          fact_at=1.2),
        # ---------------------------------------------------------------- 2 generator
        # hall windows: film/hall/windows_next.json (staged twin choreography, 2026-10-21)
        HALL("G2", 6, 18, trim=(10, 18), chapter=("02", "The generator", "A direct-drive generator, built in five stations"),
          fact=(f"{value('magnet_robots')} robots insert the permanent magnets", ("magnet_robots",), "inauguration"),
          fact_at=2.0, pulse=True),
        S("product/rotor_g2_magnets.png", 4.5, (1.0, 1.15), ((0.5, 0.5), (0.5, 0.45)), pulse=True),
        HALL("G2", 38, 50, speed=1.5, pulse=True),     # AGV run-out of the magnet portal, then the 250 t lift
        HALL("G3", 42, 58, speed=1.6, pulse=True),
        HALL("G3", 90, 98, trim=(92, 97), pulse=True),
        HALL("G4", 42, 50, pulse=True),
        HALL("G5", 79, 91, trim=(79, 84.5), pulse=True,
          fact=(f"Generator ≈ {value('generator_mass')} t", ("generator_mass",), "scaled from IEA 15 MW"), fact_at=3.0),
        E("station_g5", 4, 22, speed=1.8, pulse=True,
          fact=("Physics episode: stator lowered into the magnetised rotor", (), "Isaac Sim physics"), fact_at=0.8),
        # ---------------------------------------------------------------- 3 backend
        HALL("B7", 0, 9, trim=(0, 7.5), chapter=("03", "The backend", "Bedframe, yaw system, power and cooling"), pulse=True),
        HALL("B2", 30, 42, pulse=True,
          fact=(f"{value('cranes_count')} overhead cranes, {cmin}–{cmax} t", ("cranes_count", "crane_capacity_range"),
                "Konecranes"), fact_at=1.5),
        E("station_b2", 0, 14, pulse=True),
        HALL("B4", 43, 52, pulse=True),
        HALL("B5", 53, 61, pulse=True),
        # ---------------------------------------------------------------- 4 hub
        HALL("H2", 37, 45, chapter=("04", "The hub", "Pitch bearings on a 100-tonne casting"), pulse=True),
        HALL("H3", 51, 59, pulse=True),
        HALL("H6", 52, 62, pulse=True,
          fact=(f"Hub ≈ {value('hub_mass')} t", ("hub_mass",), "derived"), fact_at=2.0),
        # ---------------------------------------------------------------- 5 marriage and test
        HALL("M1", 8, 13, grade=GREY_ON_GREY,        # M1A: on the mating axis, the generator comes in
          chapter=("05", "Marriage and test", "An AGV brings the generator to the backend")),
        HALL("M1", 13, 20, grade=GREY_ON_GREY, xfade=0.4,   # M1B: steep from the north, the gap closes to mated
          fact=(f"{value('nacelle_mass')} t per finished nacelle",   # rotor size: SG 14-236 DD caption, offshore
                ("nacelle_mass",), "Siemens Gamesa press release"), fact_at=1.0),
        HALL("T1", 6, 14, trim=(10, 14), grade=GREY_ON_GREY),
        S("product/nacelle_exploded.png", 5.0, (1.0, 1.08)),
        # ---------------------------------------------------------------- 6 logistics
        S("site/ops_heavy_cranes.png", 5.0, (1.0, 1.1), chapter=("06", "Moving the heavy parts",
                                                                 "Cranes, AGVs, SPMTs and a yard gantry")),
        S("site/ops_tugger_trains.png", 5.0, (1.0, 1.1), ((0.5, 0.5), (0.55, 0.5))),
        S("site/ops_agv_pads.png", 4.5, (1.1, 1.0)),
        S("site/aerial_storage.png", 5.5, (1.0, 1.12), ((0.5, 0.5), (0.45, 0.5)),
          fact=(f"{value('haul_route_length')} m haul route from hall to ramp", ("haul_route_length",),
                "OpenStreetMap"), fact_at=1.0),
        # ---------------------------------------------------------------- 7 load-out
        SHIP("L_loadout_0_31", 0, 9, chapter=("07", "Load-out", "Onto Rotra Vente through the bow")),
        SHIP("L_loadout_0_31", 9, 24, speed=1.5),
        SHIP("L_loadout_0_31", 24, 30),
        SHIP("L_loadout_41_52", 1, 11,
             fact=("Four nacelles per trip", ("loadout",), "press video, counted"), fact_at=2.0),
        SHIP("rotra_name", 0, 7,
             fact=("Rotra Vente · RoRo nacelle carrier · 141.5 × 21 m", (), "SOURCED · ship-technology.com"),
             fact_at=1.0),
        # ---------------------------------------------------------------- 8 voyage
        S("offshore/voyage_map.png", 8.0, (1.3, 1.5), ((0.32, 0.5), (0.7, 0.47)),   # below the map's own route label
          chapter=("08", "To the wind farm", "Elbe, Kiel Canal, Baltic Sea"),
          fact=("≈ 390 km by RoRo, then 44 km by jack-up", (), "route ILLUSTRATIVE"), fact_at=2.0),
        # ---------------------------------------------------------------- 9 installation
        SHIP("OFF_install_0_74", 0, 10, chapter=("09", "Installation at sea", "Norse Energi at Windanker turbine T07")),
        SHIP("OFF_install_0_74", 13, 27, speed=2.0),
        SHIP("OFF_install_0_74", 29, 37),
        SHIP("OFF_install_0_74", 38, 52,
             fact=(f"Lifting {value('nacelle_mass')} t onto the tower", ("nacelle_mass",), ""), fact_at=3.0),
        SHIP("OFF_install_0_74", 52, 68),
        SHIP("norse_name", 0, 6,
             fact=("Norse Energi · DEME jack-up · 151 × 62 m · 3,200 t crane", (), "SOURCED · DEME"), fact_at=0.8),
        SHIP("OFF_install_0_74", 69, 74,
             fact=("Installed all 21 Windanker turbines, Jul–Sep 2026", (), "SOURCED · OE Resources"), fact_at=0.5),
        # ---------------------------------------------------------------- 10 rotor
        SHIP("OFF_rotor_17_31", 1, 13, chapter=("10", "The rotor", "Hub first, then three blades")),
        SHIP("OFF_rotor_35_65", 1, 29, speed=2.5),
        SHIP("OFF_rotor_119_131", 1, 11,
             fact=("236 m rotor · SG 14-236 DD", (), "SOURCED · offshoreWIND.biz 2026"), fact_at=2.0),
        # ---------------------------------------------------------------- 11 service robots
        SVC("approach", "service/svc_overview.png",
            chapter=("11", "Robots on the turbine", "A robot-only service visit"),
            fact=("Service operation vessel (generic SX175 type)", (), "ILLUSTRATIVE"), fact_at=3.5),
        SVC("drone_blade", "service/svc_inspection_drone.png",
            fact=("Inspection drone: RGB, thermal, LiDAR", (), "mission M1, ILLUSTRATIVE"), fact_at=1.0),
        SVC("contact_drone", "service/svc_blade_contact.png",
            fact=("Contact drone tests the lightning receptors", (), "IEC 61400-24"), fact_at=1.0),
        SVC("hoist_lift", "service/svc_tp_gangway.png",
            fact=("The nacelle crane hoists the robots aboard", (), "mission M1"), fact_at=1.0),
        SVC("hoist_hatch", "service/svc_crane_hatch.png"),
        SVC("nacelle_robots", "service/svc_robot_pump.png",
            fact=("ANYmal services the cooling, Spot inspects", (), "mission M1"), fact_at=1.0),
        E("filter_B", 4, 26, speed=1.4,
          fact=("Physics episode: robot and crane change a coolant filter", (), "Isaac Sim physics"), fact_at=0.8),
        SVC("cargo_drone", "service/svc_roof_drone.png",
            fact=("Cargo drone delivers the filter kit to the roof", (), "mission M1"), fact_at=1.0),
        # ---------------------------------------------------------------- closing
        S("offshore/off_farm.png", 6.0, (1.25, 1.05), ((0.5, 0.5), (0.5, 0.5))),
        dict(name="end", card=lambda: card(["From the quay", "to the sea"],
                                            "Cuxhaven nacelle site · research digital twin",
                                            "Isaac Sim 6.2 renders and physics episodes · CRAM robot plans\n"
                                            "Reconstructed from public sources; every number carries a status label.\n"
                                            "Not Siemens Gamesa material. Score: procedural."), dur=7.0, xfade=1.0),
    ]
    return v
