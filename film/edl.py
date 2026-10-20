"""Edit decision list of the Cuxhaven film: what is shown, in which order, for how long, with which titles.

Sources (read-only, by path): the Cuxhaven twin's process renders (clean 12 fps frames, no caption bar), its physics
episodes (clean frames), its path-traced stills, and this project's new service-robot animation (build/service/).
Captions quote the Cuxhaven register via titles.value() and show its status labels.
"""

from __future__ import annotations

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


def E(name, a, b, **kw):
    """A physics-episode segment from its clean frames."""
    d = f"{TWIN}/renders/episodes/{name}"
    n = len([f for f in os.listdir(d) if f.endswith(".png")])
    secs = imageio_ffmpeg.count_frames_and_secs(f"{TWIN}/renders/episodes/{name}.mp4")[1]
    return dict(name=f"{name} {a}-{b}", seq=d, pattern="frame_%05d.png", src_fps=round(n / secs),
                **{"in": a, "out": b}, grade=kw.pop("grade", "eq=contrast=1.05:saturation=1.1"), **kw)


def S(path, dur, zoom=(1.0, 1.1), pan=((0.5, 0.5), (0.5, 0.5)), **kw):
    """A still with a slow camera move."""
    return dict(name=os.path.basename(path), still=f"{TWIN}/renders/{path}", dur=dur, zoom=zoom, pan=pan, **kw)


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
        P("G2", 12, 24, chapter=("02", "The generator", "A direct-drive generator, built in five stations"),
          fact=(f"{value('magnet_robots')} robots insert the permanent magnets", ("magnet_robots",), "inauguration"),
          fact_at=6.0, pulse=True),
        S("product/rotor_g2_magnets.png", 4.5, (1.0, 1.15), ((0.5, 0.5), (0.5, 0.45)), pulse=True),
        P("G3", 24, 40, speed=1.6, pulse=True),
        P("G3", 48, 56, pulse=True),
        P("G4", 39, 47, pulse=True),
        P("G5", 15, 27, pulse=True,
          fact=(f"Generator ≈ {value('generator_mass')} t", ("generator_mass",), "scaled from IEA 15 MW"), fact_at=3.0),
        E("station_g5", 4, 22, speed=1.8, pulse=True,
          fact=("Physics episode: stator lowered into the magnetised rotor", (), "Isaac Sim physics"), fact_at=0.8),
        # ---------------------------------------------------------------- 3 backend
        P("B7", 0, 9, chapter=("03", "The backend", "Bedframe, yaw system, power and cooling"), pulse=True),
        P("B2", 60, 72, pulse=True,
          fact=(f"{value('cranes_count')} overhead cranes, {cmin}–{cmax} t", ("cranes_count", "crane_capacity_range"),
                "Konecranes"), fact_at=1.5),
        E("station_b2", 0, 14, pulse=True),
        P("B4", 39, 48, pulse=True),
        P("B5", 48, 56, pulse=True),
        # ---------------------------------------------------------------- 4 hub
        P("H2", 36, 44, chapter=("04", "The hub", "Pitch bearings on a 100-tonne casting"), pulse=True),
        P("H3", 52, 60, pulse=True),
        P("H6", 12, 22, pulse=True,
          fact=(f"Hub ≈ {value('hub_mass')} t", ("hub_mass",), "derived"), fact_at=2.0),
        # ---------------------------------------------------------------- 5 marriage and test
        P("M1", 15, 27, grade=GREY_ON_GREY,
          chapter=("05", "Marriage and test", "Generator meets backend under a 500 t crane"),
          fact=(f"{value('nacelle_mass')} t nacelle for the {value('rotor_diameter')} m rotor",
                ("nacelle_mass", "rotor_diameter"), "Siemens Gamesa press release"), fact_at=6.0),
        P("T1", 12, 20, grade=GREY_ON_GREY),
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
        P("L_loadout", 0, 9, grade=OUTDOOR, chapter=("07", "Load-out", "Onto the RoRo vessel through the bow")),
        P("L_loadout", 9, 24, speed=1.5, grade=OUTDOOR),
        P("L_loadout", 24, 30, grade=OUTDOOR),
        P("L_loadout", 42, 52, grade=OUTDOOR,
          fact=("Four nacelles per trip", ("loadout",), "press video, counted"), fact_at=2.0),
        S("dock/dock_overview.png", 5.0, (1.0, 1.1)),
        # ---------------------------------------------------------------- 8 voyage
        S("offshore/voyage_map.png", 8.0, (1.3, 1.5), ((0.32, 0.5), (0.7, 0.47)),   # below the map's own route label
          chapter=("08", "To the wind farm", "Elbe, Kiel Canal, Baltic Sea"),
          fact=("≈ 390 km by RoRo, then 44 km by jack-up", (), "route ILLUSTRATIVE"), fact_at=2.0),
        # ---------------------------------------------------------------- 9 installation
        P("OFF_install", 0, 10, grade=OUTDOOR, chapter=("09", "Installation at sea", "A jack-up vessel at turbine T07")),
        P("OFF_install", 13, 27, speed=2.0, grade=OUTDOOR),
        P("OFF_install", 29, 37, grade=OUTDOOR),
        P("OFF_install", 38, 52, grade=OUTDOOR,
          fact=(f"Lifting {value('nacelle_mass')} t onto the tower", ("nacelle_mass",), ""), fact_at=3.0),
        P("OFF_install", 52, 68, grade=OUTDOOR),
        P("OFF_install", 69, 74, grade=OUTDOOR),
        # ---------------------------------------------------------------- 10 rotor
        P("OFF_rotor", 18, 30, grade=OUTDOOR, chapter=("10", "The rotor", "Hub first, then three blades")),
        P("OFF_rotor", 36, 64, speed=2.5, grade=OUTDOOR),
        P("OFF_rotor", 120, 130, grade=OUTDOOR,
          fact=(f"{value('rotor_diameter')} m rotor, 14 MW", ("rotor_diameter", "product"), "press release"),
          fact_at=2.0),
        # ---------------------------------------------------------------- 11 service robots
        SVC("approach", "service/svc_overview.png",
            chapter=("11", "Robots on the turbine", "A robot-only service visit")),
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
