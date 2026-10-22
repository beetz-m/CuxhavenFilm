"""Swap the twin's vessel assets for the film's realistic, named models (build/vessels/*.usda) - in the stage's
SESSION layer only, so the twin's scenes stay untouched.

Every prim whose authored references point at one of the twin's vessel assets gets an explicit reference list in the
session layer that points at the film model with the same name (same origin, axes, scale and articulation split).
"""

import os

from pxr import Sdf, Usd

FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VES = os.path.join(FILM, "build", "vessels")
NAMES = ("roro_vessel.usda", "roro_visor.usda", "wtiv_hull.usda", "wtiv_leg.usda", "wtiv_crane.usda", "wtiv_boom.usda",
         "sov.usda", "w2w_gangway.usda")


def swap(stage, closed_visor=False):
    """Returns the swapped prim paths. closed_visor: use the hull variant with the visor arms folded."""
    session = stage.GetSessionLayer()
    done = []
    with Usd.EditContext(stage, session):
        for prim in stage.Traverse():
            for spec in prim.GetPrimStack():
                if spec.layer == session:
                    continue
                items = list(spec.referenceList.GetAddedOrExplicitItems())
                hit = [r for r in items if os.path.basename(r.assetPath) in NAMES
                       and not r.assetPath.startswith(VES)]
                if not hit:
                    continue
                new = []
                for r in items:
                    name = os.path.basename(r.assetPath)
                    if name in NAMES:
                        if closed_visor and name == "roro_vessel.usda":
                            name = "roro_vessel_closed.usda"
                        new.append(Sdf.Reference(os.path.join(VES, name), r.primPath, r.layerOffset))
                    else:
                        new.append(r)
                prim.GetReferences().SetReferences(new)
                done.append(str(prim.GetPath()))
                break
    missing = [n for n in NAMES if not os.path.exists(os.path.join(VES, n))]
    if missing:
        raise FileNotFoundError(f"film vessel assets missing (run film/vessels/build_vessels.py): {missing}")
    return done
