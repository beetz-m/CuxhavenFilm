"""Film-side look fixes for the hall renders, applied in memory after the twin's realism pass (the twin is never written).

cast_iron(): the v2 realism set gives CastPrimerGrey (hub casting, kingpin, stator support, bedframe casting) metallic
0.9. A rough metal shows only what it reflects, and in the hall interior that is mostly dark roof, so the castings read
black. A primed casting is a painted (dielectric) surface: metallic 0.08, the v2 albedo / roughness / normal textures
kept, a slightly lighter cast-iron grey tint. ILLUSTRATIVE (appearance only).

    import hall_looks as HL
    HL.cast_iron()           # after realism.apply_materials(), before opening the stage
"""

import os

TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
LIB = os.path.join(TWIN, "assets", "materials", "site_materials.usda")
_KEEP = []
CAST = dict(metallic=0.08, tint=(0.92, 0.93, 0.95), roughness_influence=1.0)


def cast_iron(name="CastPrimerGrey", verbose=True):
    from pxr import Sdf
    layer = Sdf.Layer.FindOrOpen(LIB)
    _KEEP.append(layer)            # keep the edited layer alive, or USD reloads the unchanged file
    sh = layer.GetPrimAtPath(f"/Looks/{name}/Shader")
    if sh is None:
        print(f"[hall_looks] {name}: no shader in the library", flush=True)
        return False

    def put(attr, tname, value):
        a = sh.attributes.get(f"inputs:{attr}")
        if a is None:
            a = Sdf.AttributeSpec(sh, f"inputs:{attr}", tname)
        a.default = value

    with Sdf.ChangeBlock():
        put("metallic_constant", Sdf.ValueTypeNames.Float, CAST["metallic"])
        put("diffuse_tint", Sdf.ValueTypeNames.Color3f, CAST["tint"])
        put("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, CAST["roughness_influence"])
    if verbose:
        print(f"[hall_looks] {name}: cast-iron primer, metallic {CAST['metallic']} (in memory)", flush=True)
    return True
