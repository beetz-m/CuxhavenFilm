"""Probe the Isaac people / vehicle assets: prim structure, skeleton, bbox, up/forward axes. Headless, no render."""
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd
from pxr import Usd, UsdGeom, UsdSkel
ISAAC = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/"
ASSETS = ["People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
          "People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
          "People/Characters/original_male_adult_construction_02/male_adult_construction_02.usd",
          "People/Characters/Reallusion/Worker/Worker.usd",
          "People/Animations/stand_idle_loop.skelanim.usd", "People/Animations/LookAround.skelanim.usd",
          "People/Animations/Sit.skelanim.usd", "People/Animations/stand_walk_loop.skelanim.usd",
          "People/Characters/Reallusion/Worker/Motions/Worker_Idle_Pose.usd",
          "People/Characters/Reallusion/Worker/Motions/StandingDiscussion_LookingDown_M.usd",
          "Props/Forklift/forklift.usd", "Robots/IsaacSim/ForkliftC/forklift_c.usd",
          "Robots/IsaacSim/ForkliftB/forklift_b.usd", "Robots/Idealworks/iwhub/iw_hub_static.usd",
          "Props/Pallet/pallet.usd", "Props/KLT_Bin/small_KLT.usd"]
for a in ASSETS:
    st = Usd.Stage.Open(ISAAC + a)
    if not st:
        print("FAIL", a); continue
    dp = st.GetDefaultPrim()
    up = UsdGeom.GetStageUpAxis(st); mpu = UsdGeom.GetStageMetersPerUnit(st)
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    r = bc.ComputeWorldBound(dp).ComputeAlignedRange() if dp else None
    skels = [p for p in st.Traverse() if p.IsA(UsdSkel.Skeleton)]
    roots = [p for p in st.Traverse() if p.IsA(UsdSkel.Root)]
    anims = [p for p in st.Traverse() if p.IsA(UsdSkel.Animation)]
    print(f"== {a}\n   default={dp.GetPath() if dp else None} up={up} mpu={mpu} tcps={st.GetTimeCodesPerSecond()} "
          f"t=[{st.GetStartTimeCode()},{st.GetEndTimeCode()}]")
    if r:
        print(f"   bbox min={tuple(round(v,3) for v in r.GetMin())} max={tuple(round(v,3) for v in r.GetMax())}")
    for p in roots[:2]:
        b = UsdSkel.BindingAPI(p)
        print("   SkelRoot", p.GetPath(), "animSource", b.GetAnimationSourceRel().GetTargets() if b.GetAnimationSourceRel() else None)
    for p in skels[:2]:
        sk = UsdSkel.Skeleton(p); j = sk.GetJointsAttr().Get() or []
        b = UsdSkel.BindingAPI(p)
        print("   Skeleton", p.GetPath(), len(j), "joints", list(j[:3]), "animSource",
              b.GetAnimationSourceRel().GetTargets() if b.GetAnimationSourceRel() else None)
    for p in anims[:2]:
        an = UsdSkel.Animation(p); j = an.GetJointsAttr().Get() or []
        print("   Animation", p.GetPath(), len(j), "joints", list(j[:3]))
    kids = [c.GetName() for c in (dp.GetChildren() if dp else [])][:12]
    print("   children", kids)
app.close()
