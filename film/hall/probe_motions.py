import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdSkel
I = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/People/Characters/"
ch = Usd.Stage.Open(I + "male_adult_construction_01_new/male_adult_construction_01_new.usd")
sk = [UsdSkel.Skeleton(p) for p in ch.Traverse() if p.IsA(UsdSkel.Skeleton)][0]
cj = list(sk.GetJointsAttr().Get())
rest = sk.GetRestTransformsAttr().Get()
print("char joints", len(cj), "rest root/hip translation", rest[0].ExtractTranslation(), rest[1].ExtractTranslation())
xf = sk.GetPrim()
p = xf
while p and p.GetPath() != "/":
    from pxr import UsdGeom
    ops = UsdGeom.Xformable(p).GetOrderedXformOps() if p.IsA(UsdGeom.Xformable) else []
    print("  ", p.GetPath(), [(o.GetOpName(), o.Get()) for o in ops][:3])
    p = p.GetParent()
for m in ["Worker_Idle_Pose", "StandingDiscussion_LookingDown_M", "TrafficGuard_M", "Market_Sales_SortOut_M",
          "Market_Sales_Assisting_M", "Kitchen_CleanTable_M"]:
    st = Usd.Stage.Open(I + f"Reallusion/Worker/Motions/{m}.usd")
    an = [UsdSkel.Animation(p) for p in st.Traverse() if p.IsA(UsdSkel.Animation)][0]
    j = list(an.GetJointsAttr().Get())
    ts = an.GetRotationsAttr().GetTimeSamples()
    tr = an.GetTranslationsAttr().Get(ts[0] if ts else 0)
    print(m, "joints==char", j == cj, "nsamples", len(ts), "t", (ts[0], ts[-1]) if ts else None,
          "root tr", tr[0] if tr is not None else None, "hip tr", tr[1] if tr is not None and len(tr) > 1 else None)
w = Usd.Stage.Open(I + "Reallusion/Worker/Worker.usd")
an = [UsdSkel.Animation(p) for p in w.Traverse() if p.IsA(UsdSkel.Animation)][0]
ts = an.GetRotationsAttr().GetTimeSamples()
print("Worker.usd own anim joints==char", list(an.GetJointsAttr().Get()) == cj, "nsamples", len(ts), (ts[0], ts[-1]) if ts else None)
app.close()
