"""Film-side North Sea: the twin's scripts/ocean.py (read-only, by path) with calmer-looking whitecaps for close shots.

The twin's foam picks the most folded texels of the full wind-sea field (Jacobian below a threshold), so short waves of
a few decimetres break as thousands of hard white flecks, and the residual foam is smeared into sharp streaks; close to
the camera that reads as painted brush strokes. Here:
  - the Jacobian is low-passed (sigma 0.7 m) before thresholding: only crests of the longer waves break, as coherent
    patches, at the same coverage target (Monahan-type W ~ 3.84e-6 U^3.41, unchanged);
  - active foam is broken up by a fine lace noise (foam is never a solid white sheet) and is less bright (0.60);
  - residual foam is weaker and softened (blur 2.5 texels) so it reads as a faint trail, not a stroke.
Textures are written to build/ocean (never into the twin).

    import sea as FS
    o = FS.install(stage, center=(x, y), sea=None)    # same contract as OC.install
"""

from __future__ import annotations

import os
import sys

import numpy as np

TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TWIN, "scripts"))
import ocean as OC  # noqa: E402

OC.OUT = os.path.join(FILM, "build", "ocean")      # module global used by the material and the texture writer


def _gauss(a, sigma_px):
    N = a.shape[0]
    f = np.fft.fftfreq(N)[:, None] ** 2 + np.fft.fftfreq(N)[None, :] ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * (np.pi * sigma_px) ** 2 * f)))


class FilmOcean(OC.Ocean):
    FOAM = (0.45, 0.47, 0.47)
    # open-sea tuning (2026-10-21): at 9 m/s even Monahan coverage (0.69 %) rendered as a dense dash pattern from
    # 20-60 m camera heights (the texture tiles every 256 m and grazing views compress it). A/B test
    # (build/sea_test/sheet_ABCD.jpg, sheet_AEFB.jpg): foam is the cause, roughness is not; variant F chosen:
    # 0.22 x Monahan (0.15 % active whitecaps, ~ a 6-7 m/s sea), dimmer foam. ILLUSTRATIVE (visual choice, not a
    # sea-state claim); crests of the longer waves only (sigma 1.6 m), weak trails
    SIGMA_M, COV_GAIN, RESID = 1.6, 0.22, 0.18

    def _textures(self, t):
        from PIL import Image
        os.makedirs(OC.OUT, exist_ok=True)
        h, dx, dy, sx, sy, J = self.wind.field(t)
        self._f = (h, dx, dy, sx, sy, J)
        N, px = self.N, self.L / self.N
        U = self.sea["wind_ms"]
        cov = min(0.06, 3.84e-6 * U ** 3.41 * self.COV_GAIN)
        Js = _gauss(J, self.SIGMA_M / px)                                   # only the longer waves' crests break
        thr = np.quantile(Js, cov)
        w_ = max(1e-4, float(np.quantile(Js, cov * 2.5) - thr))
        active = np.clip((thr + w_ - Js) / (2 * w_), 0, 1)
        rng = np.random.default_rng(3)

        def noise(scale_px, seed):
            r = np.random.default_rng(seed).standard_normal((N, N))
            n = _gauss(r, scale_px)
            return (n - n.mean()) / n.std()
        lace = np.clip(0.55 + 0.45 * noise(1.5, 11), 0, 1)        # fine foam texture (~0.2 m)
        a = np.radians(self.wave_dir)
        resid = np.zeros_like(active)
        for k in range(1, int(round(5.0 / px))):
            sh = (-int(round(np.sin(a) * k)), -int(round(np.cos(a) * k)))
            resid = np.maximum(resid, np.roll(active, sh, axis=(0, 1)) * np.exp(-k * px / 1.5))
        resid = _gauss(resid, 2.5) * np.clip(0.5 + 0.6 * noise(N / 17.0, 4), 0, 1)
        foam = np.clip(np.maximum(active * lace, self.RESID * resid), 0, 1)
        foam = np.clip(_gauss(foam, 1.0), 0, 1)
        self.foam_cover = float((foam > 0.5).mean())
        self.whitecap_cover = float((active > 0.5).mean())
        crest = np.clip((h - h.mean()) / (h.std() * 2.5), 0, 1) * np.clip(np.hypot(sx, sy) * 2.0, 0, 1)
        alb = np.array(self.WATER)[None, None] * (1 - crest[..., None]) + np.array(self.CREST)[None, None] * crest[..., None]
        alb = alb * (1 - foam[..., None]) + np.array(self.FOAM)[None, None] * foam[..., None]
        g = _gauss(rng.standard_normal((N, N)), N / 3.0 / (2 * np.pi))
        g = (g - g.mean()) / g.std()
        rough = np.clip(0.035 + 0.5 * foam + 0.025 * g, 0.015, 1)
        nrm = np.stack([-sx, -sy, np.ones_like(sx)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        srgb = lambda c: np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)  # noqa
        flip = lambda a_: a_[::-1]                                                                                  # noqa
        Image.fromarray((flip(srgb(alb)) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(OC.OUT, "sea_albedo.png"))
        Image.fromarray((flip(rough) * 255).astype(np.uint8)).save(os.path.join(OC.OUT, "sea_rough.png"))
        Image.fromarray(((flip(nrm) * 0.5 + 0.5) * 255).astype(np.uint8)).save(os.path.join(OC.OUT, "sea_normal.png"))
        det = os.path.join(OC.OUT, "sea_detail_normal.png")
        if not os.path.exists(det):                    # the twin's capillary detail normal, unchanged
            import shutil
            shutil.copy(os.path.join(TWIN, "assets", "ocean", "sea_detail_normal.png"), det)
        print(f"[sea] film textures: Hs {4 * h.std():.2f} m, active whitecaps {100 * self.whitecap_cover:.2f} % "
              f"(target {100 * cov:.2f} %), all foam {100 * self.foam_cover:.2f} %", flush=True)


def install(stage, center=(0.0, 0.0), path="/World/RenderOnly/NorthSea", t=0.0, R=2000.0, sea=None, hide=True):
    """OC.install with the film's foam (same hiding of the old water in the session layer)."""
    from pxr import UsdGeom
    stage.SetEditTarget(stage.GetSessionLayer())
    hidden = []
    if hide:
        for p in stage.Traverse():
            if p.GetName() in OC.WATER_NAMES and p.IsA(UsdGeom.Imageable) and not str(p.GetPath()).startswith(path):
                UsdGeom.Imageable(p).MakeInvisible()
                hidden.append(str(p.GetPath()))
    o = FilmOcean(sea=sea)
    o.add(stage, path, center=center, t=t, R=R)
    print(f"[sea] installed at {path}, centre {center}; hidden: {hidden}", flush=True)
    return o


if __name__ == "__main__":                              # texture-only comparison, no Isaac
    from PIL import Image
    o = FilmOcean()
    new = Image.open(os.path.join(OC.OUT, "sea_albedo.png")).convert("RGB")
    old = Image.open(os.path.join(TWIN, "assets", "ocean", "sea_albedo.png")).convert("RGB")
    crop = (0, 0, 512, 512)                                # 64 m x 64 m
    cmp_ = Image.new("RGB", (1024, 512))
    cmp_.paste(old.crop(crop), (0, 0))
    cmp_.paste(new.crop(crop), (512, 0))
    out = os.path.join(FILM, "build", "ocean", "foam_compare.png")
    cmp_.save(out)
    print(out)
