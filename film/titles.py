"""Title cards, chapter titles and fact captions for the Cuxhaven film, as RGBA overlays (PIL).

Facts come from the Cuxhaven twin's register (~/Documents/CuxhavenSiteTwin/site/params.py, loaded read-only by path)
and carry the register's status label, so the film shows how certain each number is.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
TWIN = os.path.expanduser("~/Documents/CuxhavenSiteTwin")
INTER = "/usr/share/fonts/opentype/inter/InterDisplay-{}.otf"
PLEX = "/usr/share/texlive/texmf-dist/fonts/opentype/ibm/plex/IBMPlex{}.otf"
ACCENT = (0, 178, 169)       # sea teal
WHITE = (245, 247, 248)
MUTED = (186, 198, 204)


@lru_cache(None)
def font(face: str, size: int) -> ImageFont.FreeTypeFont:
    family, weight = face.split(":")
    path = INTER.format(weight) if family == "inter" else PLEX.format(weight)
    return ImageFont.truetype(path, size)


@lru_cache(None)
def register():
    spec = importlib.util.spec_from_file_location("cux_params", os.path.join(TWIN, "site", "params.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.REGISTER


def status(key: str) -> str:
    return register()[key][2]


def value(key: str):
    return register()[key][0]


def _shadowed(layer: Image.Image, radius: int = 18, alpha: int = 150) -> Image.Image:
    """Put a soft dark halo under the drawn text so it reads on any footage."""
    a = layer.split()[3].filter(ImageFilter.GaussianBlur(radius))
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    shadow.putalpha(a.point(lambda v: min(alpha, v * 2)))
    return Image.alpha_composite(shadow, layer)


def _band(y0: int, y1: int, x1: int = 1250, alpha: int = 150) -> Image.Image:
    """A dark band fading out to the right (and softly at top and bottom) for text over bright footage."""
    import numpy as np

    a = np.zeros((H, W), np.float32)
    xs = np.clip(1.0 - np.arange(W) / x1, 0.0, 1.0) ** 1.4
    ys = np.zeros(H, np.float32)
    ys[y0:y1] = 1.0
    ys = np.convolve(ys, np.ones(60) / 60, mode="same")
    a = np.outer(ys, xs) * alpha
    band = Image.new("RGBA", (W, H), (4, 10, 16, 0))
    band.putalpha(Image.fromarray(a.astype(np.uint8)))
    return band


def card(lines, sub=None, small=None, bg=(8, 14, 20)) -> Image.Image:
    """A full-frame title card: big lines, an optional subtitle and a small note at the bottom."""
    im = Image.new("RGBA", (W, H), bg + (255,))
    d = ImageDraw.Draw(im)
    y = 380 - 40 * (len(lines) - 1)
    d.rectangle([160, y - 40, 160 + 90, y - 34], fill=ACCENT + (255,))
    for line in lines:
        d.text((160, y), line, font=font("inter:Bold", 92), fill=WHITE + (255,))
        y += 108
    if sub:
        d.text((162, y + 18), sub, font=font("inter:Regular", 38), fill=MUTED + (255,))
    if small:
        for i, s in enumerate(small.split("\n")):
            d.text((162, H - 150 + 34 * i), s, font=font("plex:Mono-Regular", 24), fill=MUTED + (255,))
    return im


def chapter(number: str, title: str, kicker: str = "") -> Image.Image:
    """Chapter title, top left: a number, a title, a one-line kicker."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((110, 92), number, font=font("plex:Mono-SemiBold", 30), fill=ACCENT + (255,))
    d.text((106, 132), title, font=font("inter:Bold", 76), fill=WHITE + (255,))
    if kicker:
        d.text((110, 228), kicker, font=font("inter:Medium", 34), fill=WHITE + (235,))
    return Image.alpha_composite(_band(60, 300, 1300, 150), _shadowed(im))


def fact(text: str, keys=(), source: str = "") -> Image.Image:
    """Lower-third caption with the register status of the numbers it quotes."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x, y = 110, H - 190
    d.rectangle([x, y + 6, x + 6, y + 82], fill=ACCENT + (255,))
    d.text((x + 28, y), text, font=font("inter:SemiBold", 46), fill=WHITE + (255,))
    tags = sorted({status(k) for k in keys})
    tag = " · ".join(tags + ([source] if source else []))
    if tag:
        d.text((x + 30, y + 62), tag, font=font("plex:Mono-Regular", 24), fill=MUTED + (255,))
    return Image.alpha_composite(_band(H - 230, H - 70, 1350, 165), _shadowed(im, radius=22, alpha=170))


def watermark() -> Image.Image:
    """Small permanent tag: what this footage is."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    t = "RESEARCH DIGITAL TWIN · PUBLIC SOURCES · ISAAC SIM"
    tw = d.textlength(t, font=font("plex:Mono-Regular", 20))
    d.text((W - 80 - tw, 60), t, font=font("plex:Mono-Regular", 20), fill=(255, 255, 255, 150))
    return _shadowed(im, radius=8, alpha=90)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "titles_preview")
    os.makedirs(out, exist_ok=True)
    card(["Cuxhaven:", "building offshore wind"], "A research digital twin of a nacelle factory, from the quay to the sea",
         "Reconstructed from public sources. Not Siemens Gamesa material.").save(os.path.join(out, "card.png"))
    chapter("02", "The generator line", "Magnets, stator segments, 274 t of copper and steel").save(os.path.join(out, "chapter.png"))
    fact(f"{value('nacelle_mass')} t nacelle · {value('rotor_diameter')} m rotor", ("nacelle_mass", "rotor_diameter"),
         "press release").save(os.path.join(out, "fact.png"))
    print("previews in", out)
