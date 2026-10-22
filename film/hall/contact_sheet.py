"""Contact sheet of the hall test frames: one frame per shot (the middle one), labelled id / sid / window / time-lapse.
Run: ~/Documents/LabAutomationTwin/env_cad/bin/python film/hall/contact_sheet.py <src dir> <out.jpg> [frame index]"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FILM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src, out = sys.argv[1], sys.argv[2]
j = int(sys.argv[3]) if len(sys.argv) > 3 else 1
W = json.load(open(os.path.join(FILM, "film", "hall", "windows_next.json")))
items = [(k, w) for k, w in W.items() if not k.startswith("_")]
cols, tw, th = 4, 640, 360
rows = (len(items) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * (th + 30)), (20, 20, 20))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except OSError:
    font = ImageFont.load_default()
for n, (k, w) in enumerate(items):
    key = f"{w['sid']}_{w['new'][0]}_{w['new'][1]}"
    f = os.path.join(src, f"{key}_{j}.jpg")
    x, y = (n % cols) * tw, (n // cols) * (th + 30)
    if os.path.exists(f):
        sheet.paste(Image.open(f).convert("RGB").resize((tw, th)), (x, y + 30))
    tl = f"  TL {w['time_lapse_s']:.1f}s" if w["time_lapse"] else ""
    d.text((x + 6, y + 5), f"{k}  {w['sid']} {w['new'][0]}-{w['new'][1]} s{tl}", fill=(255, 220, 60), font=font)
sheet.save(out, quality=88)
print(out)
