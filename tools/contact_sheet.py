"""Contact sheet: 10 clockwise frames + eye close-ups + 5-direction check row.

    python tools/contact_sheet.py      -> contact_sheet.png
"""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
meta = json.load(open(os.path.join(ROOT, "metadata.json")))
report_path = os.path.join(ROOT, "checks", "report.json")
report = json.load(open(report_path)) if os.path.exists(report_path) else {}


def font(sz, bold=False):
    for f in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


BG, FG, MUTED, ACC = (22, 24, 28), (236, 237, 241), (150, 155, 166), (255, 212, 70)
GREEN, RED = (70, 210, 120), (240, 80, 80)
T = 312                       # tile width
EH = 92                       # eye close-up height
EYE_BOX = (214, 452, 810, 576)  # eye region in the 1024 render (aspect ~ T:EH)
PAD, COLS = 18, 5
HEAD = 150
W = PAD + COLS * (T + PAD)
row_h = T + EH + 30
DIR_H = 60 + EH + 70
H = HEAD + 2 * row_h + DIR_H + PAD
sheet = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(sheet)

d.text((PAD, 22), "Track 01 · Synthetic eye movement — 10 clockwise renders", font=font(34, True), fill=FG)
d.text((PAD, 70), "θk = 2πk/10, h = sin θk, v = cos θk (k = 0 at top) · fixation 35 cm ahead of eye midpoint (±5 cm) · "
       "camera locked 40 cm · 1024×1024", font=font(19), fill=MUTED)
d.text((PAD, 98), "Rings = metadata pupil_center_2D (L green, R magenta). Lids follow vertical gaze.",
       font=font(19), fill=MUTED)

# small clockwise dial in the header
cx, cy, r = W - 80, 75, 48
d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=MUTED, width=2)
for k in range(10):
    th = 2 * math.pi * k / 10
    px, py = cx + r * math.sin(th), cy - r * math.cos(th)
    d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=ACC if k == 0 else FG)
d.text((cx - r - 105, cy - 10), "clockwise →", font=font(17), fill=MUTED)


def eyes_crop(img):
    return img.crop(EYE_BOX).resize((T, EH), Image.LANCZOS)


for r_ in meta:
    k = r_["frame"]
    col, row = k % COLS, k // COLS
    x, y = PAD + col * (T + PAD), HEAD + row * row_h
    img = Image.open(os.path.join(ROOT, "renders", f"frame_{k:02d}.png")).convert("RGB")
    dd = ImageDraw.Draw(img)
    for s, c in (("L", (40, 230, 90)), ("R", (240, 60, 220))):
        px, py = r_[f"pupil_center_2D_{s}"]
        dd.ellipse((px - 6, py - 6, px + 6, py + 6), outline=c, width=2)
    sheet.paste(img.resize((T, T), Image.LANCZOS), (x, y))
    sheet.paste(eyes_crop(img), (x, y + T))
    d.rectangle((x - 1, y - 1, x + T, y + T + EH), outline=(70, 74, 86))
    d.text((x + 10, y + 8), f"k={k}  θ={math.degrees(r_['theta']):.0f}°", font=font(20, True), fill=ACC)
    d.text((x + 10, y + 34), f"h={r_['input']['h']:+.3f}  v={r_['input']['v']:+.3f}", font=font(16), fill=FG)
    # gaze arrow
    ax, ay = x + T - 34, y + 34
    th = r_["theta"]
    d.ellipse((ax - 18, ay - 18, ax + 18, ay + 18), outline=MUTED, width=1)
    d.line((ax, ay, ax + 16 * math.sin(th), ay - 16 * math.cos(th)), fill=ACC, width=3)
    d.text((x + 2, y + T + EH + 4), f"vergence {r_['vergence_deg']:.2f}°   eye L rot "
           f"({r_['eye_rotation_L'][0]:+.1f}, {r_['eye_rotation_L'][2]:+.1f})°", font=font(14), fill=MUTED)

# direction check row
y0 = HEAD + 2 * row_h + 10
dirs = report.get("directions", {})
ok_all = dirs.get("passed")
d.text((PAD, y0), "5-direction check (pupils detected from pixels vs centre render): " +
       ("ALL PASS" if ok_all else "see checks/report.json"), font=font(22, True), fill=GREEN if ok_all else ACC)
rows = {r_["direction"].lower(): r_ for r_ in dirs.get("rows", [])}
names = ["center", "left", "right", "up", "down"]
cw = (W - PAD * (len(names) + 1)) // len(names)
for i, n in enumerate(names):
    x = PAD + i * (cw + PAD)
    img = Image.open(os.path.join(ROOT, "checks", f"{n}.png")).convert("RGB")
    sheet.paste(img.crop(EYE_BOX).resize((cw, int(cw * EH / T)), Image.LANCZOS), (x, y0 + 40))
    rr = rows.get(n)
    label = n.upper()
    if rr:
        dx = sum(e["dx"] for e in rr["eyes"]) / 2
        dy = sum(e["dy"] for e in rr["eyes"]) / 2
        label += f"  {'PASS' if rr['passed'] else 'FAIL'}  Δ=({dx:+.1f}, {dy:+.1f}) px"
    d.text((x + 2, y0 + 40 + int(cw * EH / T) + 6), label, font=font(17, True),
           fill=(GREEN if rr and rr["passed"] else FG) if n != "center" else FG)

out = os.path.join(ROOT, "contact_sheet.png")
sheet.save(out)
print("saved", out)
