"""Screen content for the application scenes (PNG sequences + per-frame gaze UV for the eye rig).

    python video/screens.py   -> video/build/screens/{keyboard,star}/####.png + gaze.json
"""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "video", "build", "screens")
W, H = 1280, 800


def F(size, bold=True):
    return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", "segoeuib.ttf" if bold else "segoeui.ttf"), size)


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ eye-typing keyboard (Arthur)
ROWS = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]


def key_rect(ch):
    if ch == " ":
        return (W * 0.3, H * 0.86, W * 0.4, H * 0.09)
    if ch == "#":   # SPEAK button
        return (W * 0.74, H * 0.86, W * 0.2, H * 0.09)
    for r, row in enumerate(ROWS):
        i = row.find(ch)
        if i >= 0:
            kw, kh = W / 10.6, H * 0.14
            x0 = (W - len(row) * kw) / 2
            return (x0 + i * kw + 4, H * 0.36 + r * (kh + 10), kw - 8, kh)
    raise KeyError(ch)


def keyboard(n=290, t_type=100, dwell=12, text="HELLO SAM"):
    d = os.path.join(OUT, "keyboard")
    os.makedirs(d, exist_ok=True)
    seq = list(text) + ["#"]
    scan = ["Q", "P", "A", "L", "Z", "M", "T", "G"]       # browsing the keys before typing
    gaze = []
    for f in range(1, n + 1):
        im = Image.new("RGB", (W, H), (12, 18, 30))
        dd = ImageDraw.Draw(im)
        if f < t_type:
            i = (f - 1) * len(scan) // t_type
            active, prog, typed, speaking = scan[i], 0.0, "", False
            nxt = scan[min(i + 1, len(scan) - 1)]
            k = ((f - 1) * len(scan) / t_type) % 1
        else:
            j = (f - t_type) // dwell
            prog = ((f - t_type) % dwell) / dwell
            active = seq[j] if j < len(seq) else "#"
            typed = text[:min(j, len(text))]
            speaking = j >= len(seq)
            nxt, k = active, 0.0
        # text bar
        dd.rounded_rectangle((40, 40, W - 40, 200), 24, fill=(22, 32, 50))
        dd.text((80, 120), typed, font=F(84), fill=(236, 244, 255), anchor="lm")
        tw = dd.textlength(typed, font=F(84))
        if (f // 15) % 2 == 0 and not speaking:
            dd.rectangle((84 + tw, 78, 92 + tw, 162), fill=(72, 230, 255))
        if speaking:
            a = min(1.0, (f - (t_type + dwell * len(seq))) / 12)
            for r in range(3):
                rr = 30 + r * 26 + 10 * math.sin(f * 0.5 + r)
                dd.arc((W - 220 - rr, 120 - rr, W - 220 + rr, 120 + rr), -45, 45, fill=(72, 230, 255), width=7)
            dd.text((W - 330, 120), "🔊", font=F(10), fill=(0, 0, 0))
            dd.text((W - 420, 175), "speaking…", font=F(30, False), fill=(150, 220, 255), anchor="mm")
        for row in ROWS:
            for ch in row:
                x, y, kw, kh = key_rect(ch)
                on = ch == active
                dd.rounded_rectangle((x, y, x + kw, y + kh), 14, fill=(40, 120, 150) if on else (30, 42, 62))
                if on and prog > 0:
                    dd.rounded_rectangle((x, y + kh * (1 - prog), x + kw, y + kh), 14, fill=(72, 230, 255))
                dd.text((x + kw / 2, y + kh / 2), ch, font=F(40), fill=(5, 15, 25) if on else (190, 205, 225), anchor="mm")
        for ch, lab in ((" ", "space"), ("#", "SPEAK")):
            x, y, kw, kh = key_rect(ch)
            on = ch == active
            dd.rounded_rectangle((x, y, x + kw, y + kh), 14, fill=(72, 230, 255) if on else (30, 42, 62))
            dd.text((x + kw / 2, y + kh / 2), lab, font=F(30), fill=(5, 15, 25) if on else (190, 205, 225), anchor="mm")
        # gaze point: centre of the active key (blend toward the next while scanning)
        x, y, kw, kh = key_rect(active)
        x2, y2, kw2, kh2 = key_rect(nxt)
        e = ease(k * 3 - 2) if f < t_type else 0
        gx = (x + kw / 2) * (1 - e) + (x2 + kw2 / 2) * e
        gy = (y + kh / 2) * (1 - e) + (y2 + kh2 / 2) * e
        dd.ellipse((gx - 16, gy - 16, gx + 16, gy + 16), outline=(255, 200, 80), width=4)
        gaze.append([gx / W, gy / H])
        im.save(os.path.join(d, f"{f:04d}.png"))
    json.dump(gaze, open(os.path.join(d, "gaze.json"), "w"))
    print("keyboard", n)


# ------------------------------------------------------------------ star game (Zuri)
def star_points(cx, cy, r1, r2, rot=0.0):
    pts = []
    for i in range(10):
        a = rot - math.pi / 2 + i * math.pi / 5
        r = r1 if i % 2 == 0 else r2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def star(n=215):
    d = os.path.join(OUT, "star")
    os.makedirs(d, exist_ok=True)
    gaze = []
    bg = Image.new("RGB", (W, H))
    bd = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / H
        bd.line((0, y, W, y), fill=(int(120 + 80 * t), int(190 + 40 * t), int(255 - 30 * t)))
    for i, (x, y, r) in enumerate(((180, 640, 260), (640, 700, 300), (1100, 650, 280))):
        bd.ellipse((x - r, y - r * 0.5, x + r, y + r * 0.5 + 300), fill=(110, 200, 120))
    trail = []
    for f in range(1, n + 1):
        t = f / 30.0
        sx = W / 2 + W * 0.33 * math.sin(t * 1.15)
        sy = H * 0.42 + H * 0.2 * math.sin(t * 2.3)
        trail.append((sx, sy))
        im = bg.copy()
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        for j, (tx, ty) in enumerate(trail[-14:]):
            a = int(160 * (j + 1) / 14)
            od.ellipse((tx - 10, ty - 10, tx + 10, ty + 10), fill=(255, 240, 150, a))
        od.polygon(star_points(sx, sy, 78, 34, t * 1.5), fill=(255, 205, 40, 255), outline=(255, 150, 20, 255))
        od.ellipse((sx - 22, sy - 14, sx - 8, sy + 2), fill=(40, 30, 20, 255))
        od.ellipse((sx + 8, sy - 14, sx + 22, sy + 2), fill=(40, 30, 20, 255))
        od.arc((sx - 18, sy - 4, sx + 18, sy + 22), 20, 160, fill=(40, 30, 20, 255), width=5)
        # the tracker's gaze estimate: a soft cyan dot that lags the star a little
        lag = trail[max(0, len(trail) - 5)]
        gx, gy = lag[0] + 10 * math.sin(f * 0.7), lag[1] + 8 * math.cos(f * 0.9)
        od.ellipse((gx - 26, gy - 26, gx + 26, gy + 26), outline=(0, 230, 255, 230), width=6)
        od.ellipse((gx - 6, gy - 6, gx + 6, gy + 6), fill=(0, 230, 255, 255))
        im.paste(ov, (0, 0), ov)
        im.save(os.path.join(d, f"{f:04d}.png"))
        gaze.append([lag[0] / W, lag[1] / H])
    json.dump(gaze, open(os.path.join(d, "gaze.json"), "w"))
    print("star", n)


if __name__ == "__main__":
    keyboard()
    star()
