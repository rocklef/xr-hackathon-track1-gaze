"""Assemble the demo video (1920x1080, 30 fps, H.264 + AAC) with ffmpeg.

    python video/narration.py                 # voiceover + word timings (edge-tts)
    python video/make_video.py                # -> video/out/demo.mp4 + demo.srt
    python video/make_video.py --preview 5    # every 5th frame, fast check

Inputs: video/build/{setup,gaze_face,gaze_eyes}/####.png (video/render_shots.py),
video/build/voice/*.mp3|json (video/narration.py), renders/, renders_eyes/,
checks/, metadata.json, _calib/ (pipeline outputs). Titles, captions, terminal,
proof cards, subtitles and the music bed are generated here.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "video", "build")
OUT = os.path.join(ROOT, "video", "out")
W, H, FPS = 1920, 1080, 30
XF = 0.6    # crossfade seconds
LEAD = 0.5  # picture before the voice starts in each scene

BG = (13, 15, 19)
FG = (238, 240, 244)
MUTED = (150, 156, 168)
ACC = (255, 190, 80)
GREEN = (80, 220, 130)
FONTS = r"C:\Windows\Fonts"


def F(size, w="r"):
    name = {"r": "segoeui.ttf", "b": "segoeuib.ttf", "l": "segoeuisl.ttf", "s": "seguisb.ttf",
            "m": "consola.ttf", "mb": "consolab.ttf"}[w]
    for p in (os.path.join(FONTS, name), name, "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def fade_in(t, start, dur=0.5):
    return ease((t - start) / dur)


def text(img, xy, s, font, fill=FG, alpha=1.0, anchor="la"):
    if alpha <= 0:
        return
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).text((xy[0], xy[1] + (1 - alpha) * 18), s, font=font,
                            fill=(*fill, int(255 * alpha)), anchor=anchor)
    img.alpha_composite(ov)


def paste(img, im, xy, alpha=1.0):
    if alpha <= 0:
        return
    im = im.convert("RGBA")
    if alpha < 1:
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    img.alpha_composite(im, (int(xy[0]), int(xy[1])))


def canvas():
    return Image.new("RGBA", (W, H), (*BG, 255))


def kenburns(im, t, z0=1.0, z1=1.08, cx=0.5, cy=0.5):
    z = z0 + (z1 - z0) * t
    w, h = im.size
    cw, ch = w / z, h / z
    x0, y0 = (w - cw) * cx, (h - ch) * cy
    return im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))


def tag(img, t, s, start=0.3):
    """Small section tag top-left (bottom of frame is reserved for subtitles)."""
    text(img, (60, 34), s, F(26, "s"), ACC, fade_in(t, start))


_cache = {}


def load(path):
    if path not in _cache:
        _cache[path] = Image.open(path).convert("RGBA")
    return _cache[path]


def load_frame(shot, f):
    return Image.open(os.path.join(BUILD, shot, f"{f:04d}.png")).convert("RGBA")


# ------------------------------------------------------------------ data
META = json.load(open(os.path.join(ROOT, "metadata.json")))
REPORT = json.load(open(os.path.join(ROOT, "checks", "report.json")))
try:
    CAL = json.load(open(os.path.join(ROOT, "_calib", "calibration.json")))
except OSError:
    CAL = {"rms": 0.37}
TRACK = json.load(open(os.path.join(BUILD, "gaze_track.json")))
SEQ_ERR = max(abs(r["binocular_err_deg"]) for r in REPORT["sequence"]["rows"])
SCENES = ("open", "vr", "car", "access", "medical", "title", "problem", "literature", "solution", "eyeball",
          "setup", "gaze", "architecture", "proof", "compare", "future", "outro")
VOICE_DIR = os.path.join(BUILD, "voice")
VOICE = {k: json.load(open(os.path.join(VOICE_DIR, f"{k}.json")))
         for k in SCENES if os.path.exists(os.path.join(VOICE_DIR, f"{k}.json"))}


def vdur(key):
    return VOICE[key]["words"][-1]["t1"] if key in VOICE else 0.0


def wt(key, word, nth=0, default=1.0):
    """Scene-local time at which `word` is spoken (nth occurrence)."""
    hits = [w["t0"] for w in VOICE.get(key, {}).get("words", []) if w["w"].lower().strip(",.") == word.lower()]
    return LEAD + hits[min(nth, len(hits) - 1)] if hits else default


# spoken phrase -> how it is shown in the subtitles
MERGES = [(("eyes", "dot", "move"), "eyes.move(h, v)"), (("eyes", "dot", "reset"), "eyes.reset()"),
          (("with", "h", "and", "v"), ""), (("X", "R"), "XR"), (("two", "D"), "2D"),
          (("thirty-five", "centimetres"), "35 cm"), (("five", "centimetres"), "5 cm"),
          (("forty", "centimetres"), "40 cm"), (("Track", "One"), "Track 01")]


def _punctuate(key, ws):
    """Re-attach the script's punctuation to edge-tts word timings."""
    toks = VOICE.get(key, {}).get("text", "").split()
    ti = 0
    for w in ws:
        for look in range(ti, min(ti + 4, len(toks))):
            core = toks[look].strip(",.:;!?")
            if core.lower() == w["w"].lower():
                w["p"] = toks[look][len(core):] if toks[look].startswith(core) else ""
                ti = look + 1
                break
    return ws


def subtitle_lines(key):
    ws = _punctuate(key, [dict(w) for w in VOICE.get(key, {}).get("words", [])])
    for w in ws:
        w["w"] = w["w"] + w.get("p", "")
    i = 0
    while i < len(ws):
        for pat, rep in MERGES:
            if tuple(w["w"].strip(",.:") for w in ws[i:i + len(pat)]) == pat:
                tail = ws[i + len(pat) - 1]["w"][len(ws[i + len(pat) - 1]["w"].rstrip(",.:")):]
                ws[i:i + len(pat)] = [{"w": (rep + tail) if rep else "", "t0": ws[i]["t0"], "t1": ws[i + len(pat) - 1]["t1"]}]
                break
        i += 1
    ws = [w for w in ws if w["w"]]
    groups, cur = [], []
    for j, w in enumerate(ws):
        if cur and len(" ".join(x["w"] for x in cur + [w])) > 42:
            groups.append(cur)
            cur = []
        cur.append(w)
        gap = ws[j + 1]["t0"] - w["t1"] if j + 1 < len(ws) else 9
        sentence_end = w["w"].endswith((".", "?", "!"))
        if (gap > 0.3 or sentence_end) and len(" ".join(x["w"] for x in cur)) >= 14:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    # merge short stragglers into a neighbour when it still fits on one line
    k = 0
    while k < len(groups):
        txt = " ".join(x["w"] for x in groups[k])
        if len(txt) < 12 and len(groups) > 1:
            if k > 0 and len(" ".join(x["w"] for x in groups[k - 1] + groups[k])) <= 50:
                groups[k - 1] += groups.pop(k)
                continue
            if k + 1 < len(groups) and len(" ".join(x["w"] for x in groups[k] + groups[k + 1])) <= 50:
                groups[k] += groups.pop(k + 1)
                continue
        k += 1
    lines = [(g[0]["t0"], g[-1]["t1"], " ".join(x["w"] for x in g).rstrip(",:;")) for g in groups]
    res = []
    for k, (a, b, t) in enumerate(lines):
        nxt = lines[k + 1][0] if k + 1 < len(lines) else b + 0.6
        res.append((LEAD + a, LEAD + min(nxt, b + 0.6), t))
    return res


SUBS = {k: subtitle_lines(k) for k in VOICE}


def draw_subtitle(img, key, t):
    """White subtitle, no background box (thin dark outline for legibility)."""
    for a, b, s in SUBS.get(key, []):
        if a <= t < b:
            al = max(0.0, min(1.0, (t - a) / 0.12, (b - t) / 0.12))
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(ov).text((W // 2, H - 62), s, font=F(46, "s"), fill=(255, 255, 255, int(255 * al)),
                                    anchor="ms", stroke_width=3, stroke_fill=(0, 0, 0, int(170 * al)))
            img.alpha_composite(ov)
            return


# ------------------------------------------------------------------ scenes
def seg_title(t, d, team):
    img = canvas()
    hero = load(os.path.join(ROOT, "checks", "center.png"))
    hb = kenburns(hero, t / d, 1.0, 1.10, 0.5, 0.45).resize((1000, 1000), Image.LANCZOS)
    paste(img, hb, (W - 1000 - 20, 40), fade_in(t, 0.0, 1.2))
    grad = Image.new("L", (900, 1), 0)
    grad.putdata([int(255 * (1 - i / 899) ** 1.5) for i in range(900)])
    g = Image.new("RGBA", (900, H), (*BG, 255))
    g.putalpha(grad.resize((900, H)))
    img.alpha_composite(g, (W - 1020, 0))
    text(img, (110, 290), "XR HACKATHON 2026  ·  TRACK 01", F(28, "s"), ACC, fade_in(t, 0.3))
    text(img, (110, 350), "Synthetic Eye Movement", F(84, "b"), FG, fade_in(t, 0.6))
    text(img, (110, 450), "& Virtual Camera Rendering", F(84, "b"), FG, fade_in(t, 0.8))
    text(img, (110, 590), "Calibrated 3D eyeball rotation · fixation convergence · eye-midpoint camera",
         F(28, "l"), MUTED, fade_in(t, 1.3))
    text(img, (110, 720), team[0], F(46, "b"), FG, fade_in(t, 1.6))
    text(img, (110, 782), team[1], F(30, "s"), MUTED, fade_in(t, 1.9))
    text(img, (110, 834), team[2], F(26, "s"), MUTED, fade_in(t, 2.2))
    return img


BAD = (255, 122, 110)
PANEL = (24, 27, 34)
LINE = (60, 64, 76)


def wrap(s, font, width):
    words, lines, cur = s.split(), [], ""
    meas = ImageDraw.Draw(Image.new("L", (1, 1)))
    for w in words:
        nxt = (cur + " " + w).strip()
        if cur and meas.textlength(nxt, font=font) > width:
            lines.append(cur)
            cur = w
        else:
            cur = nxt
    if cur:
        lines.append(cur)
    return lines


def para(img, xy, s, font, width, fill=FG, alpha=1.0, lh=None):
    lh = lh or int(font.size * 1.3)
    lines = wrap(s, font, width)
    for k, ln in enumerate(lines):
        text(img, (xy[0], xy[1] + k * lh), ln, font, fill, alpha)
    return len(lines) * lh


def panel(img, box, a, fill=PANEL, outline=None, r=16):
    if a <= 0:
        return
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle(box, r, fill=(*fill, int(235 * a)),
                                         outline=(*outline, int(255 * a)) if outline else None, width=3)
    img.alpha_composite(ov)


def arrow(img, x0, y0, x1, y1, a, col=ACC, w=4):
    if a <= 0:
        return
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    c = (*col, int(255 * min(1.0, a)))
    k = ease(a * 1.5)
    xe, ye = x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    d.line((x0, y0, xe, ye), fill=c, width=w)
    ang = math.atan2(y1 - y0, x1 - x0)
    hs = 14
    d.polygon([(xe, ye), (xe - hs * math.cos(ang - 0.45), ye - hs * math.sin(ang - 0.45)),
               (xe - hs * math.cos(ang + 0.45), ye - hs * math.sin(ang + 0.45))], fill=c)
    img.alpha_composite(ov)


def _eye(d, cx, cy, s, c, w):
    r = s * math.hypot(1, 0.6)
    d.arc((cx - r, cy + 0.6 * s - r, cx + r, cy + 0.6 * s + r), 211, 329, fill=c, width=w)
    d.arc((cx - r, cy - 0.6 * s - r, cx + r, cy - 0.6 * s + r), 31, 149, fill=c, width=w)
    d.ellipse((cx - 0.36 * s, cy - 0.36 * s, cx + 0.36 * s, cy + 0.36 * s), outline=c, width=w)
    d.ellipse((cx - 0.14 * s, cy - 0.14 * s, cx + 0.14 * s, cy + 0.14 * s), fill=c)


def icon(img, kind, cx, cy, s=56, col=FG, a=1.0, w=4):
    """Simple line icons drawn in code (no external assets)."""
    if a <= 0:
        return
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    c = (*col, int(255 * a))
    hole = (*BG, int(255 * a))
    if kind == "headset":
        d.arc((cx - 1.2 * s, cy - 1.0 * s, cx + 1.2 * s, cy + 0.6 * s), 195, 345, fill=c, width=w)
        d.rounded_rectangle((cx - s, cy - 0.45 * s, cx + s, cy + 0.5 * s), int(0.3 * s), outline=c, width=w)
        for sx in (-1, 1):
            d.ellipse((cx + sx * 0.5 * s - 0.28 * s, cy - 0.2 * s, cx + sx * 0.5 * s + 0.28 * s, cy + 0.32 * s),
                      outline=c, width=w)
    elif kind == "car":
        d.line([(cx - 0.62 * s, cy - 0.05 * s), (cx - 0.38 * s, cy - 0.5 * s), (cx + 0.38 * s, cy - 0.5 * s),
                (cx + 0.66 * s, cy - 0.05 * s)], fill=c, width=w, joint="curve")
        d.rounded_rectangle((cx - s, cy - 0.08 * s, cx + s, cy + 0.38 * s), int(0.14 * s), outline=c, width=w)
        for sx in (-1, 1):
            x = cx + sx * 0.55 * s
            d.ellipse((x - 0.2 * s, cy + 0.22 * s, x + 0.2 * s, cy + 0.62 * s), fill=hole, outline=c, width=w)
    elif kind == "eye":
        _eye(d, cx, cy, s, c, w)
    elif kind == "screen":
        d.rounded_rectangle((cx - s, cy - 0.68 * s, cx + s, cy + 0.42 * s), 8, outline=c, width=w)
        d.line((cx, cy + 0.42 * s, cx, cy + 0.72 * s), fill=c, width=w)
        d.line((cx - 0.4 * s, cy + 0.74 * s, cx + 0.4 * s, cy + 0.74 * s), fill=c, width=w)
        _eye(d, cx, cy - 0.13 * s, 0.5 * s, c, max(2, w - 1))
    elif kind == "medical":
        d.ellipse((cx - 0.82 * s, cy - 0.82 * s, cx + 0.82 * s, cy + 0.82 * s), outline=c, width=w)
        d.rectangle((cx - 0.14 * s, cy - 0.46 * s, cx + 0.14 * s, cy + 0.46 * s), fill=c)
        d.rectangle((cx - 0.46 * s, cy - 0.14 * s, cx + 0.46 * s, cy + 0.14 * s), fill=c)
    elif kind == "avatar":
        d.ellipse((cx - 0.36 * s, cy - 0.8 * s, cx + 0.36 * s, cy - 0.08 * s), outline=c, width=w)
        d.arc((cx - 0.78 * s, cy + 0.02 * s, cx + 0.78 * s, cy + 1.4 * s), 180, 360, fill=c, width=w)
    elif kind == "people":
        for dx, sc in ((-0.62, 0.7), (0.62, 0.7), (0, 0.9)):
            x, ss = cx + dx * s, sc * s
            d.ellipse((x - 0.32 * ss, cy - 0.7 * ss, x + 0.32 * ss, cy - 0.06 * ss), fill=hole, outline=c, width=w)
            d.arc((x - 0.7 * ss, cy + 0.02 * ss, x + 0.7 * ss, cy + 1.3 * ss), 180, 360, fill=c, width=w)
    elif kind == "cube":
        p = [(cx, cy - 0.85 * s), (cx + 0.8 * s, cy - 0.42 * s), (cx + 0.8 * s, cy + 0.48 * s), (cx, cy + 0.9 * s),
             (cx - 0.8 * s, cy + 0.48 * s), (cx - 0.8 * s, cy - 0.42 * s)]
        d.line(p + [p[0]], fill=c, width=w, joint="curve")
        d.line([p[5], (cx, cy + 0.02 * s), p[1]], fill=c, width=w, joint="curve")
        d.line([(cx, cy + 0.02 * s), p[3]], fill=c, width=w)
    elif kind == "target":
        for r in (0.82, 0.5):
            d.ellipse((cx - r * s, cy - r * s, cx + r * s, cy + r * s), outline=c, width=w)
        d.ellipse((cx - 0.16 * s, cy - 0.16 * s, cx + 0.16 * s, cy + 0.16 * s), fill=c)
        for a0, a1 in (((cx - s, cy), (cx - 0.6 * s, cy)), ((cx + 0.6 * s, cy), (cx + s, cy)),
                       ((cx, cy - s), (cx, cy - 0.6 * s)), ((cx, cy + 0.6 * s), (cx, cy + s))):
            d.line((a0, a1), fill=c, width=w)
    img.alpha_composite(ov)


def blurred_bg(t, d, path, blur=8):
    bg = load(path)
    return kenburns(bg, t / d, 1.05, 1.15).resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(blur))


def chip(img, x, y, s, a, col=BAD, size=36):
    f = F(size, "s")
    w = int(ImageDraw.Draw(img).textlength(s, font=f)) + 56
    if a <= 0:
        return w
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(ov)
    yy = y + (1 - a) * 16
    dd.rounded_rectangle((x, yy, x + w, yy + size + 34), (size + 34) // 2, fill=(*col, int(40 * a)),
                         outline=(*col, int(255 * a)), width=3)
    dd.text((x + 28, yy + 12), s, font=f, fill=(*col, int(255 * a)))
    img.alpha_composite(ov)
    return w


ARCH = [("build", "Build a face", ["real skin, real eyes"]),
        ("point", "Point the eyes", ["eyes.move(h, v)"]),
        ("picture", "Take the picture", ["fixed camera"]),
        ("save", "Save the answer", ["exact gaze + pupils"]),
        ("double", "Double-check", ["found in the pixels"])]


def seg_architecture(t, d):
    img = canvas()
    tag(img, t, "ONE SCRIPT DOES IT ALL", 0.1)
    bw, gap, x0, y0, bh = 320, 50, 60, 360, 260
    times = [wt("architecture", k, 0, 0.8 + 1.1 * i) for i, (k, _, _) in enumerate(ARCH)]
    active = max([i for i, st in enumerate(times) if t >= st - 0.3] or [-1])
    xa0, xa1 = x0 + 1 * (bw + gap), x0 + 4 * (bw + gap) - gap
    aa = fade_in(t, 0.3)
    panel(img, (xa0, 150, xa1, 250), aa, fill=(30, 26, 16), outline=ACC)
    text(img, ((xa0 + xa1) // 2, 162), "wrapper/eyes.py", F(26, "b"), ACC, aa, anchor="ma")
    text(img, ((xa0 + xa1) // 2, 202), "eyes.move(h, v)  ·  eyes.reset()  ·  eyes.render()", F(26, "m"), FG, aa,
         anchor="ma")
    for j in (1, 2, 3):
        cx = x0 + j * (bw + gap) + bw // 2
        arrow(img, cx, 254, cx, y0 - 10, aa, ACC, 3)
    dd = ImageDraw.Draw(img)
    for i, (k, ttl, lines) in enumerate(ARCH):
        x = x0 + i * (bw + gap)
        a = fade_in(t, times[i] - 0.3)
        on = i == active
        panel(img, (x, y0, x + bw, y0 + bh), a, outline=ACC if on else LINE)
        if a > 0:
            c = ACC if on else LINE
            dd.ellipse((x + 20, y0 + 22, x + 64, y0 + 66), fill=(*c, int(255 * a)))
        text(img, (x + 42, y0 + 26), str(i + 1), F(26, "b"), BG if on else FG, a, anchor="ma")
        text(img, (x + 20, y0 + 90), ttl, F(38, "b"), FG, a)
        for j, ln in enumerate(lines):
            text(img, (x + 20, y0 + 160 + j * 40), ln, F(27, "l"), MUTED, a)
        if i:
            arrow(img, x - gap + 4, y0 + bh // 2, x - 6, y0 + bh // 2, fade_in(t, times[i] - 0.3, 0.6), ACC, 4)
    text(img, (W // 2, 740), "python tools/pipeline.py --all",
         F(28, "m"), MUTED, fade_in(t, times[-1] + 0.3), anchor="ma")
    return img


def seg_setup(t, d):
    img = canvas()
    n = len(os.listdir(os.path.join(BUILD, "setup")))
    paste(img, load_frame("setup", min(n, 1 + int(t / d * n))), (0, 0))
    tag(img, t, "THE SETUP")
    caps = [(wt("setup", "target", 0, 2.0), "Fixation target · 35 cm ahead (moves ±5 cm)"),
            (wt("setup", "camera", 0, 4.5), "Render camera locked · 40 cm ahead"),
            (wt("setup", "picture", 0, 4.0), "Both lines of sight meet on the target")]
    for k, (st, s) in enumerate(caps):
        end = caps[k + 1][0] if k + 1 < len(caps) else d + 1
        a = fade_in(t, st) * (1 - fade_in(t, end - 0.35, 0.35))
        text(img, (W - 70, 34), s, F(36, "s"), FG, a, anchor="ra")
    return img


def gaze_label(i):
    h, v = TRACK[i]
    if i < 24:
        return "centre"
    if i < 24 + 150:
        return "clockwise sweep"
    for (th, tv), nm in (((-1, 0), "LEFT"), ((1, 0), "RIGHT"), ((0, 1), "UP"), ((0, -1), "DOWN")):
        if abs(h - th) < 0.35 and abs(v - tv) < 0.35:
            return nm
    return "→ centre" if abs(h) + abs(v) > 0.05 else "centre"


def seg_gaze(t, d):
    img = canvas()
    n = len(TRACK)
    i = min(n - 1, int(t / d * n))
    paste(img, load_frame("gaze_face", i + 1).resize((880, 880), Image.LANCZOS), (60, 90))
    paste(img, load_frame("gaze_eyes", i + 1).resize((900, 450), Image.LANCZOS), (W - 960, 90))
    dd = ImageDraw.Draw(img)
    dd.rectangle((60, 90, 940, 970), outline=(60, 64, 76), width=2)
    dd.rectangle((W - 960, 90, W - 60, 540), outline=(60, 64, 76), width=2)
    tag(img, t, "EYES IN MOTION")
    text(img, (78, 100), "CAM_Gaze · 40 cm · 1024²", F(22, "s"), MUTED)
    text(img, (W - 942, 100), "CAM_GazeEyes · same position, 140 mm", F(22, "s"), MUTED)
    cx, cy, r = W - 510, 745, 140
    dd.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(80, 84, 96), width=2)
    dd.line((cx - r, cy, cx + r, cy), fill=(50, 54, 64), width=1)
    dd.line((cx, cy - r, cx, cy + r), fill=(50, 54, 64), width=1)
    h, v = TRACK[i]
    trail = TRACK[max(0, i - 25):i + 1]
    for j, (th, tv) in enumerate(trail):
        a = j / max(1, len(trail))
        x, y = cx + th * r, cy - tv * r
        dd.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(*ACC, int(160 * a)))
    x, y = cx + h * r, cy - v * r
    dd.ellipse((x - 11, y - 11, x + 11, y + 11), fill=ACC)
    text(img, (cx, cy - r - 40), "fixation target (±5 cm)", F(22, "s"), MUTED, anchor="ma")
    text(img, (cx + r + 30, cy - 18), f"eyes.move(h={h:+.2f},", F(26, "m"), FG)
    text(img, (cx + r + 30, cy + 14), f"          v={v:+.2f})", F(26, "m"), FG)
    lab = gaze_label(i)
    text(img, (W - 960, 570), lab.upper(), F(46, "b"), GREEN if lab.isupper() else ACC)
    return img


def term_lines():
    L = [("$ blender -b scene.blend -P wrapper/eyes.py -- --sequence", "cmd"), ("[eyes] device: OPTIX", "dim")]
    for r in META:
        L.append((f"Saved: 'renders/frame_{r['frame']:02d}.png'", "dim"))
        pl, pr = r["pupil_center_2D_L"], r["pupil_center_2D_R"]
        L.append((f"[eyes] frame {r['frame']} theta={r['theta']:.4f} h={r['input']['h']:+.3f} "
                  f"v={r['input']['v']:+.3f} pL=({pl[0]:.1f},{pl[1]:.1f}) pR=({pr[0]:.1f},{pr[1]:.1f})", "out"))
    L.append(("wrote metadata.json (10 records)", "ok"))
    return L


TERM = term_lines()


def seg_wrapper(t, d):
    img = canvas()
    tag(img, t, "THE WRAPPER IN ACTION", 0.1)
    text(img, (60, 80), "One command renders the 10 clockwise frames (θk = 2πk/10, h = sin θk, v = cos θk) "
         "and writes per-frame ground truth", F(26, "l"), MUTED, fade_in(t, 0.3))
    tx, ty, tw, th = 60, 140, 1080, 540
    img.alpha_composite(Image.new("RGBA", (tw, th), (6, 8, 10, 245)), (tx, ty))
    dd = ImageDraw.Draw(img)
    dd.rectangle((tx, ty, tx + tw, ty + 34), fill=(30, 33, 40))
    for k, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        dd.ellipse((tx + 14 + k * 22, ty + 11, tx + 26 + k * 22, ty + 23), fill=c)
    text(img, (tx + 100, ty + 6), "Terminal — xr-track1", F(18, "s"), MUTED)
    t_meta = wt("wrapper", "metadata", 0, d - 3.0)
    t0, t1 = 0.8, max(2.0, t_meta - 0.4)
    nshow = len(TERM) * ease((t - t0) / (t1 - t0)) if t > t0 else 0
    visible = TERM[:int(nshow) + 1] if nshow > 0 else []
    for j, (s, kind) in enumerate(visible[-15:]):
        col = {"cmd": FG, "dim": (110, 116, 128), "out": (170, 220, 255), "ok": GREEN}[kind]
        if kind == "cmd":
            s = s[:max(1, int(len(s) * min(1, (t - t0) / 1.2 + 0.05)))]
        text(img, (tx + 20, ty + 50 + j * 31), s, F(19, "mb" if kind == "ok" else "m"), col)
    done = sum(1 for _, k in visible if k == "out")
    gx, gy = 1200, 140
    for k in range(10):
        x, y = gx + (k % 4) * 172, gy + (k // 4) * 172
        if k >= 8:
            x = gx + (k - 8 + 1) * 172
        if k < done:
            paste(img, load(os.path.join(ROOT, "renders", f"frame_{k:02d}.png")).resize((160, 160), Image.LANCZOS), (x, y))
            text(img, (x + 6, y + 4), f"k={k}", F(18, "b"), ACC)
        else:
            dd.rectangle((x, y, x + 160, y + 160), outline=(50, 54, 64), width=2)
    if t > t_meta - 0.3:
        a = fade_in(t, t_meta - 0.3)
        r = META[3]
        snippet = ['{ "frame": 3, "theta": %.4f,' % r["theta"],
                   '  "input": {"h": %.3f, "v": %.3f},' % (r["input"]["h"], r["input"]["v"]),
                   '  "eye_rotation_L": [%.2f, %.2f, %.2f],' % tuple(r["eye_rotation_L"]),
                   '  "pupil_center_2D_L": [%.1f, %.1f],' % tuple(r["pupil_center_2D_L"]),
                   '  "pupil_center_2D_R": [%.1f, %.1f],' % tuple(r["pupil_center_2D_R"]),
                   '  "camera_matrix": K (3x3), "camera_extrinsics_RT": [R|t] ... }']
        img.alpha_composite(Image.new("RGBA", (1080, 225), (20, 24, 30, int(240 * a))), (60, 700))
        text(img, (80, 710), "metadata.json · per frame", F(20, "s"), ACC, a)
        for j, s in enumerate(snippet):
            text(img, (80, 742 + j * 28), s, F(20, "m"), FG, a)
    return img


def stamp(img, x, y, s, a, col=GREEN):
    if a <= 0:
        return
    f = F(30, "b")
    w = int(ImageDraw.Draw(img).textlength(s, font=f)) + 34
    ov = Image.new("RGBA", (w, 50), (0, 0, 0, 0))
    dd = ImageDraw.Draw(ov)
    dd.rounded_rectangle((0, 0, w - 1, 49), 10, fill=(*col, int(235 * a)))
    dd.text((17, 6), s, font=f, fill=(10, 20, 14, int(255 * a)))
    img.alpha_composite(ov, (int(x), int(y)))


def counter(t, start, val, fmt, dur=1.2):
    return fmt % (val * ease((t - start) / dur))


def seg_accuracy(t, d):
    img = canvas()
    tag(img, t, "WHAT MAKES IT ACCURATE", 0.1)
    cal = load(os.path.join(ROOT, "_calib", "cal_00.png")).crop((180, 420, 860, 600)).resize((900, 238), Image.LANCZOS)
    t_green = wt("accuracy", "green", 0, 0.5)
    paste(img, cal, (60, 110), fade_in(t, min(0.4, t_green - 0.6)))
    text(img, (60, 365), "Calibration pass: pupil painted green → exact segmentation", F(26, "s"), FG, fade_in(t, t_green))
    text(img, (60, 402), "through the refracting cornea, then a 3D entrance-pupil fit", F(26, "l"), MUTED, fade_in(t, t_green + 0.2))
    cards = [(counter(t, 0.8, CAL.get("rms", 0.37), "%.2f px"), "pupil-label error (RMS)", "chroma-key calibration, 13 gaze poses"),
             ("15.09 mm", "eyeball radius, sphere-fit", "rotation about the true 3D centre"),
             (counter(t, 1.2, 9.4, "%.1f°"), "natural vergence at 35 cm", "Damped Track on the optical axis"),
             ("0.09 px", "camera-model reprojection", "K [R|t] · pupil3D = pupil2D")]
    for k, (big, lab, sub) in enumerate(cards):
        x, y = 1040 + (k % 2) * 430, 110 + (k // 2) * 290
        a = fade_in(t, 0.6 + 0.3 * k)
        img.alpha_composite(Image.new("RGBA", (400, 260), (24, 27, 34, int(255 * a))), (x, y))
        text(img, (x + 24, y + 26), big, F(64, "b"), GREEN if k == 0 else FG, a)
        text(img, (x + 24, y + 124), lab, F(25, "s"), FG, a)
        text(img, (x + 24, y + 164), sub, F(21, "l"), MUTED, a)
    for k, s in enumerate(["Eyelids follow vertical gaze (lid–gaze coupling)",
                           "Constant studio lighting · fixed seed → repeatable renders",
                           "Self-checking: pixel-based verifier + one-command pipeline"]):
        text(img, (60, 520 + k * 62), "•  " + s, F(32, "s"), FG, fade_in(t, 1.8 + 0.4 * k))
    return img


def seg_outro(t, d, team, repo, emails):
    img = canvas()
    cs = load(os.path.join(ROOT, "contact_sheet.png"))
    b = kenburns(cs, t / d, 1.0, 1.06).resize((W, int(W * cs.size[1] / cs.size[0])), Image.LANCZOS)
    paste(img, b.crop((0, 260, W, 260 + H)), (0, 0), 0.12)
    text(img, (W // 2, 80), "CONCLUSION", F(28, "s"), ACC, fade_in(t, 0.1), anchor="ma")
    words = [("calibrated", "Calibrated"), ("verifiable", "Verifiable"), ("script", "Reproducible · one script")]
    f = F(36, "s")
    widths = [int(ImageDraw.Draw(img).textlength(s, font=f)) + 56 for _, s in words]
    x = (W - sum(widths) - 30 * (len(words) - 1)) // 2
    for (w_, s), wd in zip(words, widths):
        chip(img, x, 140, s, fade_in(t, wt("outro", w_, 0, 0.8) - 0.2, 0.35), GREEN)
        x += wd + 30
    text(img, (W // 2, 262), "Track 01 · Synthetic Eye Movement & Virtual Camera Rendering", F(50, "b"), FG,
         fade_in(t, 0.6), anchor="ma")
    t_c = wt("outro", "code", 0, 3.0)
    text(img, (W // 2, 385), "For the code or further development, contact us:", F(32, "l"), FG,
         fade_in(t, t_c - 0.3), anchor="ma")
    text(img, (W // 2, 440), "    ·    ".join(emails), F(42, "mb"), ACC, fade_in(t, t_c), anchor="ma")
    text(img, (W // 2, 515), repo, F(28, "m"), MUTED, fade_in(t, t_c + 0.3), anchor="ma")
    t_th = wt("outro", "thank", 0, d - 2.5)
    text(img, (W // 2, 615), team[0], F(52, "b"), FG, fade_in(t, t_th - 1.0), anchor="ma")
    text(img, (W // 2, 685), team[1], F(30, "s"), MUTED, fade_in(t, t_th - 0.8), anchor="ma")
    text(img, (W // 2, 733), team[2], F(28, "l"), MUTED, fade_in(t, t_th - 0.6), anchor="ma")
    text(img, (W // 2, 805), "Thank you", F(40, "s"), FG, fade_in(t, t_th), anchor="ma")
    return img


# ------------------------------------------------------------------ visual-first scenes (v3)
CYAN_HUD = (0, 229, 255)


def frames_of(shot, t, d=None, speed=1.0):
    n = len(os.listdir(os.path.join(BUILD, shot)))
    if d is not None and speed == "fit":
        i = min(n, 1 + int(t / d * n))
    else:
        i = min(n, 1 + int(t * FPS * speed))
    return load_frame(shot, i)


def callout(img, t, st, x, y, title, sub, col):
    a = fade_in(t, st, 0.35)
    if a <= 0:
        return
    dd = ImageDraw.Draw(img)
    w = int(220 * ease(a * 1.4))
    dd.line((x, y, x + w, y), fill=(*col, int(255 * a)), width=3)
    dd.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(*col, int(255 * a)), width=3)
    text(img, (x + 20, y + 14), title, F(34, "b"), col, a)
    text(img, (x + 20, y + 60), sub, F(24, "l"), FG, a)


HUD_P = json.load(open(os.path.join(BUILD, "hud_pupils.json"))) if os.path.exists(os.path.join(BUILD, "hud_pupils.json")) else {}


def brackets(img, cx, cy, size, a, col=CYAN_HUD, w=3):
    if a <= 0:
        return
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(ov)
    c = (*col, int(255 * a))
    s, L = size, size * 0.45
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * s, cy + sy * s
            dd.line((x, y, x - sx * L, y), fill=c, width=w)
            dd.line((x, y, x, y - sy * L), fill=c, width=w)
    dd.line((cx - 5, cy, cx + 5, cy), fill=c, width=2)
    dd.line((cx, cy - 5, cx, cy + 5), fill=c, width=2)
    img.alpha_composite(ov)


def rolling(val, t, st, seed):
    """Monospace number that scrambles for ~5 frames then settles on the true value."""
    if t < st:
        return ""
    if t < st + 0.18:
        r = (int(t * FPS) * 9301 + seed * 49297) % 233280 / 233280.0
        return "%6.1f" % (val * (0.6 + 0.8 * r))
    return "%6.1f" % val


def lock_on(img, t, key, x, y, w, h, st, label=None):
    """Close-up image at (x, y, w, h) with both pupils locked from st."""
    im = load(os.path.join(ROOT, *key)) if isinstance(key, tuple) else key
    a = fade_in(t, st - 0.5, 0.4)
    paste(img, im.resize((w, h), Image.LANCZOS), (x, y), a)
    name = os.path.splitext(key[-1])[0] if isinstance(key, tuple) else ""
    pts = HUD_P.get(name, [])
    sx, sy = w / 1024, h / 512
    for j, (px, py) in enumerate(pts):
        k = ease((t - st - 0.12 * j) / 0.28)
        size = (1.0 + 0.7 * (1 - k)) * 34 * (w / 560)
        col = GREEN if k >= 1 else CYAN_HUD
        brackets(img, x + px * sx, y + py * sy, size, a * (0.3 + 0.7 * k) if t >= st else 0, col)
        if w >= 400 and t >= st:
            text(img, (int(x + px * sx) - 46, int(y + py * sy + size + 6)),
                 rolling(px, t, st + 0.12 * j, j + 3) + "," + rolling(py, t, st + 0.12 * j, j + 7).strip(),
                 F(17, "m"), col, a)
    if label:
        text(img, (x, y + h + 10), label, F(30, "b"), FG, a)
        if t >= st + 0.45:
            stamp(img, x + w - 150, y + h + 6, "LOCKED", fade_in(t, st + 0.45, 0.2))


def seg_proof(t, d):
    img = canvas()
    half = wt("proof", "all", 1, d * 0.5) - 0.3
    dd = ImageDraw.Draw(img)
    for gx in range(0, W, 80):           # faint HUD grid
        dd.line((gx, 0, gx, H), fill=(18, 26, 34))
    for gy in range(0, H, 80):
        dd.line((0, gy, W, gy), fill=(18, 26, 34))
    if t < half + 0.5:
        a_out = 1 - fade_in(t, half, 0.5)
        tag(img, t, "INDEPENDENT CHECK · PUPILS FOUND IN THE PIXELS", 0.1)
        t5 = wt("proof", "five", 0, 3.5) - 1.2
        for k, n in enumerate(["center", "left", "right", "up", "down", "reset"]):
            x, y = 60 + (k % 3) * 610, 110 + (k // 3) * 400
            sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            lock_on(sub, t, ("checks", f"eyes_{n}.png"), x, y, 560, 280, 0.6 + max(0, t5 - 0.6) * k / 5 + 0.25 * k, n.upper())
            if a_out < 1:
                sub.putalpha(sub.getchannel("A").point(lambda v: int(v * a_out)))
            img.alpha_composite(sub)
    else:
        tt = t - half
        tag(img, tt, "CLOCKWISE SEQUENCE · 10 FRAMES", 0.1)
        cx, cy, R = W // 2, H // 2 - 30, 360
        dd.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(0, 120, 140), width=2)
        ang = 2 * math.pi * min(1.0, max(0.0, (tt - 0.3) / 2.4))
        for k in range(10):
            th = 2 * math.pi * k / 10
            x, y = cx + R * math.sin(th) - 120, cy - R * math.cos(th) - 60
            st = 0.3 + 2.4 * k / 10
            lock_on(img, tt, ("renders_eyes", f"frame_{k:02d}.png"), x, y, 240, 120, st)
            text(img, (x + 6, y + 2), f"k={k}", F(18, "b"), ACC, fade_in(tt, st - 0.3))
        # radar sweep
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        a0 = math.degrees(ang) - 90
        od.pieslice((cx - R + 90, cy - R + 90, cx + R - 90, cy + R - 90), a0 - 40, a0, fill=(*CYAN_HUD, 40))
        od.line((cx, cy, cx + (R - 90) * math.sin(ang), cy - (R - 90) * math.cos(ang)), fill=(*CYAN_HUD, 255), width=3)
        img.alpha_composite(ov)
        text(img, (cx, cy - 70), "10 / 10", F(76, "b"), GREEN, fade_in(tt, 2.7), anchor="ma")
        text(img, (cx, cy + 20), "in clockwise order", F(30, "s"), FG, fade_in(tt, 2.9), anchor="ma")
        text(img, (cx, cy + 62), f"max angle error {SEQ_ERR:.1f}°", F(24, "l"), MUTED, fade_in(tt, 3.1), anchor="ma")
    return img


# ---------- leaner literature
LIT = [("MPIIGaze", "CVPR 2015", "Zhang et al."), ("GazeCapture", "CVPR 2016", "Krafka et al."),
       ("ETH-XGaze", "ECCV 2020", "Zhang et al.")]
LIT_SYN = [("SynthesEyes", "ICCV 2015", "Wood et al."), ("UnityEyes", "ETRA 2016", "Wood et al."),
           ("SimGAN", "CVPR 2017", "Shrivastava et al."), ("NVGaze", "CHI 2019", "Kim et al.")]


def seg_literature(t, d):
    img = canvas()
    tag(img, t, "WHAT RESEARCHERS TRIED", 0.1)
    cols = [(120, "people", "Film real volunteers", LIT, wt("literature", "filming", 0, 1.2), "+ real look", "– slow · private · labels ≈"),
            (1010, "cube", "Draw eyes with graphics", LIT_SYN, wt("literature", "drawing", 0, 3.5), "+ fast · exact", "– mostly one eye")]
    t_g = wt("literature", "gives", 0, d - 2)
    for x, ic, head, papers, st, pro, con in cols:
        a = fade_in(t, st - 0.3)
        # icon multiplies (crowd / render farm) to read as "lots"
        for k in range(5):
            ak = fade_in(t, st - 0.2 + 0.08 * k)
            icon(img, ic, x + 70 + k * 120, 200, 46, ACC if k == 0 else MUTED, ak, 4)
        text(img, (x, 290), head, F(50, "b"), FG, a)
        for k, (nm, venue, au) in enumerate(papers):
            ak = fade_in(t, st + 0.15 * k)
            y = 380 + k * 92
            panel(img, (x, y, x + 760, y + 78), ak, r=12)
            text(img, (x + 24, y + 14), nm, F(36, "b"), FG, ak)
            text(img, (x + 330, y + 22), au, F(26, "l"), MUTED, ak)
            text(img, (x + 736, y + 20), venue, F(28, "b"), ACC, ak, anchor="ra")
        yb = 380 + len(papers) * 92 + 10
        text(img, (x, yb), pro, F(32, "s"), GREEN, fade_in(t, t_g - 0.4))
        text(img, (x + 330, yb), con, F(32, "s"), BAD, fade_in(t, t_g))
    return img


# ---------- scorecard comparison
def mark(img, cx, cy, kind, a, s=30):
    if a <= 0:
        return
    col = {"y": GREEN, "n": BAD, "p": ACC}[kind]
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(ov)
    k = 0.6 + 0.4 * ease(a * 1.3) + 0.15 * math.sin(min(1, a) * math.pi)
    r = s * k
    dd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*col, int(55 * a)), outline=(*col, int(255 * a)), width=4)
    c = (*col, int(255 * a))
    if kind == "y":
        dd.line([(cx - r * 0.45, cy), (cx - r * 0.1, cy + r * 0.38), (cx + r * 0.5, cy - r * 0.38)], fill=c, width=6, joint="curve")
    elif kind == "n":
        dd.line((cx - r * 0.38, cy - r * 0.38, cx + r * 0.38, cy + r * 0.38), fill=c, width=6)
        dd.line((cx - r * 0.38, cy + r * 0.38, cx + r * 0.38, cy - r * 0.38), fill=c, width=6)
    else:
        dd.line((cx - r * 0.45, cy, cx + r * 0.45, cy), fill=c, width=6)
    img.alpha_composite(ov)


# ------------------------------------------------------------------ v4: real people, application first
def fact_chip(img, t, st, lines, src=None, x=60, y=78):
    """Small sourced fact card, top-left (bottom is reserved for subtitles)."""
    a = fade_in(t, st, 0.45)
    if a <= 0:
        return
    f1, f2 = F(30, "s"), F(20, "l")
    w = max(int(ImageDraw.Draw(img).textlength(s, font=f1)) for s in lines) + 60
    h = 26 + 42 * len(lines) + (30 if src else 0)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(ov)
    dd.rounded_rectangle((x, y, x + w, y + h), 16, fill=(8, 10, 14, int(190 * a)))
    dd.rectangle((x, y + 14, x + 6, y + h - 14), fill=(*ACC, int(255 * a)))
    img.alpha_composite(ov)
    for i, s in enumerate(lines):
        text(img, (x + 28, y + 16 + 42 * i), s, f1, FG, a)
    if src:
        text(img, (x + 28, y + h - 36), src, f2, MUTED, a)


def footage(shot, t):
    return frames_of(shot, t)


def seg_open(t, d):
    img = canvas()
    paste(img, footage("app_open", t), (0, 0))
    return img


def seg_vr(t, d):
    img = canvas()
    paste(img, footage("app_vr", t), (0, 0))
    st = wt("vr", "picture", 0, 3.3) - 0.2
    a = fade_in(t, st, 0.5)
    if a > 0:
        # "what she sees": the foveated view (sharp circle that follows her gaze)
        n0, n1 = 100, 176
        k = n0 + int(((t - st) * FPS) % (n1 - n0))
        pip = load_frame("cartoon_hook", k).resize((720, 405), Image.LANCZOS)
        x, y = W - 780, 70 + int((1 - a) * 30)
        sh = Image.new("RGBA", (740, 425), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((0, 0, 739, 424), 22, fill=(0, 0, 0, int(140 * a)))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)), (x - 10, y - 2))
        mask = Image.new("L", (720, 405), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, 719, 404), 18, fill=int(255 * a))
        pip.putalpha(mask)
        img.alpha_composite(pip, (x, y))
        ImageDraw.Draw(img).rounded_rectangle((x, y, x + 720, y + 405), 18, outline=(*ACC, int(255 * a)), width=3)
        text(img, (x + 20, y + 418), "What she sees: sharp only where she looks", F(28, "s"), FG, a)
    fact_chip(img, t, 0.8, ["Eye-tracked headsets: PlayStation VR2, Apple Vision Pro"], None)
    return img


def seg_car(t, d):
    img = canvas()
    paste(img, footage("app_car", t), (0, 0))
    fact_chip(img, t, wt("car", "since", 0, 2.5) - 0.2,
              ["EU law: since July 2026, every new car", "must warn a distracted driver"], "EU General Safety Regulation (ADDW)")
    t_w = 150 / FPS
    if t >= t_w and int((t - t_w) * 3.75) % 2 == 0:
        stamp(img, W - 420, 90, "EYES OFF ROAD", 1.0, col=(255, 70, 60))
    return img


def seg_access(t, d):
    img = canvas()
    paste(img, footage("app_access", t), (0, 0))
    fact_chip(img, t, wt("access", "like", 0, 2.6) - 0.2, ["Eye-gaze communication devices", "help people living with ALS type and speak"],
              "e.g. Tobii Dynavox TD I-Series")
    return img


def seg_medical(t, d):
    img = canvas()
    paste(img, footage("app_medical", t), (0, 0))
    fact_chip(img, t, wt("medical", "autism", 0, 4.5) - 1.2, ["FDA-cleared eye-tracking aid helps diagnose", "autism in children aged 16 to 30 months"],
              "EarliPoint Evaluation, FDA 510(k), 2022")
    return img


def seg_lineup(t, d):
    img = canvas()
    paste(img, footage("app_lineup", t), (0, 0))
    tag(img, t, "OUR SOLUTION · DIGITAL HUMANS", 0.3)
    return img


def eyewall_tiles():
    tiles = []
    for who in ("maya", "arjun", "arthur", "zuri"):
        for k in range(1, 7):
            p = os.path.join(BUILD, "eyewall", who, f"{k:04d}.png")
            if os.path.exists(p):
                tiles.append(load(p))
    return tiles


def seg_problem(t, d):
    """A wall of real eye photos multiplies (each with a green label), then the problems pop up."""
    img = canvas()
    tiles = eyewall_tiles()
    tag(img, t, "THE CATCH", 0.1)
    if not tiles:
        return img
    z = 1.0 - 0.62 * ease((t - 0.3) / max(1.0, wt("problem", "each", 0, 5.0)))
    tw, th = int(420 * z), int(210 * z)
    cols, rows = W // tw + 3, H // th + 3
    t_lab = wt("problem", "labelled", 0, 5.5)
    t_bad = wt("problem", "never", 0, d - 1.5)
    ox, oy = (W - cols * tw) // 2, (H - rows * th) // 2 - 40
    for r in range(rows):
        for c in range(cols):
            idx = (r * 7 + c * 3) % len(tiles)
            a = fade_in(t, 0.05 * ((r * cols + c) % 17), 0.3)
            x, y = ox + c * tw, oy + r * th
            if x > W or y > H or x + tw < 0 or y + th < 0:
                continue
            paste(img, tiles[idx].resize((tw - 6, th - 6), Image.BILINEAR), (x + 3, y + 3), a * 0.92)
            la = fade_in(t, t_lab + 0.01 * ((r * cols + c) % 23), 0.25)
            if la > 0:
                bad = t >= t_bad and (r + c) % 3 == 0
                col = (255, 80, 70) if bad else GREEN
                cy = y + th * 0.47
                brackets(img, x + tw * 0.25, cy, max(8, tw * 0.05), la, col, 2)
                brackets(img, x + tw * 0.75, cy, max(8, tw * 0.05), la, col, 2)
                if bad:
                    text(img, (int(x + tw * 0.5), int(y + th * 0.18)), "?", F(max(18, int(th * 0.35)), "b"), (255, 80, 70), la, anchor="ma")
    t_c = wt("problem", "collecting", 0, 7.5)
    shade = fade_in(t, t_c - 0.3, 0.6)
    if shade > 0:
        img.alpha_composite(Image.new("RGBA", (W, H), (8, 10, 14, int(150 * shade))))
    x = 160
    for word, lab in (("slow", "Slow"), ("expensive", "Expensive"), ("private", "Private"), ("never", "Labels never perfect")):
        st = wt("problem", word, 0, 9.0)
        x += chip(img, x, 440, lab, fade_in(t, st - 0.1, 0.3), BAD, 46) + 30
    return img


def seg_xray(t, d):
    img = canvas()
    paste(img, frames_of("xray", t, d, "fit"), (0, 0))
    tag(img, t, "HOW IT WORKS", 0.2)
    callout(img, t, wt("eyeball", "centre", 0, 1.8) - 0.2, 110, 170, "TRUE CENTRE", "each eye turns about it", CYAN_HUD)
    callout(img, t, wt("eyeball", "sight", 0, 3.5) - 0.2, 110, 320, "LINE OF SIGHT", "where the eye really looks", ACC)
    return img


SCORE = [("Faster", "nyy"), ("Protects privacy", "nyy"), ("Both eyes together", "ypy"), ("Checks itself", "nny")]


def seg_compare(t, d):
    img = canvas()
    tag(img, t, "COMPARED WITH TODAY'S METHODS", 0.1)
    cols = [("Real photos", "MPIIGaze · ETH-XGaze"), ("Graphics tools", "SynthesEyes · UnityEyes"), ("Ours", "digital humans")]
    xs = [880, 1230, 1590]
    y0 = 300
    a3 = fade_in(t, 0.5)
    panel(img, (xs[2] - 170, 140, xs[2] + 170, y0 + 4 * 140 - 20), a3, fill=(18, 40, 28), outline=GREEN, r=18)
    words = ["faster", "privacy", "both", "checks"]
    for c, (nm, sub) in enumerate(cols):
        a = fade_in(t, 0.2 + 0.15 * c)
        text(img, (xs[c], 170), nm, F(38, "b"), GREEN if c == 2 else FG, a, anchor="ma")
        text(img, (xs[c], 222), sub, F(22, "l"), MUTED, a, anchor="ma")
    for r, (crit, vals) in enumerate(SCORE):
        st = wt("compare", words[r], 0, 1.5 + 1.4 * r) - 0.2
        y = y0 + r * 140
        text(img, (110, y + 40), crit, F(50, "b"), FG, fade_in(t, st - 0.2))
        ImageDraw.Draw(img).line((110, y + 125, 1780, y + 125), fill=(34, 38, 48), width=2)
        for c in range(3):
            mark(img, xs[c], y + 62, vals[c], fade_in(t, st + 0.12 * c, 0.3), 40)
    return img


FUT = [("faces", "people", "More faces & skin tones"), ("infrared", "headset", "Infrared headset cameras"),
       ("blinks", "eye", "Natural blinks"), ("medical", "target", "Ready for headsets, cars & clinics")]


def seg_future(t, d):
    img = canvas()
    tag(img, t, "WHAT'S NEXT", 0.1)
    for k, (w_, ic, lab) in enumerate(FUT):
        st = wt("future", w_, 0, 1 + k)
        a = fade_in(t, st - 0.3, 0.4)
        cx = 240 + k * 480
        panel(img, (cx - 200, 260, cx + 200, 680), a, r=20)
        icon(img, ic, cx, 420, 80, ACC, a, 6)
        for j, ln in enumerate(wrap(lab, F(34, "b"), 340)):
            text(img, (cx, 550 + j * 44), ln, F(34, "b"), FG, a, anchor="ma")
    return img


# ------------------------------------------------------------------ audio
def soundtrack(path, seconds, events, sr=48000):
    """Original ambient pad + soft ticks on the PASS stamps."""
    n = int(seconds * sr)
    tt = np.arange(n) / sr
    out = np.zeros(n)
    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66), (164.81, 207.65, 246.94)]
    seg = 8.0
    for i in range(int(seconds / seg) + 1):
        a, b = int(i * seg * sr), min(n, int((i + 1) * seg * sr + 2 * sr))
        if a >= n:
            break
        ts = tt[a:b] - i * seg
        env = np.minimum(1, ts / 2.0) * np.minimum(1, np.maximum(0, (seg + 2 - ts) / 2.0))
        for f in chords[i % len(chords)]:
            for det in (-0.6, 0.6):
                out[a:b] += 0.05 * env * np.sin(2 * np.pi * (f + det) * ts) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * ts))
            out[a:b] += 0.02 * env * np.sin(2 * np.pi * f / 2 * ts)
    for et in events:
        a = int(et * sr)
        m = min(n - a, int(0.25 * sr))
        if m > 0:
            ts = np.arange(m) / sr
            out[a:a + m] += 0.18 * np.exp(-ts * 22) * np.sin(2 * np.pi * 1318.5 * ts)
    out *= np.clip(np.minimum(tt / 1.5, (seconds - tt) / 2.5), 0, 1)
    return np.tanh(out * 1.4) * 0.6


def decode(path, sr=48000):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def mix(path, music, cues, sr=48000):
    """Narration over the music bed; music ducks under speech."""
    voice = np.zeros_like(music)
    for st, key in cues:
        p = os.path.join(VOICE_DIR, f"{key}.mp3")
        if os.path.exists(p):
            v = decode(p, sr)
            a = int(st * sr)
            m = max(0, min(len(v), len(voice) - a))
            voice[a:a + m] += v[:m]
    k = int(0.3 * sr)
    # centred moving average of |voice| (same as np.convolve(..., mode="same") but O(n) via a running sum)
    c = np.concatenate([[0.0], np.cumsum(np.abs(voice))])
    i = np.arange(len(voice))
    env = (c[np.clip(i - k // 2 + k, 0, len(voice))] - c[np.clip(i - k // 2, 0, len(voice))]) / k
    out = music * 0.55 * (1.0 - 0.6 * np.clip(env / 0.03, 0, 1)) + voice
    pcm = (np.clip(out, -0.98, 0.98) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.repeat(pcm[:, None], 2, 1).tobytes())


def write_srt(path, cues):
    def ts(x):
        h, r = divmod(x, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"
    n = 1
    with open(path, "w", encoding="utf-8") as f:
        for st, key in cues:
            for a, b, s in SUBS.get(key, []):
                f.write(f"{n}\n{ts(st + a)} --> {ts(st + b)}\n{s}\n\n")
                n += 1


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Team Rocklef")
    ap.add_argument("--members", default="Sree Rohith Balaji  ·  Thackshanaramana")
    ap.add_argument("--college", default="SRM Institute of Science and Technology")
    ap.add_argument("--repo", default="github.com/rocklef/xr-hackathon-track1-gaze")
    ap.add_argument("--emails", default="sb2213@srmist.edu.in,tb2138@srmist.edu.in")
    ap.add_argument("--preview", type=int, default=0, help="render every Nth frame only")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 4)),
                    help="parallel frame painters (each paints whole frames on its own CPU core)")
    return ap.parse_args(argv)


def build(a):
    """Segment list (painter, duration), start times, scene keys and total length."""
    team = (a.team, a.members, a.college)
    base = [("open", seg_open, 4.0),
            ("vr", seg_vr, 4.0),
            ("car", seg_car, 4.0),
            ("access", seg_access, 4.0),
            ("medical", seg_medical, 4.0),
            ("title", lambda t, d: seg_title(t, d, team), 4.5),
            ("problem", seg_problem, 4.0),
            ("literature", seg_literature, 4.0),
            ("solution", seg_lineup, 4.0),
            ("eyeball", seg_xray, 4.0),
            ("setup", seg_setup, 7.0),
            ("gaze", seg_gaze, 8.0),
            ("architecture", seg_architecture, 4.0),
            ("proof", seg_proof, 10.0),
            ("compare", seg_compare, 4.0),
            ("future", seg_future, 4.0),
            ("outro", lambda t, d: seg_outro(t, d, team, a.repo, a.emails.split(",")), 7.0)]

    def with_subs(key, fn):
        def g(t, d):
            im = fn(t, d)
            draw_subtitle(im, key, t)
            return im
        return g

    segs = [(with_subs(k, fn), max(dmin, LEAD + vdur(k) + 0.8)) for k, fn, dmin in base]
    keys = [k for k, _, _ in base]
    starts, s = [], 0.0
    for _, dur in segs:
        starts.append(s)
        s += dur - XF
    return segs, starts, keys, s + XF


_W = {}


def _init_worker(argv):
    _W["segs"], _W["starts"], _, _ = build(parse_args(argv))


def paint(fi, segs=None, starts=None):
    """Paint global frame fi and return its raw RGB bytes."""
    segs, starts = segs or _W["segs"], starts or _W["starts"]
    gt = fi / FPS
    frame = None
    for (fn, dur), st in zip(segs, starts):
        if st <= gt < st + dur:
            im = fn(gt - st, dur)
            frame = im if frame is None else Image.blend(frame, im, ease((gt - st) / XF))
    return (frame or canvas()).convert("RGB").tobytes()


def video_codec():
    """H.264 on the NVIDIA card (NVENC) when the driver supports it, else fast multi-threaded x264."""
    nv = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-profile:v", "high"]
    probe = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                            *nv, "-f", "null", "-"], capture_output=True)
    if probe.returncode == 0:
        print("encoder: h264_nvenc (GPU)", flush=True)
        return nv
    print("encoder: libx264 (GPU encoder needs a newer NVIDIA driver)", flush=True)
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]


def main():
    import multiprocessing as mp
    a = parse_args()
    segs, starts, keys, total = build(a)
    os.makedirs(OUT, exist_ok=True)
    pi = keys.index("proof")
    events = [starts[pi] + 0.8 + 0.4 * k for k in range(1, 6)]
    wav = os.path.join(OUT, "soundtrack.wav")
    mix(wav, soundtrack(None, total, events), [(starts[i] + LEAD, k) for i, k in enumerate(keys)])
    write_srt(os.path.join(OUT, "demo.srt"), [(starts[i], k) for i, k in enumerate(keys)])
    out = os.path.join(OUT, "demo_preview.mp4" if a.preview else "demo.mp4")
    step = max(1, a.preview)
    # H.264 on the NVIDIA card (NVENC); the frames themselves are painted on a pool of CPU cores
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS / step), "-i", "-", "-i", wav, *video_codec(),
           "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    idx = range(0, int(total * FPS), step)
    print(f"painting {len(idx)} frames on {a.workers} cores", flush=True)
    if a.workers > 1:
        with mp.Pool(a.workers, initializer=_init_worker, initargs=(sys.argv[1:],)) as pool:
            for n, buf in enumerate(pool.imap(paint, idx, chunksize=4)):
                p.stdin.write(buf)
                if n % 300 == 0:
                    print(f"  {idx[n] / FPS:6.1f}s / {total:.1f}s", flush=True)
    else:
        for n, fi in enumerate(idx):
            p.stdin.write(paint(fi, segs, starts))
            if n % 300 == 0:
                print(f"  {fi / FPS:6.1f}s / {total:.1f}s", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out, f"({total:.1f} s)")


if __name__ == "__main__":
    main()
