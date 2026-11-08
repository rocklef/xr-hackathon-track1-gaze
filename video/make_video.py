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
SCENES = ("title", "background", "problem", "literature", "solution", "architecture", "setup", "gaze", "wrapper",
          "proof", "accuracy", "compare", "novelty", "future", "outro")
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


def seg_background(t, d):
    img = canvas()
    paste(img, blurred_bg(t, d, os.path.join(ROOT, "renders_eyes", "frame_00.png")), (0, 0), 0.22)
    tag(img, t, "WHY IT MATTERS", 0.1)
    text(img, (80, 110), "Devices are learning to follow your eyes", F(60, "b"), FG, fade_in(t, 0.2))
    cards = [("headset", "XR headsets", "Render sharp detail only where you look (foveated rendering)",
              wt("background", "headset", 0, 1.0)),
             ("car", "Cars", "Notice when a driver's eyes drift away from the road", wt("background", "car", 0, 3.0)),
             ("screen", "Accessibility", "Click, type and talk using only the eyes", wt("background", "screen", 0, 5.0))]
    for k, (ic, ttl, sub, st) in enumerate(cards):
        x, y, cw, ch = 80 + k * 600, 250, 560, 400
        a = fade_in(t, st - 0.2)
        dy = int((1 - a) * 20)
        panel(img, (x, y + dy, x + cw, y + ch + dy), a)
        icon(img, ic, x + cw // 2, y + 120 + dy, 70, ACC, a, 5)
        text(img, (x + cw // 2, y + 225), ttl, F(42, "b"), FG, a, anchor="ma")
        for j, ln in enumerate(wrap(sub, F(28, "l"), cw - 80)):
            text(img, (x + cw // 2, y + 290 + j * 38), ln, F(28, "l"), MUTED, a, anchor="ma")
    t_q = wt("background", "each", 0, d - 3)
    text(img, (W // 2, 720), "All of them need one answer:", F(34, "l"), MUTED, fade_in(t, t_q), anchor="ma")
    text(img, (W // 2, 775), "where exactly is this person looking?", F(58, "b"), ACC, fade_in(t, t_q + 0.5), anchor="ma")
    return img


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


def seg_problem(t, d):
    img = canvas()
    paste(img, blurred_bg(t, d, os.path.join(ROOT, "renders_eyes", "frame_02.png")), (0, 0), 0.25)
    tag(img, t, "THE PROBLEM · IN PLAIN WORDS", 0.1)
    text(img, (140, 200), "Smart devices learn by example.", F(70, "b"), FG,
         fade_in(t, wt("problem", "learn", 0, 0.8) - 0.3))
    text(img, (140, 320), "But collecting real examples is…", F(40, "l"), MUTED,
         fade_in(t, wt("problem", "gathering", 0, 2.0)))
    x = 140
    for word, lab in (("slow", "Slow"), ("costly", "Costly"), ("private", "Private"),
                      ("never", "Never perfectly labelled")):
        x += chip(img, x, 400, lab, fade_in(t, wt("problem", word, 0, 3.0) - 0.1, 0.35)) + 28
    t_f = wt("problem", "fail", 0, d - 3)
    text(img, (140, 580), "So they fail exactly where they've seen the least:", F(46, "s"), FG, fade_in(t, t_f - 0.4))
    text(img, (140, 650), "extreme angles · unusual faces · new headsets", F(46, "b"), ACC, fade_in(t, t_f + 0.4))
    return img


PAPERS_REAL = [("Zhang, Sugano, Fritz & Bulling", "Appearance-Based Gaze Estimation in the Wild (MPIIGaze)", "CVPR 2015"),
               ("Krafka, Khosla, Kellnhofer et al.", "Eye Tracking for Everyone (GazeCapture)", "CVPR 2016"),
               ("Zhang, Park, Beeler et al.",
                "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation",
                "ECCV 2020")]
PAPERS_SYN = [("Wood, Baltrušaitis, Zhang et al.",
               "Rendering of Eyes for Eye-Shape Registration and Gaze Estimation (SynthesEyes)", "ICCV 2015"),
              ("Wood, Baltrušaitis, Morency et al.",
               "Learning an Appearance-Based Gaze Estimator from One Million Synthesised Images (UnityEyes)", "ETRA 2016"),
              ("Shrivastava, Pfister, Tuzel et al.",
               "Learning from Simulated and Unsupervised Images through Adversarial Training", "CVPR 2017"),
              ("Kim, Stengel, Majercik et al.",
               "NVGaze: An Anatomically-Informed Dataset for Low-Latency, Near-Eye Gaze Estimation", "CHI 2019")]


def seg_literature(t, d):
    img = canvas()
    tag(img, t, "EXISTING RESEARCH", 0.1)
    cols = [(60, "people", "Route 1 · Record real people", PAPERS_REAL, wt("literature", "recording", 0, 1.5),
             ("Real appearance", "Slow, private, labels ≈ where people were asked to look")),
            (990, "cube", "Route 2 · Render with graphics", PAPERS_SYN, wt("literature", "drawing", 0, 4.0),
             ("Fast, exact geometry", "Mostly one eye region at a time"))]
    t_tr = wt("literature", "trades", 0, d - 3)
    for x, ic, head, papers, st, (pro, con) in cols:
        a = fade_in(t, st - 0.3)
        icon(img, ic, x + 40, 128, 34, ACC, a, 4)
        text(img, (x + 100, 100), head, F(40, "b"), FG, a)
        y = 190
        for k, (au, ti, venue) in enumerate(papers):
            ak = fade_in(t, st + 0.25 * k)
            lines = wrap(ti, F(24, "l"), 820)[:2]
            hgt = 58 + 32 * len(lines)
            panel(img, (x, y, x + 870, y + hgt), ak, r=10)
            text(img, (x + 22, y + 10), au, F(26, "s"), FG, ak)
            text(img, (x + 848, y + 10), venue, F(26, "b"), ACC, ak, anchor="ra")
            for j, ln in enumerate(lines):
                text(img, (x + 22, y + 50 + j * 32), ln, F(24, "l"), MUTED, ak)
            y += hgt + 14
        text(img, (x, 775), "+  " + pro, F(30, "s"), GREEN, fade_in(t, t_tr - 0.2))
        text(img, (x, 822), "–  " + con, F(30, "s"), BAD, fade_in(t, t_tr + 0.3))
    return img


def seg_solution(t, d):
    img = canvas()
    tag(img, t, "THE GAP  →  OUR SOLUTION", 0.1)
    text(img, (80, 100), "In plain words, nobody offered…", F(30, "s"), ACC, fade_in(t, 0.2))
    text(img, (1010, 100), "…so we built it. Technically:", F(30, "s"), GREEN, fade_in(t, 0.5))
    rows = [("both", "Both eyes truly looking at one point in 3D",
             "Binocular vergence rig: Damped Track onto a 3D fixation target, 35 cm ± 5 cm"),
            ("through", "Seen through a fixed, headset-style camera",
             "Camera locked 40 cm ahead · full K, [R|t] and P saved per frame"),
            ("labels", "Labels you can actually check",
             "Chroma-key pupil calibration (0.37 px) + pixel-based verifier"),
            ("so", "Open and simple to use",
             "Blender + Python API: eyes.move(h, v) · eyes.reset() · eyes.render()")]
    for k, (word, plain, tech) in enumerate(rows):
        st = wt("solution", word, 0, 1.0 + 2 * k)
        y = 170 + k * 185
        a = fade_in(t, st - 0.35)
        panel(img, (80, y, 860, y + 160), a)
        para(img, (110, y + 30), plain, F(36, "s"), 720, FG, a)
        arrow(img, 880, y + 80, 985, y + 80, fade_in(t, st, 0.6))
        b = fade_in(t, st + 0.2)
        panel(img, (1010, y, 1850, y + 160), b, fill=(20, 34, 27))
        para(img, (1040, y + 30), tech, F(30, "s"), 780, GREEN if k == 3 else FG, b, lh=40)
    return img


ARCH = [("build", "Build", ["MakeHuman face", "skin, iris, strand hair", "one bone per eye"]),
        ("find", "Calibrate eyes", ["sphere fit · r = 15.09 mm", "true centre + optical axis", "bone at eye centre"]),
        ("rig", "Fixation rig", ["target 35 cm, ± 5 cm", "Damped Track vergence", "eyelid drivers"]),
        ("locked", "Render", ["camera locked 40 cm", "Cycles · 1024 × 1024", "constant lighting"]),
        ("ground", "Ground truth", ["metadata.json per frame", "pupil 2D / 3D, gaze", "K, [R|t], P"]),
        ("verifier", "Verify", ["pupils found in pixels", "directions + clockwise", "report.json"])]


def seg_architecture(t, d):
    img = canvas()
    tag(img, t, "ARCHITECTURE", 0.1)
    bw, gap, x0, y0, bh = 270, 40, 60, 360, 300
    times = [wt("architecture", k, 0, 0.8 + 2 * i) for i, (k, _, _) in enumerate(ARCH)]
    active = max([i for i, st in enumerate(times) if t >= st - 0.3] or [-1])
    xa0, xa1 = x0 + 2 * (bw + gap), x0 + 5 * (bw + gap) - gap
    aa = fade_in(t, 0.3)
    panel(img, (xa0, 150, xa1, 250), aa, fill=(30, 26, 16), outline=ACC)
    text(img, ((xa0 + xa1) // 2, 162), "wrapper/eyes.py", F(26, "b"), ACC, aa, anchor="ma")
    text(img, ((xa0 + xa1) // 2, 202), "eyes.move(h, v)  ·  eyes.reset()  ·  eyes.render()", F(26, "m"), FG, aa,
         anchor="ma")
    for j in (2, 3, 4):
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
        text(img, (x + 20, y0 + 90), ttl, F(34, "b"), FG, a)
        for j, ln in enumerate(lines):
            text(img, (x + 20, y0 + 150 + j * 40), ln, F(23, "l"), MUTED, a)
        if i:
            arrow(img, x - gap + 4, y0 + bh // 2, x - 6, y0 + bh // 2, fade_in(t, times[i] - 0.3, 0.6), ACC, 4)
    text(img, (W // 2, 740), "python tools/pipeline.py --all   →   build · calibrate · render · verify",
         F(28, "m"), MUTED, fade_in(t, times[-1] + 0.3), anchor="ma")
    return img


def seg_setup(t, d):
    img = canvas()
    n = len(os.listdir(os.path.join(BUILD, "setup")))
    paste(img, load_frame("setup", min(n, 1 + int(t / d * n))), (0, 0))
    tag(img, t, "THE SETUP")
    caps = [(wt("setup", "fixation", 0, 2.0), "Fixation target · 35 cm ahead (moves ±5 cm)"),
            (wt("setup", "render", 0, 6.0), "Render camera locked · 40 cm ahead"),
            (wt("setup", "each", 0, 10.0), "Both lines of sight meet on the target")]
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


def seg_proof(t, d):
    img = canvas()
    half = wt("proof", "all", 1, d * 0.5) - 0.2   # switch on "...and all ten frames"
    if t < half + 0.6:
        a_out = 1 - fade_in(t, half, 0.6)
        tag(img, t, "PROOF · 5-DIRECTION CHECK", 0.1)
        text(img, (60, 80), "Pupils detected from the rendered pixels and compared with the centre frame",
             F(26, "l"), MUTED, fade_in(t, 0.3) * a_out)
        rows = {r["direction"].lower(): r for r in REPORT["directions"]["rows"]}
        for k, n in enumerate(["center", "left", "right", "up", "down", "reset"]):
            x, y = 60 + (k % 3) * 610, 150 + (k // 3) * 400
            a = fade_in(t, 0.4 + 0.4 * k) * a_out
            paste(img, load(os.path.join(ROOT, "checks", f"eyes_{n}.png")).resize((560, 280), Image.LANCZOS), (x, y), a)
            text(img, (x, y + 290), n.upper(), F(34, "b"), FG, a)
            rr = rows.get(n)
            if rr:
                dx = sum(e["dx"] for e in rr["eyes"]) / 2
                dy = sum(e["dy"] for e in rr["eyes"]) / 2
                text(img, (x + 170, y + 296), f"Δ = ({dx:+.1f}, {dy:+.1f}) px", F(26, "m"), MUTED, a)
                stamp(img, x + 420, y + 288, "PASS", fade_in(t, 0.8 + 0.4 * k, 0.25) * a_out)
            else:
                text(img, (x + 170, y + 296), "reference", F(26, "m"), MUTED, a)
    else:
        tt = t - half
        tag(img, tt, "PROOF · CLOCKWISE SEQUENCE", 0.1)
        cx, cy, R = W // 2, H // 2 - 20, 370
        dd = ImageDraw.Draw(img)
        dd.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(60, 64, 76), width=2)
        for k in range(10):
            th = 2 * math.pi * k / 10
            x, y = cx + R * math.sin(th) - 120, cy - R * math.cos(th) - 60
            a = fade_in(tt, 0.3 + 0.2 * k, 0.3)
            paste(img, load(os.path.join(ROOT, "renders_eyes", f"frame_{k:02d}.png")).resize((240, 120), Image.LANCZOS), (x, y), a)
            text(img, (x + 8, y + 4), f"k={k}", F(20, "b"), ACC, a)
        ang = 2 * math.pi * min(1.0, max(0.0, (tt - 0.3) / 2.2))
        dd.line((cx, cy, cx + (R - 90) * math.sin(ang), cy - (R - 90) * math.cos(ang)), fill=ACC, width=4)
        text(img, (cx, cy - 70), "10 / 10", F(76, "b"), GREEN, fade_in(tt, 2.4), anchor="ma")
        text(img, (cx, cy + 20), "frames in clockwise order", F(28, "s"), FG, fade_in(tt, 2.6), anchor="ma")
        text(img, (cx, cy + 60), f"max angle error {SEQ_ERR:.1f}°", F(26, "l"), MUTED, fade_in(tt, 2.8), anchor="ma")
    return img


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


COMPARE_ROWS = [("Cost to add new images", "Weeks of recording", "Minutes", "Minutes · one command"),
                ("Privacy", "Real volunteers' faces", "No real people", "No real people"),
                ("Label accuracy", "≈ where people were asked to look", "Exact geometry",
                 "Exact + pixel-verified (0.37 px)"),
                ("Both eyes on one 3D point", "Yes, naturally", "Mostly one eye region", "Yes · 9.4° vergence at 35 cm"),
                ("Built-in self-check", "—", "—", "Automatic pixel verifier")]


def seg_compare(t, d):
    img = canvas()
    tag(img, t, "VS. THE STATE OF THE ART", 0.1)
    xs, ws = [60, 500, 960, 1400], [430, 450, 430, 460]
    y0, hh, rh = 100, 120, 118
    heads = [("", ""), ("Real-world datasets", "MPIIGaze · GazeCapture · ETH-XGaze"),
             ("Graphics renderers", "SynthesEyes · UnityEyes · NVGaze"), ("Ours", "calibrated Blender eye rig")]
    times = [0.3, wt("compare", "real", 0, 2.0), wt("compare", "graphics", 0, 5.0), wt("compare", "ours", 0, 8.0)]
    a_ours = fade_in(t, times[3] - 0.3)
    panel(img, (xs[3] - 10, y0 - 10, xs[3] + ws[3], y0 + hh + rh * len(COMPARE_ROWS) + 6), a_ours,
          fill=(18, 40, 28), outline=GREEN, r=18)
    for c in range(4):
        a = fade_in(t, times[c] - 0.3)
        if c:
            text(img, (xs[c] + 14, y0 + 16), heads[c][0], F(36, "b"), GREEN if c == 3 else FG, a)
            text(img, (xs[c] + 14, y0 + 68), heads[c][1], F(21, "l"), MUTED, a)
        for r, row in enumerate(COMPARE_ROWS):
            y = y0 + hh + r * rh
            if c == 0:
                ImageDraw.Draw(img).line((60, y, 1860, y), fill=(40, 44, 54), width=1)
                text(img, (xs[0] + 6, y + 36), row[0], F(30, "s"), FG, fade_in(t, 0.3 + 0.12 * r))
                continue
            ar = fade_in(t, times[c] - 0.1 + 0.15 * r)
            col = GREEN if c == 3 else MUTED
            fnt = F(27, "s" if c == 3 else "r")
            lines = wrap(row[c], fnt, ws[c] - 30)
            for j, ln in enumerate(lines[:2]):
                text(img, (xs[c] + 14, y + (36 if len(lines) == 1 else 20) + j * 36), ln, fnt, col, ar)
    return img


NOVEL = [("pupils", "Cornea-calibrated pupil labels", "green-pupil pass → 3D entrance-pupil fit, 0.37 px"),
         ("binocular", "True binocular vergence", "both eyes converge on one 3D point, ± 5 cm"),
         ("eyelids", "Gaze-linked eyelids", "lids follow vertical gaze, like real eyes"),
         ("pipeline", "Self-checking pipeline", "an independent pixel verifier proves every frame")]
APPS = [("headsets", "headset", "XR foveated rendering"), ("driver", "car", "Driver monitoring"),
        ("accessibility", "screen", "Eye-controlled accessibility"), ("medical", "medical", "Medical eye screening"),
        ("digital", "avatar", "Digital humans & avatars"), (None, "target", "Eye-tracker testing & R&D")]


def seg_novelty(t, d):
    img = canvas()
    tag(img, t, "NOVELTY & APPLICATIONS", 0.1)
    text(img, (60, 100), "What's new", F(44, "b"), GREEN, fade_in(t, 0.2))
    for k, (w_, ttl, sub) in enumerate(NOVEL):
        a = fade_in(t, wt("novelty", w_, 0, 1 + 1.2 * k) - 0.3)
        y = 190 + k * 150
        panel(img, (60, y, 880, y + 130), a, r=12)
        text(img, (110, y + 30), f"{k + 1}", F(52, "b"), GREEN, a, anchor="ma")
        text(img, (160, y + 22), ttl, F(34, "b"), FG, a)
        text(img, (160, y + 76), sub, F(24, "l"), MUTED, a)
    t_b = wt("novelty", "beyond", 0, d / 2)
    text(img, (960, 100), "Where it's useful", F(44, "b"), ACC, fade_in(t, t_b - 0.6))
    last = t_b
    for k, (w_, ic, lab) in enumerate(APPS):
        st = wt("novelty", w_, 0, last + 0.6) if w_ else last + 0.6
        last = st
        a = fade_in(t, st - 0.3)
        x, y = 960 + (k % 3) * 300, 190 + (k // 3) * 300
        panel(img, (x, y, x + 280, y + 280), a, r=14)
        icon(img, ic, x + 140, y + 105, 52, ACC, a, 4)
        for j, ln in enumerate(wrap(lab, F(27, "s"), 240)):
            text(img, (x + 140, y + 190 + j * 34), ln, F(27, "s"), FG, a, anchor="ma")
    return img


FUTURE = [("infrared", "Visible-light studio camera", "Infrared near-eye headset cameras"),
          ("faces", "One adult face model", "Many faces, skin tones, ages and glasses"),
          ("blinks", "Smooth, steady gaze", "Natural blinks and rapid eye jumps (saccades)"),
          ("testing", "Synthetic images only", "Train on ours → test on real eyes (sim-to-real)")]


def seg_future(t, d):
    img = canvas()
    tag(img, t, "SCOPE & FUTURE WORK", 0.1)
    text(img, (100, 110), "Today", F(40, "b"), MUTED, fade_in(t, 0.2))
    text(img, (920, 110), "Next", F(40, "b"), ACC, fade_in(t, 0.4))
    for k, (w_, now, nxt) in enumerate(FUTURE):
        st = wt("future", w_, 0, 1 + 1.6 * k)
        y = 200 + k * 160
        a = fade_in(t, st - 0.4)
        panel(img, (80, y, 760, y + 130), a, r=12)
        text(img, (110, y + 44), now, F(32, "r"), MUTED, a)
        arrow(img, 780, y + 65, 875, y + 65, fade_in(t, st - 0.1, 0.6))
        b = fade_in(t, st)
        panel(img, (900, y, 1840, y + 130), b, fill=(34, 30, 20), r=12)
        text(img, (930, y + 42), nxt, F(34, "s"), FG, b)
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
    env = np.convolve(np.abs(voice), np.ones(k) / k, mode="same")
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Team Rocklef")
    ap.add_argument("--members", default="Sree Rohith Balaji  ·  Thackshanaramana")
    ap.add_argument("--college", default="SRM Institute of Science and Technology")
    ap.add_argument("--repo", default="github.com/rocklef/xr-hackathon-track1-gaze")
    ap.add_argument("--emails", default="sb2213@srmist.edu.in,tb2138@srmist.edu.in")
    ap.add_argument("--preview", type=int, default=0, help="render every Nth frame only")
    a = ap.parse_args()
    team = (a.team, a.members, a.college)
    n_setup = len(os.listdir(os.path.join(BUILD, "setup")))
    base = [("title", lambda t, d: seg_title(t, d, team), 6.0),
            ("background", seg_background, 4.0),
            ("problem", seg_problem, 4.0),
            ("literature", seg_literature, 4.0),
            ("solution", seg_solution, LEAD + vdur("solution") + 2.2),
            ("architecture", seg_architecture, 4.0),
            ("setup", seg_setup, n_setup / FPS),
            ("gaze", seg_gaze, len(TRACK) / FPS + 0.6),
            ("wrapper", seg_wrapper, 4.0),
            ("proof", seg_proof, 10.0),
            ("accuracy", seg_accuracy, 4.0),
            ("compare", seg_compare, 4.0),
            ("novelty", seg_novelty, 4.0),
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
    total = s + XF
    os.makedirs(OUT, exist_ok=True)
    pi = keys.index("proof")
    events = [starts[pi] + 0.8 + 0.4 * k for k in range(1, 6)]
    wav = os.path.join(OUT, "soundtrack.wav")
    mix(wav, soundtrack(None, total, events), [(starts[i] + LEAD, k) for i, k in enumerate(keys)])
    write_srt(os.path.join(OUT, "demo.srt"), [(starts[i], k) for i, k in enumerate(keys)])
    out = os.path.join(OUT, "demo_preview.mp4" if a.preview else "demo.mp4")
    step = max(1, a.preview)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS / step), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(0, int(total * FPS), step):
        gt = fi / FPS
        frame = None
        for (fn, dur), st in zip(segs, starts):
            if st <= gt < st + dur:
                im = fn(gt - st, dur)
                frame = im if frame is None else Image.blend(frame, im, ease((gt - st) / XF))
        p.stdin.write((frame or canvas()).convert("RGB").tobytes())
        if fi % 300 == 0:
            print(f"  {gt:6.1f}s / {total:.1f}s", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out, f"({total:.1f} s)")


if __name__ == "__main__":
    main()
