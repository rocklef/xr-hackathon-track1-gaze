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
LEAD = 0.7  # picture before the voice starts in each scene

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
SCENES = ("title", "problem", "setup", "gaze", "wrapper", "proof", "accuracy", "outro")
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


def seg_problem(t, d):
    img = canvas()
    bg = load(os.path.join(ROOT, "renders_eyes", "frame_02.png"))
    b = kenburns(bg, t / d, 1.05, 1.15).resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(6))
    paste(img, b, (0, 0), 0.35)
    text(img, (140, 220), "THE PROBLEM", F(30, "s"), ACC, fade_in(t, 0.2))
    lines = [("XR gaze trackers learn from huge sets of labelled eye images.", wt("problem", "eye", 0, 0.7)),
             ("Capturing real eyes at extreme angles is slow, costly", wt("problem", "capturing", 0, 2.4)),
             ("and restricted by privacy rules.", wt("problem", "restricted", 0, 2.6))]
    for i, (s, st) in enumerate(lines):
        text(img, (140, 290 + i * 70 + (40 if i else 0)), s, F(50, "s"), FG, fade_in(t, st))
    t_ans = wt("problem", "so", 0, 5.0)
    text(img, (140, 610), "OUR ANSWER", F(30, "s"), GREEN, fade_in(t, t_ans - 0.3))
    text(img, (140, 670), "A calibrated synthetic eye rig in Blender:", F(50, "s"), FG, fade_in(t, t_ans))
    text(img, (140, 740), "every image comes with perfect 2D + 3D gaze labels.", F(50, "s"), FG,
         fade_in(t, wt("problem", "every", 0, 6.2)))
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


def seg_outro(t, d, team, repo):
    img = canvas()
    cs = load(os.path.join(ROOT, "contact_sheet.png"))
    b = kenburns(cs, t / d, 1.0, 1.06).resize((W, int(W * cs.size[1] / cs.size[0])), Image.LANCZOS)
    paste(img, b.crop((0, 0, W, H)), (0, 0), 0.22)
    text(img, (W // 2, 300), "Track 01 · Synthetic Eye Movement", F(64, "b"), FG, fade_in(t, 0.2), anchor="ma")
    text(img, (W // 2, 400), "scene.blend · eyes.py · 10 renders · metadata.json · verifier",
         F(30, "l"), MUTED, fade_in(t, 0.5), anchor="ma")
    text(img, (W // 2, 520), repo, F(34, "m"), ACC, fade_in(t, 0.9), anchor="ma")
    text(img, (W // 2, 630), team[0], F(52, "b"), FG, fade_in(t, 1.2), anchor="ma")
    text(img, (W // 2, 700), team[1], F(30, "s"), MUTED, fade_in(t, 1.4), anchor="ma")
    text(img, (W // 2, 752), team[2], F(28, "l"), MUTED, fade_in(t, 1.6), anchor="ma")
    text(img, (W // 2, 820), "Thank you", F(34, "l"), MUTED, fade_in(t, 1.8), anchor="ma")
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
    ap.add_argument("--preview", type=int, default=0, help="render every Nth frame only")
    a = ap.parse_args()
    team = (a.team, a.members, a.college)
    n_setup = len(os.listdir(os.path.join(BUILD, "setup")))
    base = [("title", lambda t, d: seg_title(t, d, team), 7.0),
            ("problem", seg_problem, 9.5),
            ("setup", seg_setup, n_setup / FPS),
            ("gaze", seg_gaze, len(TRACK) / FPS + 1.0),
            ("wrapper", seg_wrapper, 14.0),
            ("proof", seg_proof, 13.0),
            ("accuracy", seg_accuracy, 10.0),
            ("outro", lambda t, d: seg_outro(t, d, team, a.repo), 6.5)]

    def with_subs(key, fn):
        def g(t, d):
            im = fn(t, d)
            draw_subtitle(im, key, t)
            return im
        return g

    segs = [(with_subs(k, fn), max(dmin, LEAD + vdur(k) + 1.1)) for k, fn, dmin in base]
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
