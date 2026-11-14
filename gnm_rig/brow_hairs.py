"""Paint individual eyebrow hairs into each set's skin albedo (tile 1001), in GNM UV space.

Hairs are grown on the face in GNM's metric space (x across, y up), following a natural brow flow field: near
the inner end they point up, in the body they run laterally upward, upper hairs tilt down over the lower ones
(herringbone), and the tail runs outward and slightly down. Each hair is a tapered, slightly curved stroke,
mapped into UV through a piecewise-linear interpolant built from GNM's own vertex UVs (no seams cross the brows).

python gnm_rig/brow_hairs.py [--ids T01,...]      (run after textures.py; rewrites <ID>_albedo.1001.png)
"""
import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import LinearNDInterpolator

import gnm_model as G
from presets import PRESETS

SS = 3  # supersampling for anti-aliased hairs


def vertex_uv(m):
    uv, reg = m.quad_uvs_udim()
    vu = np.full((m.nv, 2), np.nan)
    for qi in np.where(reg == "skin")[0]:
        for k, vi in enumerate(m.quads[qi]):
            vu[vi] = uv[qi, k]
    return vu


def centerline(s):
    s = np.clip(s, 0, 1)
    rise = np.sin(np.minimum(s / 0.62, 1.0) * np.pi / 2)
    fall = (np.maximum(s - 0.62, 0) / 0.38) ** 1.3
    return 0.3155 + 0.007 * rise - 0.010 * fall


def halfwidth(s):
    return 0.0056 - 0.0040 * np.clip(s, 0, 1)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def grow(rng, n):
    """Returns a list of hairs, each an (k, 2) array of (lateral x>0, y) points in metres, plus a thickness."""
    hairs = []
    while len(hairs) < n:
        s = rng.uniform(-0.02, 1.02)
        off = rng.uniform(-1, 1)
        dens = smoothstep(-0.03, 0.08, s) * smoothstep(1.03, 0.85, s) * (1 - 0.55 * abs(off) ** 2)
        dens *= 1.0 - 0.35 * smoothstep(0.6, 1.0, s)
        if rng.random() > dens:
            continue
        x0 = 0.008 + 0.052 * s
        y0 = centerline(s) + off * halfwidth(s) * 0.9 - 0.0012     # roots sit low in the band
        theta = np.interp(s, [0.0, 0.12, 0.3, 0.6, 1.0], [82, 65, 28, 14, -12])
        theta += -12 * off + rng.normal(0, 7)
        length = 0.0075 * np.interp(s, [0, 0.2, 0.6, 1.0], [0.6, 0.9, 1.0, 0.7]) * rng.uniform(0.75, 1.2)
        bend = -np.radians(rng.uniform(8, 22))                       # hairs curve laterally/down
        pts, ang = [np.array([x0, y0])], np.radians(theta)
        k = 8
        for i in range(k):
            ang += bend / k
            pts.append(pts[-1] + length / k * np.array([math.cos(ang), math.sin(ang)]))
        hairs.append((np.array(pts), rng.uniform(0.75, 1.25)))
    return hairs


def paint(P, m, vu, interp, size_hint=None):
    d = os.path.join(G.ROOT, "textures", f"{P['id']}_{P['name']}")
    path = os.path.join(d, f"{P['id']}_albedo.1001.png")
    im = Image.open(path).convert("RGB")
    W = im.size[0]
    rng = np.random.default_rng(1000 + int(P["id"][1:]))
    n = int(900 * P["brow_k"])
    col = np.array(P["brow"], dtype=float)
    for sign in (1, -1):
        hairs = grow(rng, n)
        allp = np.concatenate([h[0] for h, _ in zip(hairs, range(len(hairs)))])
        uvp = interp(np.c_[sign * allp[:, 0], allp[:, 1]])
        ok = ~np.isnan(uvp).any(1)
        px = np.c_[uvp[:, 0] * W, (1 - uvp[:, 1]) * W]
        x0, y0 = np.floor(px[ok].min(0)).astype(int) - 8
        x1, y1 = np.ceil(px[ok].max(0)).astype(int) + 8
        crop = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * SS, (y1 - y0) * SS), Image.BICUBIC)
        over = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(over)
        # px per metre (for stroke width): ~ distance of 1 mm in UV pixels at the brow
        a_ = interp([[sign * 0.03, 0.32]])[0]
        b_ = interp([[sign * 0.031, 0.32]])[0]
        ppm = np.linalg.norm((b_ - a_) * W) / 0.001 * SS
        idx = 0
        for pts, th in hairs:
            k = len(pts)
            q = px[idx: idx + k]
            idx += k
            if np.isnan(q).any():
                continue
            q = (q - [x0, y0]) * SS
            shade = rng.uniform(0.7, 1.15)
            c = np.clip(col * shade, 0, 255).astype(int)
            for i in range(k - 1):
                t = i / (k - 1)
                w = max(1, int(round(0.075e-3 * ppm * th * (1 - 0.6 * t))))
                alpha = int(235 * (1 - 0.45 * t))
                dr.line([tuple(q[i]), tuple(q[i + 1])], fill=(*c, alpha), width=w)
        crop = Image.alpha_composite(crop.convert("RGBA"), over).convert("RGB")
        crop = crop.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        im.paste(crop, (x0, y0))
    im.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(p["id"] for p in PRESETS))
    a = ap.parse_args()
    m = G.GNM()
    vu = vertex_uv(m)
    T = m.T
    sel = (m.groups["skin"] > 0.5) & (T[:, 2] > 0.06) & (T[:, 1] > 0.27) & (T[:, 1] < 0.36) & ~np.isnan(vu[:, 0])
    sel &= m.groups["eye_sockets"] < 0.5
    interp = LinearNDInterpolator(T[sel][:, :2], vu[sel])
    for P in PRESETS:
        if P["id"] in a.ids.split(","):
            print("brows ->", paint(P, m, vu, interp))


if __name__ == "__main__":
    main()
