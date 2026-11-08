"""Pixel-only pupil detector (numpy only, so it runs in Blender's Python too)."""
import math

import numpy as np


def _components(mask):
    """4-connected components -> list of (area, cx, cy, pixels)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack, pix = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        p = np.array(pix)
        comps.append((len(p), p[:, 1].mean(), p[:, 0].mean(), p))
    return comps


def detect(rgb, approx, half=45, min_area=150, flat=False):
    """rgb: HxWx3 float array 0..255 (row 0 = top). approx: (x, y) px guess.
    Returns pupil centre (x, y) in px, or None.

    Pupil = the most disc-like, sizeable, dark & unsaturated blob near `approx`
    (eyelashes are dark too but long and thin). Catchlights are excluded by the
    darkness threshold, so for an unbiased centre use catchlight-free renders.
    """
    x0, y0 = int(approx[0]) - half, int(approx[1]) - half
    roi = rgb[y0:y0 + 2 * half, x0:x0 + 2 * half, :3].astype(np.float32)
    lum = roi @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sat = roi.max(2) - roi.min(2)
    # lit renders: pupil is the darkest thing -> tight threshold above p1 keeps
    # it separate from the lashes it may touch.  flat (lights-off calibration)
    # renders: lashes are darker than the grey pupil -> threshold above p5.
    if flat:
        thr = float(np.percentile(lum, 5)) + 15.0
    else:
        # adaptive split between the darkest pixels (pupil) and the iris/sclera
        lo, mid = float(np.percentile(lum, 1)), float(np.percentile(lum, 50))
        thr = min(90.0, lo + 0.28 * (mid - lo))
    dark = lum < thr
    mask = dark & (sat < 30)
    # morphological erosion cuts the thin dark bridges to eyelashes / lid line
    er = max(1, int(round(half / 22)))
    for _ in range(er):
        m2 = mask.copy()
        m2[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                          & mask[1:-1, :-2] & mask[1:-1, 2:])
        m2[0, :] = m2[-1, :] = m2[:, 0] = m2[:, -1] = False
        mask = m2
    min_area = max(20, int(min_area * 0.5))
    comps = [c for c in _components(mask) if c[0] >= min_area]
    if not comps:
        return None

    def disc_score(c):
        area, cx, cy, p = c
        r = np.hypot(p[:, 1] - cx, p[:, 0] - cy).max() + 1
        return area / (math.pi * r * r)

    def score(c):  # disc-like, sizeable, and near the expected position
        d = math.hypot(c[1] - half, c[2] - half) / half
        return disc_score(c) * min(c[0], 600) * (1.0 - 0.6 * min(d, 1.0))

    best = max(comps, key=score)
    # fill catchlight holes / notches: row-span AND column-span fill of the blob
    p = best[3]
    blob = np.zeros_like(mask)
    blob[p[:, 0], p[:, 1]] = True
    rows = np.zeros_like(mask)
    cols = np.zeros_like(mask)
    for y in np.unique(p[:, 0]):
        xs = p[p[:, 0] == y, 1]
        rows[y, xs.min():xs.max() + 1] = True
    for x in np.unique(p[:, 1]):
        ys = p[p[:, 1] == x, 0]
        cols[ys.min():ys.max() + 1, x] = True
    filled = blob | (rows & cols)
    ys, xs = np.nonzero(filled)
    cx, cy = float(xs.mean()), float(ys.mean())
    return (x0 + cx, y0 + cy)


def detect_green(rgb, approx, half=45, min_area=40):
    """Chroma-key pupil (calibration renders use a pure-green pupil)."""
    x0, y0 = int(approx[0]) - half, int(approx[1]) - half
    roi = rgb[y0:y0 + 2 * half, x0:x0 + 2 * half, :3].astype(np.float32)
    g = roi[:, :, 1] - np.maximum(roi[:, :, 0], roi[:, :, 2])
    mask = g > 25
    comps = [c for c in _components(mask) if c[0] >= min_area]
    if not comps:
        return None
    best = max(comps, key=lambda c: c[0])
    p = best[3]
    w = g[p[:, 0], p[:, 1]]
    return (x0 + float((p[:, 1] * w).sum() / w.sum()), y0 + float((p[:, 0] * w).sum() / w.sum()))
