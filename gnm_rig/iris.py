"""Procedural, high-resolution iris + sclera textures laid out for GNM's eye-interior UV tiles (1002 left, 1004 right).

GNM's eye-interior UV is radial around (0.5, 0.5): pupil r < 0.050, iris r < 0.185, sclera 0.191..0.461.
Each eye gets its own map (different random seed, same colour family), as real left and right irises differ.
Layers: radial stroma fibres, collarette ring, crypts, pigment flecks, a dark limbal ring, pupil, and a sclera
with a soft limbus blend, pinkish corners and fine red veins that grow from the edges.
"""
import numpy as np
from PIL import Image

PUPIL, IRIS, SCL0, SCL1 = 0.050, 0.185, 0.191, 0.461

PALETTES = {  # (inner/collarette colour, outer stroma colour, limbal ring) sRGB
    "blue": ((150, 120, 80), (90, 130, 170), (30, 45, 65)),
    "green": ((150, 110, 50), (90, 120, 75), (35, 45, 30)),
    "brown": ((80, 48, 24), (110, 70, 38), (30, 20, 14)),
    "dark_brown": ((55, 32, 18), (75, 46, 26), (22, 15, 11)),
    "very_dark": ((38, 22, 14), (52, 32, 20), (16, 11, 8)),
    "grey_blue": ((140, 130, 110), (120, 135, 150), (45, 50, 58)),
    "hazel": ((120, 80, 35), (110, 110, 70), (35, 32, 22)),
}


def value_noise(shape, scale, rng, octaves=4, wrap_x=True):
    """Tileable-in-x smooth noise on a (h, w) grid, in [0,1]."""
    h, w = shape
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh, gw = max(2, int(scale[0] * 2 ** o)), max(2, int(scale[1] * 2 ** o))
        grid = rng.random((gh + 1, gw + 1))
        if wrap_x:
            grid[:, -1] = grid[:, 0]
        yy = np.linspace(0, gh, h, endpoint=False)
        xx = np.linspace(0, gw, w, endpoint=False)
        y0, x0 = yy.astype(int), xx.astype(int)
        fy, fx = (yy - y0)[:, None], (xx - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = grid[y0][:, x0]
        b = grid[y0][:, x0 + 1]
        c = grid[y0 + 1][:, x0]
        d = grid[y0 + 1][:, x0 + 1]
        out += amp * ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)
        tot += amp
        amp *= 0.5
    return out / tot


def srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def eye_texture(palette, seed, size=2048):
    rng = np.random.default_rng(seed)
    inner, outer, limbal = (srgb_to_lin(c) for c in PALETTES[palette])
    inner = 0.55 * inner + 0.45 * outer          # muted collarette, real irises blend
    yy, xx = np.mgrid[0:size, 0:size]
    u, v = (xx + 0.5) / size, 1 - (yy + 0.5) / size
    dx, dy = u - 0.5, v - 0.5
    r = np.hypot(dx, dy)
    th = (np.arctan2(dy, dx) / (2 * np.pi)) % 1.0
    # polar noise fields (computed on a polar grid, then sampled)
    PH, PW = 512, 2048
    fib = value_noise((PH, PW), (2, 180), rng, 3)          # fine radial fibres (stretched along r)
    fib2 = value_noise((PH, PW), (3, 60), rng, 3)
    fib3 = value_noise((PH, PW), (2, 420), rng, 2)        # very fine trabeculae
    blot = value_noise((PH, PW), (6, 24), rng, 4)
    crypt = value_noise((PH, PW), (3, 120), rng, 3)      # radially stretched lacunae
    ri = np.clip((r / IRIS) * (PH - 1), 0, PH - 1).astype(int)
    ti = np.clip(th * (PW - 1), 0, PW - 1).astype(int)
    Bl = blot[ri, ti]
    warp = value_noise((PH, PW // 8), (5, 6), rng, 3)[ri, ti // 8]
    tw = np.clip(((th + 0.012 * (warp - 0.5)) % 1.0) * (PW - 1), 0, PW - 1).astype(int)
    F, F2, F3, Cr = fib[ri, tw], fib2[ri, tw], fib3[ri, tw], crypt[ri, tw]
    t = np.clip((r - PUPIL) / (IRIS - PUPIL), 0, 1)        # 0 at pupil edge, 1 at limbus
    # base: collarette colour near the pupil -> stroma colour outside, wavy collarette boundary
    coll = 0.33 + 0.05 * (Bl - 0.5) + 0.03 * np.sin(th * 2 * np.pi * 9 + Bl * 4)
    mixc = smoothstep(coll - 0.06, coll + 0.08, t)[..., None]
    col = inner * (1 - mixc) + outer * mixc
    # fibres: bright/dark streaks along the radius
    streak = (F - 0.5) * 1.0 + (F2 - 0.5) * 0.5 + (F3 - 0.5) * 0.9
    col = col * np.clip(1 + streak[..., None] * 0.9, 0.35, 1.8)
    # collarette ridge (lighter ring) and crypts (dark lacunae) in the middle stroma
    ridge = np.exp(-((t - coll) / 0.025) ** 2)
    col = col * (1 + 0.35 * ridge[..., None])
    crypts = smoothstep(0.70, 0.84, Cr) * smoothstep(0.28, 0.42, t) * smoothstep(0.9, 0.7, t)
    col = col * (1 - 0.45 * crypts[..., None])
    # pigment flecks
    fl = rng.random((size, size)) < 0.00025
    flecks = np.array(Image.fromarray((fl * 255).astype(np.uint8)).filter(
        __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(2.5)), dtype=float) / 255
    col = col * (1 - 0.6 * np.clip(flecks * 6, 0, 1)[..., None] * smoothstep(0.2, 0.5, t)[..., None])
    # darker limbal ring and slight darkening to the pupil margin
    lim = smoothstep(0.86, 0.99, t)
    col = col * (1 - lim[..., None] * 0.65) + limbal * (lim[..., None] * 0.4)
    col = col * (0.7 + 0.3 * smoothstep(0.0, 0.08, t))[..., None]
    # pupil
    pup = smoothstep(PUPIL + 0.002, PUPIL - 0.004, r)[..., None]
    col = col * (1 - pup) + np.array([0.004, 0.004, 0.005]) * pup
    # sclera
    scl_base = srgb_to_lin((236, 228, 220))
    s = np.clip((r - SCL0) / (SCL1 - SCL0), 0, 1)
    sn = value_noise((size // 4, size // 4), (8, 8), rng, 4, wrap_x=False)
    sn = np.array(Image.fromarray((sn * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), dtype=float) / 255
    scl = scl_base * (0.94 + 0.08 * sn[..., None])
    # pink/yellow toward the edges (where the conjunctiva meets the lids)
    edge = smoothstep(0.35, 1.0, s)[..., None]
    scl = scl * (1 - edge * 0.25) + srgb_to_lin((214, 160, 150)) * edge * 0.25
    # veins: thin, wiggly red capillaries = contour lines of a noise field, only towards the sclera edge
    vn = value_noise((PH, PW), (7, 40), rng, 4)
    vr = np.clip((r - SCL0) / (SCL1 - SCL0) * (PH - 1), 0, PH - 1).astype(int)
    v = vn[vr, ti]
    vein = smoothstep(0.012, 0.0, np.abs(v - 0.55)) * smoothstep(0.35, 0.95, s)
    vein = vein * (value_noise((PH, PW), (3, 10), rng, 2)[vr, ti] > 0.5)
    scl = scl * (1 - vein[..., None] * 0.22) + srgb_to_lin((190, 60, 55)) * vein[..., None] * 0.18
    # limbus: soft blend from iris to sclera (no hard edge)
    lb = smoothstep(IRIS - 0.004, SCL0 + 0.006, r)[..., None]
    out = col * (1 - lb) + scl * lb
    return (lin_to_srgb(out) * 255 + 0.5).astype(np.uint8)


def save_pair(palette, seed, path_l, path_r, size=2048):
    Image.fromarray(eye_texture(palette, seed, size)).save(path_l)
    Image.fromarray(eye_texture(palette, seed + 1000, size)).save(path_r)


if __name__ == "__main__":
    import os
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    tiles = []
    for i, p in enumerate(PALETTES):
        im = eye_texture(p, 100 + i, 1024)
        Image.fromarray(im).save(os.path.join(out, f"iris_{p}.png"))
        tiles.append(Image.fromarray(im).crop((280, 280, 744, 744)).resize((300, 300)))
    sheet = Image.new("RGB", (300 * len(tiles), 300))
    for i, t in enumerate(tiles):
        sheet.paste(t, (300 * i, 0))
    sheet.save(os.path.join(out, "iris_sheet.png"))
    print("ok")
