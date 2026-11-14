"""Semantic eye/brow expressions for GNM, solved inside GNM's own eye-region expression space.

GNM's eye expressions are 100 PCA components per eye (no names). We define each semantic shape by a few
geometric targets on the eyelid margin and brow (e.g. "upper lid margin meets the lower lid") and solve a
ridge-regularised least-squares fit for the 100 coefficients. Because the result is a combination of GNM's
own scan-derived components, the whole lid / brow / skin deforms the way real faces do, not just the targets.

Output: {name: expression vector (383,)} per side, cached in gnm_rig/expressions_cache.npz.
"""
import collections
import os

import numpy as np

import gnm_model as G

CACHE = os.path.join(G.HERE, "expressions_cache.npz")
MM = 0.001


def lid_margins(m):
    """Ordered upper and lower eyelid margins for each eye: {'left': (upper_idx, lower_idx), ...}."""
    reg = m.face_regions()
    quads = m.quads[reg == "skin"]
    sock = m.groups["eye_sockets"] > 0.5
    ring, edges = set(), collections.defaultdict(set)
    for f in quads:
        s = sock[f]
        if s.any() and (~s).any():
            ring.update(int(i) for i in f[~s])
    for f in quads:
        for a, b in zip(f, np.roll(f, -1)):
            a, b = int(a), int(b)
            if a in ring and b in ring:
                edges[a].add(b)
                edges[b].add(a)
    out = {}
    for side, sg in (("left", 1), ("right", -1)):
        pts = [i for i in ring if np.sign(m.T[i, 0]) == sg]
        start = min(pts, key=lambda i: m.T[i, 0])          # one canthus
        loop, prev, cur = [start], None, start
        while True:
            nxt = [n for n in edges[cur] if n != prev and n in pts]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
        loop = np.array(loop)
        a = int(np.argmin(m.T[loop, 0]))
        b = int(np.argmax(m.T[loop, 0]))
        loop = np.roll(loop, -a)
        b = (b - a) % len(loop)
        p1, p2 = loop[:b + 1], np.concatenate([loop[b:], loop[:1]])
        up, lo = (p1, p2) if m.T[p1, 1].mean() > m.T[p2, 1].mean() else (p2, p1)
        out[side] = (up, lo)
    return out


def solve(m, side, targets, lam=2e-7, keep=()):
    """targets: list of (vertex_idx array, (n,3) displacement in GNM space, weight).
    keep: vertex index arrays that should stay put (weight 0.5)."""
    idx = m.expr_index(f"{side}_eye_region")
    B = m.B_ex[idx].astype(np.float64)                # (k, V, 3)
    rows_A, rows_d, rows_w = [], [], []
    for vi, d, w in list(targets) + [(k, np.zeros((len(k), 3)), 0.5) for k in keep]:
        rows_A.append(B[:, vi, :].reshape(len(idx), -1).T)
        rows_d.append(np.asarray(d).reshape(-1))
        rows_w.append(np.full(3 * len(vi), w))
    A, d, w = np.vstack(rows_A), np.concatenate(rows_d), np.concatenate(rows_w)
    Aw = A * w[:, None]
    c = np.linalg.solve(A.T @ Aw + lam * np.eye(len(idx)), Aw.T @ d)
    e = np.zeros(m.B_ex.shape[0])
    e[idx] = c
    return e


def build(m):
    marg = lid_margins(m)
    brows = {s: np.where((m.groups[f"{s}_brow_region"] > 0.5))[0] for s in ("left", "right")}
    shapes = {}
    for side in ("left", "right"):
        up, lo = marg[side]
        pu, pl = m.T[up], m.T[lo]
        order = np.argsort(pl[:, 0])
        lo_y = np.interp(pu[:, 0], pl[order, 0], pl[order, 1])
        lo_z = np.interp(pu[:, 0], pl[order, 0], pl[order, 2])
        # aperture profile: 0 at the corners, 1 in the middle
        span = (pu[:, 0] - pu[:, 0].min()) / max(1e-9, np.ptp(pu[:, 0]))
        bell = np.sin(np.pi * span) ** 1.2

        def lid(du_mm, dl_mm, z_follow=0.35):
            dy_u = du_mm * MM * bell
            dy_l = dl_mm * MM * np.sin(np.pi * (pl[:, 0] - pl[:, 0].min()) / max(1e-9, np.ptp(pl[:, 0])))
            tu = np.stack([0 * dy_u, dy_u, -np.abs(dy_u) * z_follow * 0.3], 1)
            tl = np.stack([0 * dy_l, dy_l, -np.abs(dy_l) * z_follow * 0.3], 1)
            return [(up, tu, 1.0), (lo, tl, 1.0)]

        # BLINK: upper margin travels down to 0.4 mm above the lower margin; lower lid rises a little
        close = np.stack([0 * pu[:, 0], (lo_y - 0.35 * MM - pu[:, 1]) * (0.2 + 0.8 * bell),
                          (lo_z - pu[:, 2]) * 0.5 * bell], 1)
        tl = np.stack([0 * pl[:, 0], 0.9 * MM * np.ones(len(lo)), 0 * pl[:, 0]], 1)
        shapes[f"blink_{side}"] = solve(m, side, [(up, close, 1.0), (lo, tl, 0.6)], keep=[brows[side]])
        shapes[f"wide_{side}"] = solve(m, side, lid(2.6, -0.6), keep=[brows[side]])
        shapes[f"squint_{side}"] = solve(m, side, lid(-1.2, 1.8), keep=[brows[side]])
        shapes[f"lidsUp_{side}"] = solve(m, side, lid(2.2, 1.4), keep=[brows[side]])        # gaze up
        shapes[f"lidsDown_{side}"] = solve(m, side, lid(-4.4, -1.2), keep=[brows[side]])    # gaze down
        b = brows[side]
        bx = m.T[b, 0]
        inner = 1 - (np.abs(bx) - np.abs(bx).min()) / max(1e-9, np.ptp(np.abs(bx)))
        raise_t = np.stack([0 * bx, 5.5 * MM * (0.6 + 0.4 * inner), 0.8 * MM * np.ones(len(b))], 1)
        shapes[f"browRaise_{side}"] = solve(m, side, [(b, raise_t, 1.0)] + lid(1.2, 0.0), lam=6e-7)
        frown_t = np.stack([-np.sign(bx) * 2.5 * MM * inner, -3.5 * MM * (0.4 + 0.6 * inner), 1.0 * MM * inner], 1)
        shapes[f"browDown_{side}"] = solve(m, side, [(b, frown_t, 1.0)] + lid(-0.6, 0.0), lam=6e-7)
    pupil = np.zeros(m.B_ex.shape[0])
    pupil[m.expr_names.index("pupils_000")] = 1.0
    shapes["pupil"] = pupil
    return shapes, marg


def load(m=None, rebuild=False):
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    m = m or G.GNM()
    shapes, _ = build(m)
    np.savez(CACHE, **shapes)
    return shapes


if __name__ == "__main__":
    m = G.GNM()
    shapes, marg = build(m)
    np.savez(CACHE, **shapes)
    for side in ("left", "right"):
        up, lo = marg[side]
        for name in ("blink", "wide", "squint", "lidsUp", "lidsDown", "browRaise", "browDown"):
            e = shapes[f"{name}_{side}"]
            v, _ = m(expression=e)
            gap = (v[up, 1].mean() - v[lo, 1].mean()) * 1000
            gap0 = (m.T[up, 1].mean() - m.T[lo, 1].mean()) * 1000
            brow = (v[m.groups[f"{side}_brow_region"] > 0.5, 1] - m.T[m.groups[f"{side}_brow_region"] > 0.5, 1]).mean() * 1000
            print(f"{name+'_'+side:16s} |c|max={np.abs(e).max():5.2f}  lid gap {gap0:4.1f} -> {gap:5.1f} mm   brow dy {brow:+5.1f} mm")
