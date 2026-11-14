"""Soft per-vertex masks on GNM's topology, used to drive the procedural skin layers.

All masks come from GNM's own vertex groups plus simple geometric rules, then are diffused over the mesh
(Laplacian smoothing), so every layer fades naturally and is continuous across UV seams.
"""
import numpy as np

import gnm_model as G


def adjacency(m):
    q = m.quads
    e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    return e


def smooth(w, edges, n, iters):
    w = w.astype(np.float64).copy()
    deg = np.bincount(edges[:, 0], minlength=n).astype(np.float64)
    for _ in range(iters):
        acc = np.zeros(n)
        np.add.at(acc, edges[:, 0], w[edges[:, 1]])
        w = 0.5 * w + 0.5 * acc / np.maximum(deg, 1)
    return w


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def build(m=None):
    m = m or G.GNM()
    g, T, n = m.groups, m.T, m.nv
    E = adjacency(m)
    x, y, z = T[:, 0], T[:, 1], T[:, 2]
    skin = (g["skin"] > 0.5).astype(float)
    face = g["hockey_mask"]
    out = {}
    out["m_lips"] = np.clip(smooth(g["upper_lip"] + g["lower_lip"], E, n, 2) * 1.3, 0, 1)
    red = 0.9 * g["nose_region"] + 0.75 * (g["left_cheek_region"] + g["right_cheek_region"]) + 0.6 * g["ears"] \
        + 0.35 * g["chin_region"] + 0.25 * (g["left_zygomatic_region"] + g["right_zygomatic_region"])
    out["m_red"] = np.clip(smooth(red, E, n, 14), 0, 1)
    # brows: GNM brow regions, tapered towards the outer tail and lifted into an arch band
    brow = g["left_brow_region"] + g["right_brow_region"] + 0.35 * g["middle_brow_region"]
    out["m_brow"] = np.clip(smooth(brow, E, n, 3), 0, 1)
    # under-eye shadow
    out["m_undereye"] = np.clip(smooth(g["left_infraorbital_region"] + g["right_infraorbital_region"], E, n, 6), 0, 1)
    # T-zone (oilier, less rough)
    out["m_tzone"] = np.clip(smooth(g["forehead_region"] + g["nose_region"] + 0.6 * g["chin_region"], E, n, 8), 0, 1)
    out["m_forehead"] = np.clip(smooth(g["forehead_region"], E, n, 6), 0, 1)
    # crow's feet: lateral orbital corners
    orb = g["left_orbital_region"] + g["right_orbital_region"]
    lat = smoothstep(0.035, 0.05, np.abs(x)) * orb
    out["m_crow"] = np.clip(smooth(lat, E, n, 4), 0, 1)
    # beard: below the cheekbones on the front/sides of the face, plus the moustache area; never the lips
    mouth_y = T[g["upper_lip"] > 0.5, 1].mean()
    lower = smoothstep(mouth_y + 0.035, mouth_y + 0.005, y)
    front = smoothstep(-0.02, 0.04, z)
    beard_area = (face + g["chin_region"] + g["left_parotid_region"] + g["right_parotid_region"]).clip(0, 1)
    beard = lower * front * beard_area + g["upper_lip_region"] * 0.9
    beard = beard * (1 - np.clip(g["upper_lip"] + g["lower_lip"], 0, 1)) * (1 - g["nose_region"]) * skin
    out["m_beard"] = np.clip(smooth(beard, E, n, 6), 0, 1) * (1 - np.clip(smooth(g["upper_lip"] + g["lower_lip"], E, n, 2) * 2, 0, 1))
    # scalp (buzz-cut hair area): skin above the forehead and behind the temples, not ears/face
    fh_top = T[g["forehead_region"] > 0.5, 1].max()
    scalp = smoothstep(fh_top - 0.012, fh_top + 0.006, y) + smoothstep(0.0, -0.04, z) * smoothstep(0.24, 0.30, y)
    temple = (g["left_temple_region"] + g["right_temple_region"]) * smoothstep(0.30, 0.33, y)
    scalp = np.clip(scalp + temple, 0, 1) * (1 - g["ears"]) * (1 - face * smoothstep(0.34, 0.33, y)) * skin
    out["m_scalp"] = np.clip(smooth(scalp, E, n, 5), 0, 1)
    out["m_face"] = np.clip(smooth(face, E, n, 8), 0, 1)
    out["m_ears"] = np.clip(smooth(g["ears"], E, n, 3), 0, 1)
    out["m_socket"] = np.clip(smooth(g["eye_sockets"] + g["mouth_sock"], E, n, 2), 0, 1)
    # waterline: the lid-margin ring and the first row inside it (wet, pinkish)
    import expressions as X
    ring = np.zeros(n)
    for side, (up, lo) in X.lid_margins(m).items():
        ring[up] = 0.6
        ring[lo] = 1.0
    out["m_margin"] = np.clip(smooth(ring, E, n, 1) * 1.6, 0, 1) * np.clip(g["eye_sockets"] + ring, 0, 1)
    return out


if __name__ == "__main__":
    import os
    ms = build()
    np.savez(os.path.join(G.HERE, "skin_masks.npz"), **ms)
    for k, v in ms.items():
        print(f"{k:12s} >0.5: {int((v > 0.5).sum()):6d}  max {v.max():.2f}")
