"""Write the eye (iris + sclera), teeth and tongue UDIM tiles for every texture set (system Python + Pillow).
python gnm_rig/eye_tiles.py [--ids T01,T02] [--size 2048]"""
import argparse
import os

import numpy as np
from PIL import Image

import gnm_model as G
import iris_hq as IRH
from presets import PRESETS


def flat(color, size):
    return Image.fromarray(np.tile(np.array(color, dtype=np.uint8), (size, size, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(p["id"] for p in PRESETS))
    ap.add_argument("--size", type=int, default=4096)
    a = ap.parse_args()
    for P in PRESETS:
        if P["id"] not in a.ids.split(","):
            continue
        d = os.path.join(G.ROOT, "textures", f"{P['id']}_{P['name']}")
        os.makedirs(d, exist_ok=True)
        # separate maps per eye: same colour family, different seeds (real left/right irises differ);
        # albedo + relief normal map for each eye
        IRH.save_pair(P["iris"], P["iris_seed"], os.path.join(d, f"{P['id']}_albedo.1002.png"),
                      os.path.join(d, f"{P['id']}_albedo.1004.png"), os.path.join(d, f"{P['id']}_normal.1002.png"),
                      os.path.join(d, f"{P['id']}_normal.1004.png"), a.size)
        for tile in (1006, 1007):
            flat((128, 128, 255), 256).save(os.path.join(d, f"{P['id']}_normal.{tile}.png"))
        for tile, r in ((1002, 60), (1004, 60), (1006, 90), (1007, 110)):
            flat((r, r, r), 256).save(os.path.join(d, f"{P['id']}_roughness.{tile}.png"))
        flat((236, 228, 214), 512).save(os.path.join(d, f"{P['id']}_albedo.1006.png"))
        flat((178, 92, 92), 512).save(os.path.join(d, f"{P['id']}_albedo.1007.png"))
        print(P["id"], "eye/teeth/tongue tiles ->", d)


if __name__ == "__main__":
    main()
