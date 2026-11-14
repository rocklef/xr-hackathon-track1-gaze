"""Render the GNM head with each generated texture set (rubric deliverable: 5-10 renders with different textures).

blender -b --factory-startup gnm_rig/out/gnm_head.blend -P gnm_rig/render_textures.py -- [--tex T01,T02|all]
        [--views portrait,eyes] [--samples 128] [--res 1920x1080] [--out renders_textures]

Each render: same GNM head + rig, a natural gaze, constant studio lighting. Only the UV texture set changes,
which shows the maps wrap directly onto GNM's topology.
"""
import argparse
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import materials as MT  # noqa: E402
import stage as S  # noqa: E402
from presets import PRESETS  # noqa: E402

VIEWS = {
    # name: (camera offset from eye midpoint, look-at offset, lens)
    "portrait": (Vector((-0.30, -0.58, 0.02)), Vector((0.0, 0.0, -0.045)), 70),
    "front": (Vector((0.0, -0.62, -0.01)), Vector((0.0, 0.0, -0.04)), 60),
    "eyes": (Vector((0.0, -0.36, 0.005)), Vector((0.0, 0.0, 0.0)), 100),
    "macro": (Vector((0.045, -0.16, 0.012)), Vector((0.031, 0.0, 0.0)), 100),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="all")
    ap.add_argument("--views", default="portrait,eyes")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "renders_textures"))
    ap.add_argument("--gaze", default="-6,3", help="h,v degrees")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    a.out = os.path.abspath(a.out)          # Blender would resolve a relative path against the drive root
    ids = [p["id"] for p in PRESETS] if a.tex == "all" else a.tex.split(",")
    w, h = (int(v) for v in a.res.split("x"))
    S.cycles((w, h), a.samples)
    S.world("interior.exr", 0.28, rot=2.4)
    mid = S.eye_mid()
    S.portrait_lights(mid)
    S.backdrop(mid)
    gh, gv = (float(v) for v in a.gaze.split(","))
    S.set_ctrl(blink_L=0, blink_R=0, wide=0, squint=0, brow_raise=0, brow_down=0, pupil=-1.0, auto_lids=1)
    S.set_gaze(gh, gv)
    cams = {}
    for v in a.views.split(","):
        off, look, lens = VIEWS[v]
        foc = mid + Vector((0.031, -0.013, 0)) if v == "macro" else mid + Vector((0, -0.012, 0))
        cams[v] = S.camera(f"CAM_{v}", mid + off, mid + look, lens=lens,
                           fstop={"eyes": 4.0, "macro": 8.0}.get(v, 5.6), focus=foc)
    os.makedirs(a.out, exist_ok=True)
    for tid in ids:
        MT.apply(tid)
        name = os.path.basename(MT.set_dir(tid))
        for v, cam in cams.items():
            bpy.context.scene.camera = cam
            bpy.context.scene.render.filepath = os.path.join(a.out, f"{name}_{v}.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", name, v, flush=True)


if __name__ == "__main__":
    main()
