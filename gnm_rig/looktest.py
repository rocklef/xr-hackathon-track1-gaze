"""Quick geometry / rig look test: renders a few poses with flat placeholder materials (or the real textures
if --skin is given).  blender -b --factory-startup gnm_rig/out/gnm_head.blend -P gnm_rig/looktest.py -- --out DIR"""
import argparse
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--placeholder", action="store_true")
ap.add_argument("--poses", default="neutral,left,up,down,blink,brow")
ap.add_argument("--view", default="front")
ap.add_argument("--tex", default=None)
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

if a.placeholder:
    def flat(name, col, rough=0.5, alpha=1.0, ior=None, sss=0.0):
        m = bpy.data.materials[name]
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        b.inputs["Base Color"].default_value = (*col, 1)
        b.inputs["Roughness"].default_value = rough
        if sss:
            b.inputs["Subsurface Weight"].default_value = sss
            b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
            b.inputs["Subsurface Scale"].default_value = 0.004
        if ior:
            b.inputs["Transmission Weight"].default_value = 1.0
            b.inputs["IOR"].default_value = ior
            b.inputs["Roughness"].default_value = 0.0
    flat("MAT_skin", (0.72, 0.5, 0.4), 0.45, sss=0.25)
    for s in "LR":
        flat(f"MAT_eye_interior_{s}", (0.8, 0.8, 0.78), 0.3)
        flat(f"MAT_cornea_{s}", (1, 1, 1), 0.0, ior=1.376)
    flat("MAT_teeth", (0.85, 0.82, 0.75), 0.3)
    flat("MAT_tongue", (0.6, 0.2, 0.2), 0.4)

if a.tex:
    import materials as MT
    MT.apply(a.tex)
S.cycles((1280, 720), a.samples)
S.world("studio.exr", 0.4)
mid = S.eye_mid()
S.portrait_lights(mid)
S.backdrop(mid)
if a.view == "front":
    S.camera("CAM", mid + Vector((0, -0.62, -0.02)), mid + Vector((0, 0, -0.035)), lens=50)
elif a.view == "eyes":
    S.camera("CAM", mid + Vector((0, -0.40, 0.0)), mid, lens=85)
else:
    S.camera("CAM", mid + Vector((-0.45, -0.45, 0.02)), mid + Vector((0, 0, -0.03)), lens=60)

POSES = {"neutral": dict(g=(0, 0)), "left": dict(g=(-25, 0)), "right": dict(g=(25, 0)), "up": dict(g=(0, 22)),
         "down": dict(g=(0, -25)), "blink": dict(g=(0, 0), blink_L=1, blink_R=1),
         "half": dict(g=(0, 0), blink_L=0.5, blink_R=0.5), "brow": dict(g=(0, 8), brow_raise=1, wide=0.6),
         "frown": dict(g=(0, -5), brow_down=1, squint=0.7), "pupil": dict(g=(0, 0), pupil=1.5)}
os.makedirs(a.out, exist_ok=True)
for name in a.poses.split(","):
    p = dict(POSES[name])
    h, v = p.pop("g")
    S.set_ctrl(blink_L=0, blink_R=0, wide=0, squint=0, brow_raise=0, brow_down=0, pupil=0)
    S.set_ctrl(**p)
    S.set_gaze(h, v)
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = os.path.join(a.out, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print("rendered", name, flush=True)
