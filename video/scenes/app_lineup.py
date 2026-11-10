"""SOLUTION: the whole cast side by side in a dark studio; one shared target moves and every pair of eyes
follows it in sync ("eyes we can point anywhere").
blender -b --factory-startup -P video/scenes/app_lineup.py -- [--frames a-b] [--samples n]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scene_kit as K  # noqa: E402
from mathutils import Vector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=None)
ap.add_argument("--samples", type=int, default=16)
a = ap.parse_args(K.argv())
N = 210

K.reset()
K.cycles(samples=a.samples)
K.world_hdri("studio.exr", 0.05)
people = []
# group-photo layout: adults in a shallow arc, the child in front between them
PLACE = {"arthur": (-0.60, 0.08), "maya": (-0.19, 0.0), "zuri": (0.16, -0.40), "arjun": (0.56, 0.08)}
for who, (x, y) in PLACE.items():
    rig, tgt = K.append_cast(who)
    rig.location.x, rig.location.y = x, y
    K.arms_down(rig)
    people.append((rig, tgt))
K.bpy.context.view_layer.update()
mids = [K.eye_mid(r) for r, _ in people]
top = max(m.z for m in mids) + 0.17      # just above the tallest head
bot = min(m.z for m in mids) - 0.20      # the child's chin/collar
xl, xr = min(m.x for m in mids) - 0.24, max(m.x for m in mids) + 0.24
c = Vector(((xl + xr) / 2, sum(m.y for m in mids) / len(mids), (top + bot) / 2))
# distance so the whole group fits a 50 mm, 16:9 frame (36 x 20.25 mm sensor)
D = max((top - bot) / (20.25 / 50), (xr - xl) / (36 / 50))
print("[lineup] eyes z", [round(m.z, 3) for m in mids], "dist", round(D, 2), flush=True)

K.box("SET_Floor", (12, 12, 0.02), Vector((0, 0, -0.01)), K.pbr("MAT_Floor", (0.02, 0.022, 0.026), 0.35), 0)
K.box("SET_Back", (12, 0.1, 6), Vector((0, 2.2, 3)), K.pbr("MAT_Back", (0.015, 0.017, 0.02), 0.6), 0)

cam = K.camera("CAM_Lineup", c + Vector((0, -D * 1.12, 0)), c, lens=50, fstop=8, focus=K.empty("FOCUS", c + Vector((0, -0.15, 0))))
for f in range(1, N + 1):
    t = K.ease((f - 1) / (N - 1))
    cam.location = c + Vector((0.18 - 0.36 * t, -D * (1.12 - 0.12 * t), 0))
    cam.keyframe_insert("location", frame=f)
    K.aim(cam, c)
    cam.keyframe_insert("rotation_euler", frame=f)

# one shared point everybody looks at (in front of the group)
base = Vector((c.x, c.y - 1.6, sum(m.z for m in mids) / len(mids)))
path = [(1, base), (25, base), (55, base + Vector((-1.1, 0, 0.05))), (80, base + Vector((-1.1, 0, 0.05))),
        (110, base + Vector((1.1, 0, 0.0))), (135, base + Vector((1.1, 0, 0.0))), (160, base + Vector((0, 0, 0.75))),
        (180, base + Vector((0, 0, 0.75))), (198, cam.location.copy()), (N, cam.location.copy())]
for rig, tgt in people:
    K.key_gaze(tgt, path)

K.area("KEY", c + Vector((-1.5, -2.2, 1.6)), c, 380, 2.5, (1.0, 0.95, 0.9))
K.area("RIM_L", c + Vector((-2.2, 1.2, 0.8)), c, 420, 1.2, (0.35, 0.75, 1.0))
K.area("RIM_R", c + Vector((2.2, 1.2, 0.8)), c, 420, 1.2, (0.35, 0.75, 1.0))
K.area("FILL", c + Vector((1.5, -2.2, 0.3)), c, 60, 2.0)

K.sc.frame_start, K.sc.frame_end = 1, N
fr = tuple(int(x) for x in a.frames.split("-")) if a.frames else None
K.render(os.path.join(K.BUILD, "app_lineup"), fr)
