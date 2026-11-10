"""OPEN: extreme close-up of Maya's eyes glancing around, pull back to reveal her at home.
blender -b --factory-startup -P video/scenes/app_open.py -- [--frames a-b] [--samples n]"""
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
N = 200

K.reset()
K.cycles(samples=a.samples)
K.world_hdri("interior.exr", 0.35, rot=0.6)
rig, tgt = K.append_cast("maya")
K.arms_down(rig)
mid = K.eye_mid(rig)

# soft shelf of real scanned objects behind her (out of focus)
wood = K.pbr("MAT_Shelf", (0.32, 0.2, 0.12), 0.55)
K.box("SET_Shelf", (1.4, 0.35, 0.04), mid + Vector((0.25, 1.1, -0.42)), wood)
for name, dx, rz in (("Crosley_Alarm_Clock_Vintage_Metal", -0.35, 0.4), ("JBL_Charge_Speaker_portable_wireless_wired_Green", 0.05, -0.3),
                     ("Cole_Hardware_Orchid_Pot_85", 0.42, 0.0), ("Cole_Hardware_Mug_Classic_Blue", 0.7, 1.0)):
    K.gso(name, mid + Vector((0.25 + dx, 1.1, -0.40)), rot_z=rz)

focus = K.empty("FOCUS", mid)
cam = K.camera("CAM_Open", mid + Vector((0.0, -0.3, 0.0)), mid, lens=85, fstop=2.2, focus=focus)
for f in range(1, N + 1):
    t = K.ease((f - 1) / (N - 1))
    cam.location = mid + Vector((0.06 * t, -0.30 - 0.62 * t, -0.02 * t))
    cam.keyframe_insert("location", frame=f)
    K.aim(cam, mid + Vector((0, 0, -0.035 * t)))
    cam.keyframe_insert("rotation_euler", frame=f)

c = lambda f: cam.location  # noqa: E731
K.sc.frame_set(1)
far = 0.6
pts = [(1, mid + Vector((0, -far, 0))), (14, mid + Vector((-0.22, -far, 0.04))), (40, mid + Vector((-0.22, -far, 0.04))),
       (52, mid + Vector((0.22, -far, 0.0))), (80, mid + Vector((0.22, -far, 0.0))), (92, mid + Vector((0.1, -far, 0.25))),
       (112, mid + Vector((0.1, -far, 0.25))), (126, mid + Vector((0.06, -0.92, -0.02))), (N, mid + Vector((0.06, -0.92, -0.02)))]
K.key_gaze(tgt, pts)

K.area("KEY", mid + Vector((-0.9, -0.8, 0.55)), mid, 60, 1.2, (1.0, 0.9, 0.78))
K.area("RIM", mid + Vector((0.8, 0.7, 0.45)), mid, 70, 0.6, (0.75, 0.85, 1.0))
K.area("FILL", mid + Vector((0.8, -0.9, -0.1)), mid, 8, 1.5)
K.area("BG", mid + Vector((0.3, 0.6, 0.3)), mid + Vector((0.25, 1.1, -0.4)), 60, 1.0, (1.0, 0.75, 0.5))

K.sc.frame_start, K.sc.frame_end = 1, N
fr = tuple(int(x) for x in a.frames.split("-")) if a.frames else None
K.render(os.path.join(K.BUILD, "app_open"), fr)
