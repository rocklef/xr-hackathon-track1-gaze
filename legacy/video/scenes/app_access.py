"""ACCESS: Arthur types with his eyes. Shot A: his face while his eyes hop across the keys; shot B: over his
shoulder onto the screen as "HELLO SAM" appears and is spoken. The eye rig follows the screen's gaze points.
blender -b --factory-startup -P video/scenes/app_access.py -- [--frames a-b] [--samples n]"""
import argparse
import json
import os
import sys

import bpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scene_kit as K  # noqa: E402
from mathutils import Vector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=None)
ap.add_argument("--samples", type=int, default=16)
a = ap.parse_args(K.argv())
N, CUT = 290, 140
SEQ = os.path.join(K.BUILD, "screens", "keyboard")

K.reset()
K.cycles(samples=a.samples)
K.world_hdri("interior.exr", 0.35, rot=4.0)
rig, tgt = K.append_cast("arthur")
K.sit(rig)
rig.location.z -= 0.42
K.arm(rig, "L", (0.15, -0.35, -0.9), (0.0, -1.0, -0.1))
K.arm(rig, "R", (-0.15, -0.35, -0.9), (0.0, -1.0, -0.1))
mid = K.eye_mid(rig)

wood = K.pbr("MAT_Desk", (0.45, 0.31, 0.2), 0.45)
dz = mid.z - 0.5
K.box("SET_Desk", (1.6, 0.8, 0.04), Vector((mid.x, mid.y - 0.62, dz)), wood, 0.01)
K.box("SET_DeskLeg", (1.5, 0.04, dz), Vector((mid.x, mid.y - 0.95, dz / 2)), wood, 0.005)
K.box("SET_Chair", (0.6, 0.12, 0.8), Vector((mid.x, mid.y + 0.25, mid.z - 0.55)), K.pbr("MAT_Chair", (0.08, 0.09, 0.1), 0.6), 0.05)
K.box("SET_Wall", (6, 0.1, 4), Vector((mid.x, mid.y - 2.2, 1.5)), K.pbr("MAT_Wall", (0.62, 0.6, 0.56), 0.9), 0)
# monitor on a stand
mon_c = Vector((mid.x, mid.y - 0.78, mid.z - 0.08))
bez = K.pbr("MAT_Bezel", (0.02, 0.02, 0.022), 0.3)
K.box("SET_Monitor", (0.66, 0.035, 0.42), mon_c + Vector((0, -0.02, 0)), bez, 0.01)
K.box("SET_MonStand", (0.05, 0.05, mon_c.z - dz - 0.2), Vector((mon_c.x, mon_c.y - 0.03, (mon_c.z + dz) / 2 - 0.1)), bez, 0.005)
scr = K.image_screen("SCR_Keyboard", 0.62, 0.3875, mon_c + Vector((0, 0.0, 0)), mid, SEQ, N, strength=1.4)
# real scanned objects on the desk
for name, p, rz in (("Kanex_MultiSync_Wireless_Keyboard", (0.0, -0.4), 0.0), ("Cole_Hardware_Mug_Classic_Blue", (0.42, -0.45), 0.9),
                    ("Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book", (-0.5, -0.55), 0.3),
                    ("Crosley_Alarm_Clock_Vintage_Metal", (0.55, -0.75), -0.5)):
    K.gso(name, (mid.x + p[0], mid.y + p[1], dz + 0.02), rot_z=rz)

gaze = json.load(open(os.path.join(SEQ, "gaze.json")))
K.key_gaze(tgt, [(f, K.screen_point(scr, u, v)) for f, (u, v) in enumerate(gaze, start=1)])

focus_a = K.empty("FOCUS_A", mid)
cam_a = K.camera("CAM_A", mid + Vector((0.42, -0.62, -0.02)), mid + Vector((0, 0, -0.04)), lens=55, fstop=2.4, focus=focus_a)
focus_b = K.empty("FOCUS_B", mon_c)
cam_b = K.camera("CAM_B", mid + Vector((-0.28, 0.42, 0.12)), mon_c + Vector((0.02, 0, -0.02)), lens=38, fstop=3.5, focus=focus_b)
for f in range(1, N + 1):
    t = (f - 1) / (N - 1)
    cam_a.location = mid + Vector((0.42 - 0.06 * t, -0.62 + 0.08 * t, -0.02))
    cam_a.keyframe_insert("location", frame=f)
    cam_b.location = mid + Vector((-0.28 + 0.05 * t, 0.42 - 0.12 * t, 0.12))
    cam_b.keyframe_insert("location", frame=f)
K.aim(cam_a, mid + Vector((0, 0, -0.04)))
m1 = K.sc.timeline_markers.new("A", frame=1)
m1.camera = cam_a
m2 = K.sc.timeline_markers.new("B", frame=CUT)
m2.camera = cam_b
K.sc.camera = cam_a

K.area("KEY", mid + Vector((0.9, -0.8, 0.6)), mid, 70, 1.4, (1.0, 0.9, 0.78))
K.area("RIM", mid + Vector((-0.8, 0.8, 0.6)), mid, 80, 0.8, (0.75, 0.85, 1.0))
K.area("DESK", Vector((mid.x - 0.4, mid.y - 0.6, mid.z + 0.9)), Vector((mid.x, mid.y - 0.6, dz)), 60, 1.0, (1.0, 0.88, 0.72))

K.sc.frame_start, K.sc.frame_end = 1, N
fr = tuple(int(x) for x in a.frames.split("-")) if a.frames else None
K.render(os.path.join(K.BUILD, "app_access"), fr)
