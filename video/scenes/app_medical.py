"""MEDICAL: Zuri watches a star move on a tablet; her eyes follow it (eye rig driven by the star path) while the
tablet's tracker dot follows her gaze. Shot A: her face from beside the tablet; shot B: over her shoulder.
blender -b --factory-startup -P video/scenes/app_medical.py -- [--frames a-b] [--samples n]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scene_kit as K  # noqa: E402
from mathutils import Vector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=None)
ap.add_argument("--samples", type=int, default=16)
a = ap.parse_args(K.argv())
N, CUT = 215, 125
SEQ = os.path.join(K.BUILD, "screens", "star")

K.reset()
K.cycles(samples=a.samples)
K.world_hdri("interior.exr", 0.5, rot=1.2)
rig, tgt = K.append_cast("zuri")
K.sit(rig)
rig.location.z -= 0.3
K.arm(rig, "L", (0.2, -0.45, -0.85), (-0.15, -1.0, -0.05))
K.arm(rig, "R", (-0.2, -0.45, -0.85), (0.15, -1.0, -0.05))
mid = K.eye_mid(rig)

tz = mid.z - 0.38
table = K.pbr("MAT_KidsTable", (0.93, 0.93, 0.9), 0.35)
K.box("SET_Table", (1.3, 0.8, 0.04), Vector((mid.x, mid.y - 0.55, tz)), table, 0.015)
K.box("SET_TableLeg", (1.2, 0.04, tz), Vector((mid.x, mid.y - 0.9, tz / 2)), K.pbr("MAT_Leg", (0.95, 0.6, 0.2), 0.4), 0.005)
K.box("SET_Wall", (6, 0.1, 4), Vector((mid.x, mid.y - 2.0, 1.5)), K.pbr("MAT_ClinicWall", (0.7, 0.82, 0.86), 0.9), 0)
K.box("SET_Chair", (0.45, 0.1, 0.5), Vector((mid.x, mid.y + 0.2, mid.z - 0.45)), K.pbr("MAT_KidChair", (0.2, 0.55, 0.9), 0.5), 0.04)
# tablet on a little stand
tab_c = Vector((mid.x, mid.y - 0.5, tz + 0.2))
K.box("SET_Tablet", (0.27, 0.012, 0.18), tab_c + Vector((0, -0.008, 0)), K.pbr("MAT_TabletBody", (0.08, 0.08, 0.09), 0.3), 0.008)
K.box("SET_TabStand", (0.05, 0.12, 0.17), Vector((tab_c.x, tab_c.y - 0.06, tz + 0.09)), K.pbr("MAT_Stand", (0.6, 0.6, 0.62), 0.3, 0.8), 0.01)
scr = K.image_screen("SCR_Star", 0.252, 0.1575, tab_c, mid, SEQ, N, strength=1.6)
# real scanned toys on the table
for name, p, rz in (("CASTLE_BLOCKS", (-0.42, -0.45), 0.4), ("Dog", (0.4, -0.38), -0.8), ("BABY_CAR", (0.3, -0.7), 1.2),
                    ("Melissa_Doug_Shape_Sorting_Clock", (-0.35, -0.75), 0.2)):
    K.gso(name, (mid.x + p[0], mid.y + p[1], tz + 0.02), rot_z=rz)

gaze = json.load(open(os.path.join(SEQ, "gaze.json")))
K.key_gaze(tgt, [(f, K.screen_point(scr, u, v)) for f, (u, v) in enumerate(gaze, start=1)])

focus_a = K.empty("FOCUS_A", mid)
cam_a = K.camera("CAM_A", mid + Vector((-0.3, -0.48, -0.06)), mid + Vector((0, 0, -0.02)), lens=55, fstop=2.2, focus=focus_a)
focus_b = K.empty("FOCUS_B", tab_c)
cam_b = K.camera("CAM_B", mid + Vector((0.22, 0.3, 0.1)), tab_c + Vector((0, 0, -0.01)), lens=42, fstop=3.2, focus=focus_b)
for f in range(1, N + 1):
    t = (f - 1) / (N - 1)
    cam_a.location = mid + Vector((-0.3 + 0.05 * t, -0.48 + 0.06 * t, -0.06))
    cam_a.keyframe_insert("location", frame=f)
    cam_b.location = mid + Vector((0.22 - 0.04 * t, 0.3 - 0.08 * t, 0.1))
    cam_b.keyframe_insert("location", frame=f)
m1 = K.sc.timeline_markers.new("A", frame=1)
m1.camera = cam_a
m2 = K.sc.timeline_markers.new("B", frame=CUT)
m2.camera = cam_b
K.sc.camera = cam_a

K.area("KEY", mid + Vector((-0.8, -0.9, 0.7)), mid, 60, 1.6, (1.0, 0.95, 0.88))
K.area("RIM", mid + Vector((0.8, 0.8, 0.5)), mid, 60, 0.8, (0.85, 0.9, 1.0))
K.area("TABLE", Vector((mid.x, mid.y - 0.55, mid.z + 0.9)), Vector((mid.x, mid.y - 0.55, tz)), 50, 1.2)

K.sc.frame_start, K.sc.frame_end = 1, N
fr = tuple(int(x) for x in a.frames.split("-")) if a.frames else None
K.render(os.path.join(K.BUILD, "app_medical"), fr)
