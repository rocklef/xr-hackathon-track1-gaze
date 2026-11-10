"""CAR: Arjun driving; his eyes drift from the road to a phone, the dashboard distraction warning lights up.
blender -b --factory-startup -P video/scenes/app_car.py -- [--frames a-b] [--samples n]"""
import argparse
import math
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
N = 245
F_DRIFT, F_WARN = 128, 150          # eyes leave the road ("Since July..."), warning on "warn"

K.reset()
K.cycles(samples=a.samples)
w = K.world_hdri("city.exr", 0.9, rot=0.0)
# fake forward motion: slowly rotate the environment
mp = next(n for n in w.node_tree.nodes if n.type == "MAPPING")
for f, r in ((1, 0.0), (N, 0.35)):
    mp.inputs["Rotation"].default_value[2] = r
    mp.inputs["Rotation"].keyframe_insert("default_value", frame=f)

rig, tgt = K.append_cast("arjun")
K.sit(rig)
rig.location.z -= 0.42
mid = K.eye_mid(rig)
# hands forward onto the wheel
K.arm(rig, "L", (0.25, -0.6, -0.75), (-0.25, -0.95, 0.15))
K.arm(rig, "R", (-0.25, -0.6, -0.75), (0.25, -0.95, 0.15))
hl, hr = K.bone_world(rig, "wrist.L"), K.bone_world(rig, "wrist.R")
wc = (hl + hr) / 2 + Vector((0, -0.04, 0.02))
wr = (hl - hr).length / 2 + 0.01

dark = K.pbr("MAT_Dash", (0.03, 0.032, 0.036), 0.55)
trim = K.pbr("MAT_Trim", (0.12, 0.12, 0.13), 0.35, metal=0.6)
seat = K.pbr("MAT_Seat", (0.05, 0.05, 0.055), 0.7)
bpy.ops.mesh.primitive_torus_add(major_radius=wr, minor_radius=0.018, location=wc, rotation=(math.radians(70), 0, 0))
wheel = bpy.context.active_object
wheel.data.materials.append(dark)
K.box("CAR_Hub", (0.09, 0.03, 0.09), wc + Vector((0, 0.03, -0.01)), trim, 0.01, rot=(math.radians(-20), 0, 0))
K.box("CAR_Column", (0.05, 0.35, 0.05), wc + Vector((0, -0.2, -0.08)), dark, 0.01, rot=(math.radians(-20), 0, 0))
K.box("CAR_Dash", (1.9, 0.55, 0.32), Vector((mid.x - 0.25, wc.y - 0.45, wc.z - 0.12)), dark, 0.05)
K.box("CAR_DashTop", (1.9, 0.35, 0.05), Vector((mid.x - 0.25, wc.y - 0.55, wc.z + 0.06)), trim, 0.02)
for x in (mid.x + 0.6, mid.x - 1.1):
    K.box("CAR_Pillar", (0.07, 0.07, 0.75), Vector((x, wc.y - 0.55, wc.z + 0.35)), dark, 0.02, rot=(math.radians(-35), 0, 0))
K.box("CAR_Roof", (1.9, 0.9, 0.05), Vector((mid.x - 0.25, mid.y + 0.1, mid.z + 0.28)), K.pbr("MAT_Headliner", (0.12, 0.115, 0.11), 0.9), 0.02)
K.box("CAR_SeatBack", (0.55, 0.12, 0.75), Vector((mid.x, mid.y + 0.22, mid.z - 0.45)), seat, 0.05, rot=(math.radians(-10), 0, 0))
K.box("CAR_Headrest", (0.28, 0.1, 0.2), Vector((mid.x, mid.y + 0.2, mid.z + 0.05)), seat, 0.04)
K.box("CAR_Door", (0.08, 1.4, 0.55), Vector((mid.x + 0.42, mid.y - 0.2, mid.z - 0.7)), dark, 0.03)
K.box("CAR_Console", (0.25, 0.7, 0.25), Vector((mid.x - 0.45, mid.y - 0.3, mid.z - 0.75)), dark, 0.03)
# phone on the console, screen lit
phone = K.box("CAR_Phone", (0.075, 0.155, 0.009), Vector((mid.x - 0.42, mid.y - 0.45, mid.z - 0.615)), K.pbr("MAT_Phone", (0.02, 0.02, 0.02), 0.2), 0.005,
              rot=(math.radians(-25), 0, math.radians(-15)))
scr = K.box("CAR_PhoneScreen", (0.066, 0.14, 0.002), phone.location + Vector((0, 0, 0.006)), K.emissive("MAT_PhoneScr", (0.55, 0.8, 1.0), 4.0), 0,
            rot=(math.radians(-25), 0, math.radians(-15)))
# distraction warning icon on the instrument cluster (blinks)
warn_mat = K.emissive("MAT_Warn", (1.0, 0.12, 0.05), 0.0)
warn = K.box("CAR_Warn", (0.07, 0.004, 0.05), wc + Vector((0, -0.17, 0.0)), warn_mat, 0, rot=(math.radians(-20), 0, 0))
em = warn_mat.node_tree.nodes["Emission"].inputs["Strength"]
for f in range(1, N + 1):
    on = f >= F_WARN and ((f - F_WARN) // 8) % 2 == 0
    em.default_value = 40.0 if on else 0.0
    em.keyframe_insert("default_value", frame=f)
warn_light = bpy.data.lights.new("WarnGlow", "POINT")
warn_light.color, warn_light.shadow_soft_size = (1.0, 0.15, 0.08), 0.05
wl = bpy.data.objects.new("WarnGlow", warn_light)
K.sc.collection.objects.link(wl)
wl.location = wc + Vector((0, -0.12, 0.05))
for f in range(1, N + 1):
    on = f >= F_WARN and ((f - F_WARN) // 8) % 2 == 0
    warn_light.energy = 6.0 if on else 0.0
    warn_light.keyframe_insert("energy", frame=f)

# gaze: down the road, then to the phone
road = mid + Vector((0.0, -8.0, -0.4))
K.key_gaze(tgt, [(1, road), (60, road + Vector((0.6, 0, 0))), (90, road), (F_DRIFT - 8, road),
                 (F_DRIFT + 6, scr.location), (N, scr.location)])
K.key_bone_rot(rig, "head", [(1, (0, 0, 0)), (F_DRIFT - 4, (0, 0, 0)), (F_DRIFT + 14, (18, -12, 0)), (N, (20, -12, 0))])

focus = K.empty("FOCUS", mid)
cam = K.camera("CAM_Car", mid + Vector((-0.75, -0.75, -0.05)), mid + Vector((0, 0, -0.1)), lens=32, fstop=2.8, focus=focus)
for f in range(1, N + 1):
    t = K.ease((f - 1) / (N - 1))
    j = Vector((math.sin(f * 0.9) * 0.002, math.sin(f * 1.3) * 0.0015, math.sin(f * 1.7) * 0.002))
    cam.location = mid + Vector((-0.75 + 0.12 * t, -0.75 + 0.1 * t, -0.05)) + j
    cam.keyframe_insert("location", frame=f)
    K.aim(cam, mid + Vector((0, 0, -0.12)))
    cam.keyframe_insert("rotation_euler", frame=f)

K.area("KEY", mid + Vector((-0.6, -0.9, 0.5)), mid, 45, 1.2, (1.0, 0.95, 0.88))
K.area("WINDOW", mid + Vector((0.9, -0.2, 0.3)), mid, 70, 1.0, (0.85, 0.9, 1.0))

K.sc.frame_start, K.sc.frame_end = 1, N
fr = tuple(int(x) for x in a.frames.split("-")) if a.frames else None
K.render(os.path.join(K.BUILD, "app_car"), fr)
