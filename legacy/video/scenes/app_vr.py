"""VR: Maya in her living room wearing a VR headset, looking around the virtual world (head turns).
blender -b --factory-startup -P video/scenes/app_vr.py -- [--frames a-b] [--samples n]"""
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
N = 310

K.reset()
K.cycles(samples=a.samples)
K.world_hdri("interior.exr", 0.4, rot=2.2)
rig, tgt = K.append_cast("maya")
K.arms_down(rig)
# hands come up a little, as if feeling the virtual space
K.arm(rig, "R", (-0.25, -0.35, -0.9), (-0.1, -1.0, 0.15))
mid = K.eye_mid(rig)
K.vr_headset(rig)

# living room: sofa behind, rug, coffee table in front with real scanned objects
fabric = K.pbr("MAT_Sofa", (0.16, 0.22, 0.26), 0.85)
K.box("SET_SofaBase", (2.2, 0.9, 0.45), Vector((0, 1.0, 0.225)), fabric, 0.06)
K.box("SET_SofaBack", (2.2, 0.25, 0.55), Vector((0, 1.38, 0.7)), fabric, 0.08)
for x in (-1.15, 1.15):
    K.box("SET_SofaArm", (0.22, 0.9, 0.62), Vector((x, 1.0, 0.31)), fabric, 0.07)
K.box("SET_Rug", (3.2, 2.4, 0.01), Vector((0, -0.3, 0.005)), K.pbr("MAT_Rug", (0.55, 0.47, 0.38), 0.95), 0)
floor = K.box("SET_Floor", (12, 12, 0.02), Vector((0, 0, -0.01)), K.pbr("MAT_Floor", (0.28, 0.18, 0.11), 0.45), 0)
table = K.pbr("MAT_Table", (0.42, 0.28, 0.17), 0.4)
tz = 0.42
K.box("SET_Table", (1.1, 0.55, 0.05), Vector((0.15, -0.85, tz)), table, 0.01)
for dx in (-0.48, 0.48):
    for dy in (-0.22, 0.22):
        K.box("SET_Leg", (0.04, 0.04, tz), Vector((0.15 + dx, -0.85 + dy, tz / 2)), table, 0.005)
# side table at waist height, in frame
sz = 0.72
K.box("SET_SideTable", (0.6, 0.45, 0.04), Vector((0.78, -0.25, sz)), table, 0.01)
K.box("SET_SideLeg", (0.05, 0.05, sz), Vector((0.78, -0.25, sz / 2)), table, 0.005)
for name, p, rz in (("Cole_Hardware_Orchid_Pot_85", (0.62, -0.18), 0.0), ("ACE_Coffee_Mug_Kristen_16_oz_cup", (0.9, -0.34), 2.0),
                    ("Crosley_Alarm_Clock_Vintage_Metal", (0.95, -0.12), 0.7)):
    K.gso(name, (p[0], p[1], sz + 0.02), rot_z=rz)
for name, p, rz in (("Cole_Hardware_Mug_Classic_Blue", (0.42, -0.82), 0.6),
                    ("Nintendo_Wii_Party_U_with_Controller_Wii_U_Game", (-0.12, -0.9), 0.25),
                    ("JBL_Charge_Speaker_portable_wireless_wired_Green", (0.2, -0.75), -0.4),
                    ("Razer_Naga_MMO_Gaming_Mouse", (0.6, -0.95), 0.9)):
    K.gso(name, (p[0], p[1], tz + 0.025), rot_z=rz)

# head turns: looking around the virtual world
K.key_bone_rot(rig, "head", [(1, (0, 0, 0)), (60, (0, 0, 0)), (110, (-6, 18, 0)), (170, (-6, 18, 0)),
                             (220, (8, -16, 0)), (270, (8, -16, 0)), (N, (0, 4, 0))])
K.key_bone_rot(rig, "neck02", [(1, (0, 0, 0)), (60, (0, 0, 0)), (110, (0, 8, 0)), (170, (0, 8, 0)),
                               (220, (0, -8, 0)), (270, (0, -8, 0)), (N, (0, 0, 0))])

focus = K.empty("FOCUS", mid)
cam = K.camera("CAM_VR", mid + Vector((-0.55, -1.6, -0.15)), mid + Vector((0.05, 0, -0.25)), lens=45, fstop=2.8, focus=focus)
for f in range(1, N + 1):
    t = K.ease((f - 1) / (N - 1))
    cam.location = mid + Vector((-0.55 + 0.25 * t, -1.6 + 0.55 * t, -0.15 + 0.05 * t))
    cam.keyframe_insert("location", frame=f)
    K.aim(cam, mid + Vector((0.05, 0, -0.25 + 0.15 * t)))
    cam.keyframe_insert("rotation_euler", frame=f)

K.area("KEY", mid + Vector((-1.4, -1.2, 0.9)), mid, 220, 2.0, (1.0, 0.88, 0.75))
K.area("RIM", mid + Vector((1.2, 1.0, 0.8)), mid, 160, 1.0, (0.7, 0.82, 1.0))
K.area("TABLE", Vector((0.15, -0.85, 1.6)), Vector((0.15, -0.85, tz)), 60, 0.8, (1.0, 0.85, 0.7))

K.sc.frame_start, K.sc.frame_end = 1, N
fr = tuple(int(x) for x in a.frames.split("-")) if a.frames else None
K.render(os.path.join(K.BUILD, "app_vr"), fr)
