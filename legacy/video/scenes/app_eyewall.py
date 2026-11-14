"""PROBLEM b-roll: eye close-ups of each cast member at 6 gaze directions (1024x512, like CAM_GazeEyes).
blender -b --factory-startup -P video/scenes/app_eyewall.py -- maya"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scene_kit as K  # noqa: E402
from mathutils import Vector  # noqa: E402

who = K.argv()[0]
K.reset()
K.cycles(res=(1024, 512), samples=24)
K.world_hdri("interior.exr", 0.4)
rig, tgt = K.append_cast(who)
K.arms_down(rig)
mid = K.eye_mid(rig)
cam = K.camera("CAM_Eyes", mid + Vector((0, -0.4, 0)), mid, lens=140)
dirs = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), (0.7, 0.7)]
pts = [(i + 1, mid + Vector((h * 0.05, -0.35, v * 0.05))) for i, (h, v) in enumerate(dirs)]
for f, p in pts:
    K.look_at_world(tgt, p)
    tgt.location = tgt.parent.matrix_world.inverted() @ p
    tgt.keyframe_insert("location", frame=f)
K.area("KEY", mid + Vector((-0.5, -0.6, 0.4)), mid, 40, 1.2, (1.0, 0.93, 0.85))
K.area("FILL", mid + Vector((0.5, -0.6, 0.1)), mid, 20, 1.2)
K.sc.frame_start, K.sc.frame_end = 1, len(dirs)
K.render(os.path.join(K.BUILD, "eyewall", who))
