"""Portrait look test of each cast member (EEVEE): blender -b -P video/scenes/looktest.py -- maya"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scene_kit as K  # noqa: E402
from mathutils import Vector  # noqa: E402

who = K.argv()[0]
K.reset()
K.cycles(res=(1080, 1080), samples=24) if "cycles" in K.argv() else K.eevee(res=(1080, 1080), samples=32)
K.world_hdri("interior.exr", 0.45)
rig, tgt = K.append_cast(who)
mid = K.eye_mid(rig)
cam = K.camera("CAM", mid + Vector((0.18, -0.85, 0.02)), mid + Vector((0, 0, -0.06)), lens=70)
K.look_at_world(tgt, cam.location)
K.area("KEY", mid + Vector((-0.8, -0.9, 0.6)), mid, 55, 1.0, (1.0, 0.93, 0.85))
K.area("RIM", mid + Vector((0.9, 0.6, 0.5)), mid, 60, 0.6, (0.85, 0.9, 1.0))
K.area("FILL", mid + Vector((0.9, -0.8, 0.0)), mid, 10, 1.5)
K.sc.frame_start = K.sc.frame_end = 1
K.render(os.path.join(K.BUILD, "looktest", who))
