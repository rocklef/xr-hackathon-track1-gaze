"""Blender side of eyes_cli.py: keys the GNM rig from motion.synthesize() and renders frames (or saves a .blend).

blender -b --factory-startup gnm_rig/out/gnm_head.blend -P gnm_rig/animate.py -- --input left|right|motion|expression
        [--tex T03] [--res 1920x1080] [--samples 48] [--frames-dir DIR] [--save-blend PATH] [--no-render]

Directions are from the viewer's side: "left" = the eyes move to the left of the screen.
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import materials as MT  # noqa: E402
import motion as MO  # noqa: E402
import stage as S  # noqa: E402

DIST = 1.0   # natural viewing distance for the fixation target (m)


def key_frame(f, d, i, rig, tgt):
    eh, ev = math.radians(d["eye_h"][i]), math.radians(d["eye_v"][i])
    tgt.location = (DIST * math.tan(eh), -DIST, DIST * math.tan(ev))
    tgt.keyframe_insert("location", frame=f)
    yaw, pitch = math.radians(d["head_yaw"][i]), math.radians(d["head_pitch"][i])
    for bone, share in (("neck", 0.4), ("head", 0.6)):
        pb = rig.pose.bones[bone]
        pb.rotation_mode = "XYZ"
        # bones point up (+Z): local Y = up axis (yaw), local X = left-right axis (pitch, +X tips the face down)
        pb.rotation_euler = (-pitch * share, yaw * share, 0.0)
        pb.keyframe_insert("rotation_euler", frame=f)
    for k in ("blink_L", "blink_R"):
        rig[k] = float(d["blink"][i])
        rig.keyframe_insert(f'["{k}"]', frame=f)
    for k in ("brow_raise", "wide", "squint", "pupil"):
        rig[k] = float(d[k][i])
        rig.keyframe_insert(f'["{k}"]', frame=f)
    if "brow_down" in d:
        rig["brow_down"] = float(d["brow_down"][i])
        rig.keyframe_insert('["brow_down"]', frame=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, choices=["left", "right", "motion", "expression"])
    ap.add_argument("--tex", default="T03")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--save-blend", default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

    MT.apply(a.tex)
    w, h = (int(v) for v in a.res.split("x"))
    S.cycles((w, h), a.samples, a.fps)
    sc = bpy.context.scene
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.35
    S.world("interior.exr", 0.28, rot=2.4)
    mid = S.eye_mid()
    S.portrait_lights(mid)
    S.backdrop(mid)
    S.camera("CAM_Face", mid + Vector((0.0, -0.66, -0.025)), mid + Vector((0, 0, -0.03)), lens=85, fstop=5.6,
             focus=mid + Vector((0, -0.012, 0)))

    d = MO.synthesize(a.input, fps=a.fps)
    n = len(d["t"])
    rig, tgt = S.rig(), S.target()
    rig["auto_lids"] = 1.0
    for i in range(n):
        key_frame(i + 1, d, i, rig, tgt)
    for fc_owner in (rig, tgt):
        ad = fc_owner.animation_data
        if ad and ad.action:
            for layer in getattr(ad.action, "layers", []):
                for strip in layer.strips:
                    for cb in strip.channelbags:
                        for fc in cb.fcurves:
                            for kp in fc.keyframe_points:
                                kp.interpolation = "LINEAR"
    sc.frame_start, sc.frame_end = 1, n
    sc.frame_set(1)
    if a.save_blend:
        # open in the camera view with material preview, ready to press Play in the Timeline
        for scr in bpy.data.screens:
            for area in scr.areas:
                if area.type == "VIEW_3D":
                    sp = area.spaces.active
                    sp.shading.type = "MATERIAL"
                    sp.region_3d.view_perspective = "CAMERA"
                    sp.overlay.show_relationship_lines = False
        sc.render.use_motion_blur = False
        path = os.path.abspath(a.save_blend)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=path)
        for img in bpy.data.images:                  # the studio HDRI lives inside Blender's install: pack it
            if "studiolights" in img.filepath and not img.packed_file:
                img.pack()
        bpy.ops.file.make_paths_relative()          # textures load from ../../textures wherever the repo lives
        bpy.ops.wm.save_mainfile()
        print("saved", a.save_blend, flush=True)
    if a.no_render:
        return
    out = a.frames_dir or os.path.join(os.path.dirname(HERE), "renders_eyes_cli", a.input, "frames")
    os.makedirs(out, exist_ok=True)
    sc.render.filepath = os.path.join(out, "####")
    bpy.ops.render.render(animation=True)
    print("frames", n, "->", out, flush=True)


if __name__ == "__main__":
    main()
