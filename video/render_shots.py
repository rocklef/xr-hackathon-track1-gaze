"""Render the 3D shots for the demo video (does NOT modify scene.blend).

    blender -b --factory-startup scene.blend -P video/render_shots.py -- --shot setup
    blender -b --factory-startup scene.blend -P video/render_shots.py -- --shot gaze
    (add --frames 1-10 --samples 8 for a quick preview)

Shots
  setup : orbit from front-3/4 to full side view. Shows the face, the glowing
          fixation target (35 cm), a stylised camera body at the locked render
          camera (40 cm), live gaze rays hooked to the eye bones, distance labels.
  gaze  : the two locked cameras (CAM_Gaze 1080x1080, CAM_GazeEyes 1080x540)
          while the target goes once round the circle clockwise, then
          Left / Right / Up / Down with holds. Writes gaze_track.json (h, v per frame).
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "video", "build")
sc = bpy.context.scene
cfg = sc["gaze_cfg"]
MID = Vector(cfg["eye_midpoint"])
B = float(cfg["motion_bound_m"])
TGT = bpy.data.objects["TGT_Fixation"]
RIG = bpy.data.objects["ARM_Human"]
FPS = 30


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def emissive(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


def link(ob, coll="COL_Video"):
    c = bpy.data.collections.get(coll)
    if c is None:
        c = bpy.data.collections.new(coll)
        sc.collection.children.link(c)
    c.objects.link(ob)
    return ob


def common_render(samples, res):
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.fps = FPS
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sys.path.insert(0, os.path.join(ROOT, "wrapper"))
    import eyes
    print("[video] device:", eyes.setup_device(), flush=True)


def key_target(track):
    """track: list of (h, v) per frame (1-based frames)."""
    TGT.animation_data_clear()
    for f, (h, v) in enumerate(track, start=1):
        TGT.location = (h * B, 0.0, v * B)
        TGT.keyframe_insert("location", frame=f)
    if TGT.animation_data and TGT.animation_data.action:
        for fc in getattr(TGT.animation_data.action, "fcurves", []):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    sc.frame_start, sc.frame_end = 1, len(track)


# ------------------------------------------------------------------ props
def build_props():
    # glowing target ball
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0055, segments=32, ring_count=16)
    ball = bpy.context.active_object
    ball.name = "VID_TargetBall"
    for c in list(ball.users_collection):
        c.objects.unlink(ball)
    link(ball)
    ball.data.materials.append(emissive("VID_Gold", (1.0, 0.72, 0.18), 12.0))
    ball.parent = TGT
    ball.matrix_parent_inverse.identity()
    ball.location = (0, 0, 0)

    # stylised camera body at the locked render camera
    cam = bpy.data.objects["CAM_Gaze"]
    body_mat = bpy.data.materials.new("VID_CamBody")
    body_mat.use_nodes = True
    bsdf = body_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.04, 0.045, 0.05, 1)
    bsdf.inputs["Metallic"].default_value = 0.6
    bsdf.inputs["Roughness"].default_value = 0.35
    bpy.ops.mesh.primitive_cube_add(size=1)
    box = bpy.context.active_object
    box.name = "VID_CamBox"
    box.scale = (0.045, 0.032, 0.05)              # camera-local: x right, y up, z back
    bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.03, vertices=48)
    lens = bpy.context.active_object
    lens.name = "VID_CamLens"
    lens.location = (0, 0, -0.01)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.0125, minor_radius=0.0012)
    ring = bpy.context.active_object
    ring.name = "VID_CamRing"
    ring.location = (0, 0, -0.0255)
    ring.data.materials.append(emissive("VID_Red", (1.0, 0.15, 0.1), 6.0))
    for o in (box, lens, ring):
        for c in list(o.users_collection):
            c.objects.unlink(o)
        link(o)
        if o is not ring:
            o.data.materials.append(body_mat)
        o.parent = cam
        o.matrix_parent_inverse.identity()
    box.location = (0, 0, 0.03)

    # gaze rays: one curve, two splines, hooked to the eye bones and the target
    cu = bpy.data.curves.new("VID_Rays", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.00075
    cu.bevel_resolution = 3
    rays = link(bpy.data.objects.new("VID_Rays", cu))
    rays.data.materials.append(emissive("VID_Orange", (1.0, 0.45, 0.08), 9.0))
    tw = TGT.matrix_world.translation
    for side in ("L", "R"):
        sp = cu.splines.new("POLY")
        sp.points.add(1)
        head = RIG.matrix_world @ RIG.pose.bones[f"eye.{side}"].head
        sp.points[0].co = (*head, 1)
        sp.points[1].co = (*tw, 1)
    bpy.context.view_layer.update()
    idx = 0
    for side in ("L", "R"):
        for pi, (obj, sub) in enumerate(((RIG, f"eye.{side}"), (TGT, ""))):
            hk = rays.modifiers.new(f"hook_{side}_{pi}", "HOOK")
            hk.object = obj
            if sub:
                hk.subtarget = sub
            # bind: inverse of the hook target's current matrix
            if sub:
                mat = obj.matrix_world @ obj.pose.bones[sub].matrix
            else:
                mat = obj.matrix_world
            hk.matrix_inverse = mat.inverted()
            hk.vertex_indices_set([idx])
            idx += 1

    # distance labels (face the shot camera via Track To)
    def label(text, loc, name):
        td = bpy.data.curves.new(name, "FONT")
        td.body = text
        td.size = 0.016
        td.align_x = "CENTER"
        ob = link(bpy.data.objects.new(name, td))
        ob.location = loc
        ob.data.materials.append(emissive(name + "_m", (0.92, 0.95, 1.0), 4.0))
        return ob
    l35 = label("fixation target · 35 cm  (±5 cm)", MID + Vector((0.0, -0.35, 0.072)), "VID_L35")
    l40 = label("locked render camera · 40 cm", MID + Vector((0.0, -0.36, -0.08)), "VID_L40")
    tshirt()
    return [l35, l40]


def tshirt():
    """Simple dark tee: shell copy of the torso below the neckline."""
    import bmesh
    body = bpy.data.objects["SM_Head_Body"]
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(body.evaluated_get(dg))
    bm = bmesh.new()
    bm.from_mesh(me)
    mw = body.matrix_world
    neck = MID.z - 0.155
    kill = [v for v in bm.verts if not ((mw @ v.co).z < neck - 0.012 * (1 - min(abs((mw @ v.co).x) / 0.07, 1))
                                         and abs((mw @ v.co).x) < 0.40 and (mw @ v.co).z > MID.z - 0.75)]
    bmesh.ops.delete(bm, geom=kill, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.materials.clear()
    tee = link(bpy.data.objects.new("VID_Tee", me))
    tee.matrix_world = mw
    so = tee.modifiers.new("Thick", "SOLIDIFY")
    so.thickness, so.offset = 0.004, 1.0
    tee.modifiers.new("Smooth", "SUBSURF").render_levels = 1
    m = bpy.data.materials.new("VID_TeeFabric")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.008, 0.009, 0.011, 1)
    b.inputs["Roughness"].default_value = 0.85
    b.inputs["Sheen Weight"].default_value = 0.08
    me.materials.append(m)


# ------------------------------------------------------------------ shots
def shot_setup(samples, frames):
    common_render(samples, (1920, 1080))
    labels = build_props()
    n = 330
    # target drifts once round the circle so the rays visibly converge/move
    key_target([(math.sin(2 * math.pi * i / n), math.cos(2 * math.pi * i / n)) for i in range(n)])

    pivot = link(bpy.data.objects.new("EMPTY_CamTarget_Setup", None))
    pivot.location = MID + Vector((0, -0.21, -0.01))
    cd = bpy.data.cameras.new("CAM_Setup_Orbit")
    cd.lens = 45
    cd.dof.use_dof = True
    cd.dof.focus_object = pivot
    cd.dof.aperture_fstop = 4.0
    cam = link(bpy.data.objects.new("CAM_Setup_Orbit", cd))
    tr = cam.constraints.new("TRACK_TO")
    tr.target, tr.track_axis, tr.up_axis = pivot, "TRACK_NEGATIVE_Z", "UP_Y"
    for lb in labels:
        c = lb.constraints.new("TRACK_TO")
        c.target, c.track_axis, c.up_axis = cam, "TRACK_Z", "UP_Y"
    # orbit: azimuth from 35 deg (front-3/4, camera-right side) to 92 deg (side)
    R, H = 0.84, 0.07
    for f in range(1, n + 1):
        t = ease((f - 1) / (n - 1))
        az = math.radians(35 + 57 * t)
        cam.location = pivot.location + Vector((R * math.sin(az), -R * math.cos(az) * 0.85 - 0.02, H - 0.03 * t))
        cam.keyframe_insert("location", frame=f)
    sc.camera = cam
    # stage lights only: keep the portrait rig, darken backdrop a bit
    out = os.path.join(BUILD, "setup", "####")
    render_frames(out, frames)


def gaze_track():
    T = []

    def hold(h, v, n):
        T.extend([(h, v)] * n)

    def move(a, b, n):
        for i in range(1, n + 1):
            t = ease(i / n)
            T.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    hold(0, 0, 12)
    move((0, 0), (0, 1), 12)
    n = 150
    for i in range(1, n + 1):          # one clockwise loop from the top
        th = 2 * math.pi * i / n
        T.append((math.sin(th), math.cos(th)))
    move((0, 1), (0, 0), 14)
    hold(0, 0, 8)
    for d in ((-1, 0), (1, 0), (0, 1), (0, -1)):   # LEFT RIGHT UP DOWN
        move((0, 0), d, 12)
        hold(*d, 20)
        move(d, (0, 0), 12)
        hold(0, 0, 6)
    hold(0, 0, 12)
    return T


def shot_gaze(samples, frames):
    common_render(samples, (1080, 1080))
    tshirt()
    track = gaze_track()
    key_target(track)
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, "gaze_track.json"), "w") as f:
        json.dump(track, f)
    sc.camera = bpy.data.objects["CAM_Gaze"]
    render_frames(os.path.join(BUILD, "gaze_face", "####"), frames)
    sc.camera = bpy.data.objects["CAM_GazeEyes"]
    sc.render.resolution_x, sc.render.resolution_y = 1080, 540
    render_frames(os.path.join(BUILD, "gaze_eyes", "####"), frames)


def render_frames(path, frames):
    sc.render.filepath = path
    if frames:
        a, b = frames
        sc.frame_start, sc.frame_end = a, b
    print(f"[video] rendering {sc.frame_start}-{sc.frame_end} -> {path}", flush=True)
    bpy.ops.render.render(animation=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", choices=["setup", "gaze"], required=True)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--frames", default=None, help="e.g. 1-10")
    a = ap.parse_args(argv)
    frames = tuple(int(x) for x in a.frames.split("-")) if a.frames else None
    if a.shot == "setup":
        shot_setup(a.samples, frames)
    else:
        shot_gaze(a.samples, frames)


main()
