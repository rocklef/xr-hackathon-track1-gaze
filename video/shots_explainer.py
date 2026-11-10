"""Explainer shots for the demo video (imported by video/render_shots.py).

hero : dolly in on the digital human while the eyes look around, then settle on the lens.
xray : skin fades to a ghost, wireframe eyeballs spin about their true centres, optical axes grow.
"""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector


def ease_keys(ease, f, keys):
    """keys: [(frame, value)], smoothstep between keys."""
    if f <= keys[0][0]:
        return keys[0][1]
    for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
        if f <= f1:
            return v0 + (v1 - v0) * ease((f - f0) / (f1 - f0))
    return keys[-1][1]


def xray_materials(objs):
    """Insert Mix(original, Transparent) before each material output; returns [(object name, mix node)]."""
    mixes, seen = [], set()
    for ob in objs:
        for slot in getattr(ob, "material_slots", []):
            m = slot.material
            if m is None or m.name in seen or not m.use_nodes:
                continue
            seen.add(m.name)
            nt = m.node_tree
            out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
            if out is None or not out.inputs["Surface"].is_linked:
                continue
            src = out.inputs["Surface"].links[0].from_socket
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            mx = nt.nodes.new("ShaderNodeMixShader")
            mx.name = "XRAY_Mix"
            nt.links.new(src, mx.inputs[1])
            nt.links.new(tr.outputs[0], mx.inputs[2])
            nt.links.new(mx.outputs[0], out.inputs["Surface"])
            mx.inputs[0].default_value = 0.0
            mixes.append((ob.name, mx))
    return mixes


def _sphere(name, radius, u=24, v=14):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=radius)
    bm.to_mesh(me)
    bm.free()
    return me


def _axis_mesh(name, length, radius):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=radius, radius2=radius, depth=length)
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(-90), 3, "X"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, length / 2, 0))
    bm.to_mesh(me)
    bm.free()
    return me


def shot_hero(ctx, samples, frames):
    sc, MID, ease, link, BUILD = ctx["sc"], ctx["MID"], ctx["ease"], ctx["link"], ctx["BUILD"]
    ctx["common_render"](samples, (1920, 1080))
    ctx["tshirt"]()
    n = 160
    hk = [(1, 0.0), (22, 0.0), (42, -0.85), (62, -0.85), (82, 0.85), (100, 0.85), (118, 0.0), (160, 0.0)]
    vk = [(1, 0.0), (82, 0.0), (100, 0.35), (118, 0.0), (160, 0.0)]
    ctx["key_target"]([(ease_keys(ease, f, hk), ease_keys(ease, f, vk)) for f in range(1, n + 1)])
    pivot = link(bpy.data.objects.new("EMPTY_CamTarget_Hero", None))
    cd = bpy.data.cameras.new("CAM_Hero")
    cd.lens = 85
    cd.dof.use_dof = True
    cd.dof.focus_object = pivot
    cd.dof.aperture_fstop = 2.8
    cam = link(bpy.data.objects.new("CAM_Hero", cd))
    tr = cam.constraints.new("TRACK_TO")
    tr.target, tr.track_axis, tr.up_axis = pivot, "TRACK_NEGATIVE_Z", "UP_Y"
    for f in range(1, n + 1):
        t = ease((f - 1) / (n - 1))
        pivot.location = MID + Vector((0, 0, -0.02 + 0.02 * t))
        pivot.keyframe_insert("location", frame=f)
        az = math.radians(-14 + 14 * t)
        R = 0.85 - 0.33 * t
        cam.location = pivot.location + Vector((R * math.sin(az), -R * math.cos(az), 0.05 - 0.04 * t))
        cam.keyframe_insert("location", frame=f)
    sc.camera = cam
    sc.frame_start, sc.frame_end = 1, n
    ctx["render_frames"](os.path.join(BUILD, "hero", "####"), frames)


def shot_xray(ctx, samples, frames):
    sc, MID, ease, link, BUILD, RIG, cfg = ctx["sc"], ctx["MID"], ctx["ease"], ctx["link"], ctx["BUILD"], ctx["RIG"], ctx["cfg"]
    ctx["common_render"](samples, (1920, 1080))
    ctx["tshirt"]()
    n = 232
    r = float(cfg["eyeball_radius_m"]) if "eyeball_radius_m" in cfg else 0.01509
    hk = [(1, 0.0), (80, 0.0), (118, -0.9), (172, 0.9), (212, 0.0), (232, 0.0)]
    ctx["key_target"]([(ease_keys(ease, f, hk), 0.0) for f in range(1, n + 1)])

    names = ("SM_Head_Body", "SM_Eyebrows", "SM_Eyelashes", "SM_Hair", "SM_HairStrands", "VID_Tee")
    objs = [bpy.data.objects[nm] for nm in names if nm in bpy.data.objects]
    for nm, mx in xray_materials(objs):
        top = 0.72 if nm in ("SM_Head_Body", "VID_Tee") else 1.0
        for f, v in ((1, 0.0), (24, 0.0), (56, top), (n, top)):
            mx.inputs[0].default_value = v
            mx.inputs[0].keyframe_insert("default_value", frame=f)

    em = ctx["emissive"]
    cyan = em("VID_Cyan", (0.25, 0.85, 1.0), 6.0)
    gold = em("VID_GoldAxis", (1.0, 0.72, 0.18), 10.0)
    white = em("VID_White", (1.0, 1.0, 1.0), 14.0)
    for side in ("L", "R"):
        ball = link(bpy.data.objects.new(f"VID_Ball_{side}", _sphere(f"VID_Ball_{side}", r * 1.02)))
        wf = ball.modifiers.new("Wire", "WIREFRAME")
        wf.thickness = 0.00028
        ball.data.materials.append(cyan)
        cen = link(bpy.data.objects.new(f"VID_Centre_{side}", _sphere(f"VID_Centre_{side}", 0.0011, 16, 10)))
        cen.data.materials.append(white)
        axis = link(bpy.data.objects.new(f"VID_Axis_{side}", _axis_mesh(f"VID_Axis_{side}", 0.085, 0.00045)))
        axis.data.materials.append(gold)
        for ob in (ball, cen, axis):
            c = ob.constraints.new("CHILD_OF")
            c.target, c.subtarget = RIG, f"eye.{side}"
            c.set_inverse_pending = False
            c.inverse_matrix.identity()
        # appear on "centre" (balls, dots) and "optical axis" (axes)
        for ob, keys in ((ball, [(1, 0.0), (52, 0.0), (72, 1.0)]), (cen, [(1, 0.0), (60, 0.0), (74, 1.0)])):
            for f, v in keys:
                ob.scale = (max(v, 1e-4),) * 3
                ob.keyframe_insert("scale", frame=f)
        for f, v in ((1, 1e-4), (150, 1e-4), (182, 1.0)):
            axis.scale = (1.0, v, 1.0) if v > 1e-3 else (1e-4, 1e-4, 1e-4)
            axis.keyframe_insert("scale", frame=f)

    pivot = link(bpy.data.objects.new("EMPTY_CamTarget_Eye", None))
    pivot.location = MID + Vector((0.006, -0.004, -0.002))
    cd = bpy.data.cameras.new("CAM_Eye_XRay")
    cd.lens = 70
    cd.dof.use_dof = True
    cd.dof.focus_object = pivot
    cd.dof.aperture_fstop = 16
    cam = link(bpy.data.objects.new("CAM_Eye_XRay", cd))
    tr = cam.constraints.new("TRACK_TO")
    tr.target, tr.track_axis, tr.up_axis = pivot, "TRACK_NEGATIVE_Z", "UP_Y"
    for f in range(1, n + 1):
        t = ease((f - 1) / (n - 1))
        az = math.radians(26 + 20 * t)
        R = 0.25 - 0.03 * t
        cam.location = pivot.location + Vector((R * math.sin(az), -R * math.cos(az), 0.03 - 0.01 * t))
        cam.keyframe_insert("location", frame=f)
    sc.camera = cam
    sc.frame_start, sc.frame_end = 1, n
    sc.cycles.transparent_max_bounces = 32
    ctx["render_frames"](os.path.join(BUILD, "xray", "####"), frames)
