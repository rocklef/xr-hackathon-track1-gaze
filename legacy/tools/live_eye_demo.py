"""Run: blender scene.blend -P tools/live_eye_demo.py
Live eye-rotation demo for scene.blend (GUI only; nothing is saved).
Frames 1-120: the fixation target sweeps a clockwise circle (+-5 cm at 35 cm).
Frames 121-240: centre -> left -> right -> up -> down -> centre.
Each eye gets a glowing line of sight that follows its eye bone, and the target gets a red dot,
so you can watch both eyes rotate and converge."""
import math

import bmesh
import bpy
from mathutils import Vector

N1, N2 = 120, 240
STEPS = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]


def hv(f):
    if f <= N1:
        a = 2 * math.pi * (f - 1) / N1           # clockwise from the top
        return math.sin(a), math.cos(a)
    k = (f - N1 - 1) / (N2 - N1) * (len(STEPS) - 1)
    i = min(int(k), len(STEPS) - 2)
    t = k - i
    t = t * t * (3 - 2 * t)
    (h0, v0), (h1, v1) = STEPS[i], STEPS[i + 1]
    return h0 + (h1 - h0) * t, v0 + (v1 - v0) * t


def on_frame(scene, *_):
    tgt = bpy.data.objects.get("TGT_Fixation")
    if tgt:
        h, v = hv(scene.frame_current)
        tgt.location = (h * 0.05, 0.0, v * 0.05)


def glow(name, col, strength=6.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*col, 1)
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


def ray_mesh(name, length=0.36, r=0.0012):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=r, radius2=r, depth=length)
    for v in bm.verts:                        # cone is along Z, centred: move to +Y from the origin
        x, y, z = v.co
        v.co = Vector((x, z + length / 2, -y))
    bm.to_mesh(me)
    bm.free()
    return me


def setup():
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, N2
    bpy.app.handlers.frame_change_pre[:] = [h for h in bpy.app.handlers.frame_change_pre if h.__name__ != "on_frame"]
    bpy.app.handlers.frame_change_pre.append(on_frame)
    sc.frame_set(1)
    bpy.context.view_layer.update()

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and "eye.L" in o.pose.bones)
    tgt = bpy.data.objects["TGT_Fixation"]
    col = bpy.context.scene.collection
    heads = []
    for side, c in (("L", (0.2, 0.85, 1.0)), ("R", (1.0, 0.75, 0.2))):
        ray = bpy.data.objects.new(f"DEMO_Ray_{side}", ray_mesh(f"DEMO_Ray_{side}"))
        ray.data.materials.append(glow(f"DEMO_Glow_{side}", c))
        col.objects.link(ray)
        cn = ray.constraints.new("COPY_TRANSFORMS")
        cn.target, cn.subtarget = arm, f"eye.{side}"
        heads.append(arm.matrix_world @ arm.pose.bones[f"eye.{side}"].head)
    dot_me = bpy.data.meshes.new("DEMO_TargetDot")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.006)
    bm.to_mesh(dot_me)
    bm.free()
    dot = bpy.data.objects.new("DEMO_TargetDot", dot_me)
    dot.data.materials.append(glow("DEMO_Glow_Target", (1.0, 0.15, 0.1), 8.0))
    col.objects.link(dot)
    cn = dot.constraints.new("COPY_LOCATION")
    cn.target = tgt

    mid = (heads[0] + heads[1]) / 2
    tp = tgt.matrix_world.translation
    centre = mid + (tp - mid) * 0.45
    fwd = (tp - mid).normalized()
    side = fwd.cross(Vector((0, 0, 1))).normalized()
    campos = centre - side * 0.42 + Vector((0, 0, 0.16)) + fwd * 0.05
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                sp = a.spaces.active
                sp.shading.type = "MATERIAL"
                sp.overlay.show_overlays = True
                sp.overlay.show_bones = False
                sp.overlay.show_floor = False
                sp.overlay.show_axis_x = sp.overlay.show_axis_y = False
                sp.overlay.show_relationship_lines = False
                sp.show_region_ui = False
                sp.show_gizmo = False
                sp.lens = 50
                r3 = sp.region_3d
                r3.view_perspective = "PERSP"
                r3.view_location = centre
                r3.view_rotation = (centre - campos).to_track_quat("-Z", "Y")
                r3.view_distance = (centre - campos).length
    bpy.app.timers.register(play, first_interval=4.0)
    return None


def play():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                with bpy.context.temp_override(window=w, area=a):
                    if not bpy.context.screen.is_animation_playing:
                        bpy.ops.screen.animation_play()
                return None
    return None


bpy.app.timers.register(setup, first_interval=3.0)
