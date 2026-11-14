"""Build the GNM head in Blender: mesh with GNM's UDIM UVs, 4-bone rig (neck, head, eye.L, eye.R) with GNM's
skinning weights, semantic expression shape keys solved in GNM's expression space (expressions.py), a gaze
target the eyes track (Damped Track), eyelids that follow gaze automatically, and Timeline-friendly controls.

blender -b --factory-startup -P gnm_rig/build_head.py -- [--identity zeros|seed:N|tex:T01] [--out gnm_rig/out/gnm_head.blend]

Controls (custom properties on the rig object GNM_Rig, all keyframeable in the Timeline):
  blink_L, blink_R   0..1   eyelid closure
  wide, squint       0..1   both eyes
  brow_raise, brow_down 0..1
  pupil             -1..2   iris/pupil size (GNM's pupil component)
  auto_lids          0..1   how much the lids follow vertical gaze
Gaze: move the empty GAZE_Target (child of GAZE_Rest, 35 cm in front of the eyes); both eyes converge on it.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expressions as X  # noqa: E402
import gnm_model as G  # noqa: E402

FIX_DIST = 0.35


def argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def identity_vector(m, spec):
    ident = np.zeros(m.B_id.shape[0])
    if spec.startswith("seed:"):
        rng = np.random.default_rng(int(spec.split(":")[1]))
        head = [i for i, n in enumerate(m.id_names) if n.startswith("head_")]
        ident[head] = rng.normal(0, 0.7, len(head)) * np.linspace(1.0, 0.4, len(head))
    elif spec.startswith("tex:"):            # GNM's semantic identity sampler, matched to a texture set
        import identities
        ident[:] = np.clip(identities.for_texture(spec.split(":")[1]), -4.0, 4.0)
    elif os.path.exists(spec):
        ident[:] = np.load(spec)
    return ident


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def make_mesh(m, ident):
    v, j = m.bind(ident, np.zeros(m.B_ex.shape[0]))
    vb = G.to_blender(v)
    me = bpy.data.meshes.new("GNM_Head")
    me.from_pydata(vb.tolist(), [], m.quads.tolist())
    me.update()
    uv, reg = m.quad_uvs_udim()
    layer = me.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", uv.reshape(-1).astype(np.float32))   # loops follow quad vertex order
    names = ["skin", "eye_interior_L", "cornea_L", "eye_interior_R", "cornea_R", "teeth", "tongue"]
    key = {"skin": 0, "left_eye_interior": 1, "left_eye_exterior": 2, "right_eye_interior": 3,
           "right_eye_exterior": 4, "upper_teeth_and_gums": 5, "lower_teeth_and_gums": 5, "tongue": 6}
    me.polygons.foreach_set("material_index", np.array([key[r] for r in reg], dtype=np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    ob = bpy.data.objects.new("GNM_Head", me)
    bpy.context.scene.collection.objects.link(ob)
    for n in names:
        ob.data.materials.append(bpy.data.materials.get(f"MAT_{n}") or bpy.data.materials.new(f"MAT_{n}"))
    # GNM vertex groups (regions) are useful for texturing; keep the ones we use
    for gname in ("skin", "upper_lip", "lower_lip", "nose_region", "left_cheek_region", "right_cheek_region",
                  "left_brow_region", "right_brow_region", "middle_brow_region", "forehead_region", "ears",
                  "left_orbital_region", "right_orbital_region", "chin_region", "eye_sockets", "mouth_sock",
                  "left_infraorbital_region", "right_infraorbital_region", "upper_lip_region", "lower_lip_region",
                  "left_zygomatic_region", "right_zygomatic_region", "left_temple_region", "right_temple_region",
                  "hockey_mask"):
        w = m.groups[gname]
        vg = ob.vertex_groups.new(name=f"R_{gname}")
        nz = np.where(w > 1e-4)[0]
        for i in nz:
            vg.add([int(i)], float(w[i]), "REPLACE")
    return ob, j, v


def make_rig(m, joints_g, head_ob):
    jb = G.to_blender(joints_g)
    arm = bpy.data.armatures.new("GNM_Rig")
    rig = bpy.data.objects.new("GNM_Rig", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    neck = eb.new("neck")
    neck.head, neck.tail = Vector(jb[0]), Vector(jb[1])
    head = eb.new("head")
    head.head, head.tail = Vector(jb[1]), Vector(jb[1]) + Vector((0, 0, 0.14))
    head.parent = neck
    for name, k in (("eye.L", 2), ("eye.R", 3)):
        e = eb.new(name)
        e.head, e.tail = Vector(jb[k]), Vector(jb[k]) + Vector((0, -0.024, 0))   # +Y of the bone = line of sight
        e.roll = 0.0
        e.parent = head
    bpy.ops.object.mode_set(mode="OBJECT")
    for name, k in (("neck", 0), ("head", 1), ("eye.L", 2), ("eye.R", 3)):
        vg = head_ob.vertex_groups.new(name=name)
        w = m.W[k]
        for i in np.where(w > 1e-5)[0]:
            vg.add([int(i)], float(w[i]), "REPLACE")
    head_ob.parent = rig
    mod = head_ob.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    return rig, jb


def make_gaze(rig, jb):
    mid = Vector(((jb[2] + jb[3]) / 2).tolist())
    rest = bpy.data.objects.new("GAZE_Rest", None)
    rest.empty_display_type, rest.empty_display_size = "PLAIN_AXES", 0.01
    bpy.context.scene.collection.objects.link(rest)
    rest.parent, rest.parent_type, rest.parent_bone = rig, "BONE", "head"
    bpy.context.view_layer.update()
    rest.matrix_world = Matrix.Translation(mid)
    tgt = bpy.data.objects.new("GAZE_Target", None)
    tgt.empty_display_type, tgt.empty_display_size = "SPHERE", 0.006
    bpy.context.scene.collection.objects.link(tgt)
    tgt.parent = rest
    tgt.location = (0.0, -FIX_DIST, 0.0)
    for b in ("eye.L", "eye.R"):
        c = rig.pose.bones[b].constraints.new("DAMPED_TRACK")
        c.target, c.track_axis = tgt, "TRACK_Y"
    return rest, tgt, mid


def add_controls(rig):
    spec = {"blink_L": (0, 0, 1), "blink_R": (0, 0, 1), "wide": (0, 0, 1), "squint": (0, 0, 1),
            "brow_raise": (0, 0, 1), "brow_down": (0, 0, 1), "pupil": (0, -1, 2), "auto_lids": (1, 0, 1)}
    for k, (val, lo, hi) in spec.items():
        rig[k] = float(val)
        ui = rig.id_properties_ui(k)
        ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi)


def add_shape_keys(m, head_ob, rig, tgt):
    shapes = X.load(m)
    head_ob.shape_key_add(name="Basis", from_mix=False)
    keys = {}
    for name, e in shapes.items():
        d = G.to_blender(np.tensordot(e, m.B_ex, 1))
        sk = head_ob.shape_key_add(name=name.replace("_left", "_L").replace("_right", "_R"), from_mix=False)
        co = np.empty(len(sk.data) * 3)
        head_ob.data.shape_keys.key_blocks["Basis"].data.foreach_get("co", co)
        sk.data.foreach_set("co", (co.reshape(-1, 3) + d).reshape(-1).astype(np.float32))
        sk.slider_min, sk.slider_max = -1.0, 2.0
        keys[sk.name] = sk

    def drive(sk_name, expr, props=(), use_pitch=False):
        fc = keys[sk_name].driver_add("value")
        drv = fc.driver
        drv.type = "SCRIPTED"
        for p in props:
            v = drv.variables.new()
            v.name, v.type = p, "SINGLE_PROP"
            v.targets[0].id_type, v.targets[0].id, v.targets[0].data_path = "OBJECT", rig, f'["{p}"]'
        if use_pitch:
            for nm, ch in (("ty", "LOC_Y"), ("tz", "LOC_Z")):
                v = drv.variables.new()
                v.name, v.type = nm, "TRANSFORMS"
                v.targets[0].id, v.targets[0].transform_type, v.targets[0].transform_space = tgt, ch, "LOCAL_SPACE"
        drv.expression = expr

    pitch = "atan(tz/max(0.05,-ty))"
    for s in ("L", "R"):
        drive(f"blink_{s}", f"blink_{s}", [f"blink_{s}"])
        drive(f"lidsUp_{s}", f"auto_lids*max(0,min(1.3,{pitch}/0.40))*(1-blink_{s})", ["auto_lids", f"blink_{s}"], True)
        drive(f"lidsDown_{s}", f"auto_lids*max(0,min(1.2,-{pitch}/0.40))*(1-blink_{s})", ["auto_lids", f"blink_{s}"], True)
        drive(f"wide_{s}", f"wide*(1-blink_{s})", ["wide", f"blink_{s}"])
        drive(f"squint_{s}", f"squint*(1-blink_{s})", ["squint", f"blink_{s}"])
        drive(f"browRaise_{s}", "brow_raise", ["brow_raise"])
        drive(f"browDown_{s}", "brow_down", ["brow_down"])
    drive("pupil", "pupil", ["pupil"])


def make_lashes(m, head_ob, eye_centers, seed=7):
    """Eyelashes: tapered, curled 3-sided strands rooted on GNM's eyelid margins, bound to the head surface
    with Surface Deform so they follow blinks, lid-follow and the head rig."""
    rng = np.random.default_rng(seed)
    marg = X.lid_margins(m)
    verts, faces = [], []
    fwd = Vector((0, -1, 0))
    for side, ec in (("left", eye_centers[0]), ("right", eye_centers[1])):
        ec = Vector(ec)
        for upper, (count, base_len, curl) in ((True, (120, 0.0098, 0.85)), (False, (34, 0.0042, 0.25))):
            idx = marg[side][0] if upper else marg[side][1]
            pts = np.array([np.array(head_ob.data.vertices[int(i)].co) for i in idx])
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg)])
            clump_dir = None
            for k in range(count):
                if k % 4 == 0:                            # lashes gather into small clumps
                    clump_dir = Vector(rng.normal(0, 0.18, 3))
                u = (k + rng.uniform(0.1, 0.9)) / count
                d = u * cum[-1]
                j = min(int(np.searchsorted(cum, d)) - 1, len(pts) - 2)
                j = max(j, 0)
                f = (d - cum[j]) / max(seg[j], 1e-9)
                root = Vector(pts[j] * (1 - f) + pts[j + 1] * f)
                prof = math.sin(math.pi * min(max(u, 0.04), 0.96)) ** 0.6       # longer in the middle
                length = base_len * (0.45 + 0.55 * prof) * rng.uniform(0.7, 1.15)
                out = (root - ec).normalized()
                up = Vector((0, 0, 1 if upper else -1))
                nasal = abs(root.x) < abs(ec.x)
                if not upper:
                    if nasal and abs(abs(root.x) - abs(ec.x)) > 0.004 and rng.random() < 0.85:
                        continue                         # few lower lashes at the inner corner
                    d0 = (out * 0.6 + fwd * 0.55 + up * 0.55 + Vector(rng.normal(0, 0.2, 3))).normalized()
                else:
                    d0 = (out * 0.45 + fwd * 0.75 + up * 0.25 + clump_dir * (0.6 if upper else 0)).normalized()
                # outer lashes sweep sideways
                sweep = Vector(((1 if root.x > ec.x else -1) * 0.35 * abs(root.x - ec.x) / 0.012, 0, 0))
                d0 = (d0 + sweep * 0.5).normalized()
                n_seg, r0 = 7, (0.000075 if upper else 0.00005) * rng.uniform(0.75, 1.25)
                ring_prev, p, dvec = None, root - out * 0.0004, d0.copy()
                axis_c = d0.cross(up).normalized() if d0.cross(up).length > 1e-6 else Vector((1, 0, 0))
                for sgi in range(n_seg + 1):
                    tt = sgi / n_seg
                    r = r0 * (1 - 0.85 * tt)
                    side_v = axis_c
                    norm_v = dvec.cross(side_v).normalized()
                    ring = []
                    for a_ in range(3):
                        ang = 2 * math.pi * a_ / 3
                        ring.append(len(verts))
                        verts.append(tuple(p + (side_v * math.cos(ang) + norm_v * math.sin(ang)) * r))
                    if ring_prev:
                        for a_ in range(3):
                            b_ = (a_ + 1) % 3
                            faces.append((ring_prev[a_], ring_prev[b_], ring[b_], ring[a_]))
                    ring_prev = ring
                    # advance and curl (rotate the direction towards 'up')
                    p = p + dvec * (length / n_seg)
                    dvec = (dvec + up * curl * (0.9 / n_seg) * (0.5 + tt)).normalized()
    me = bpy.data.meshes.new("GNM_Lashes")
    me.from_pydata(verts, [], faces)
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    ob = bpy.data.objects.new("GNM_Lashes", me)
    bpy.context.scene.collection.objects.link(ob)
    mat = bpy.data.materials.get("MAT_lashes") or bpy.data.materials.new("MAT_lashes")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.012, 0.009, 0.008, 1)
    b.inputs["Roughness"].default_value = 0.4
    me.materials.append(mat)
    ob.parent = head_ob.parent
    sd = ob.modifiers.new("SurfaceDeform", "SURFACE_DEFORM")
    sd.target = head_ob
    bpy.context.view_layer.update()
    with bpy.context.temp_override(object=ob, active_object=ob):
        bpy.ops.object.surfacedeform_bind(modifier=sd.name)
    return ob


def make_tearline(m, head_ob, eye_centers):
    """Tear meniscus: a thin wet strip where each lid margin meets the eyeball. It catches sharp highlights,
    which is one of the strongest cues that an eye is real. Bound with Surface Deform so it follows blinks."""
    marg = X.lid_margins(m)
    verts, faces = [], []
    for side, ec in (("left", eye_centers[0]), ("right", eye_centers[1])):
        ec = Vector(ec)
        for upper in (True, False):
            idx = marg[side][0] if upper else marg[side][1]
            pts = [Vector(head_ob.data.vertices[int(i)].co) for i in idx]
            n_sides, base = 8, len(verts)
            for k, p in enumerate(pts):
                t = k / (len(pts) - 1)
                radius = 0.00016 * max(0.0, math.sin(math.pi * min(max((t - 0.08) / 0.84, 0), 1))) ** 1.5 * (0.45 if upper else 1.0)
                inward = (ec - p).normalized()
                c = p + inward * 0.0006 + Vector((0, 0, -0.00010 if upper else 0.00008))
                tang = (pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]).normalized()
                a = tang.cross(inward).normalized()
                b = tang.cross(a).normalized()
                for j in range(n_sides):
                    ang = 2 * math.pi * j / n_sides
                    verts.append(tuple(c + (a * math.cos(ang) + b * math.sin(ang)) * radius))
            for k in range(len(pts) - 1):
                for j in range(n_sides):
                    j2 = (j + 1) % n_sides
                    faces.append((base + k * n_sides + j, base + k * n_sides + j2,
                                  base + (k + 1) * n_sides + j2, base + (k + 1) * n_sides + j))
    me = bpy.data.meshes.new("GNM_TearLine")
    me.from_pydata(verts, [], faces)
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    ob = bpy.data.objects.new("GNM_TearLine", me)
    bpy.context.scene.collection.objects.link(ob)
    mat = bpy.data.materials.get("MAT_tearline") or bpy.data.materials.new("MAT_tearline")
    me.materials.append(mat)
    ob.parent = head_ob.parent
    sd = ob.modifiers.new("SurfaceDeform", "SURFACE_DEFORM")
    sd.target = head_ob
    bpy.context.view_layer.update()
    with bpy.context.temp_override(object=ob, active_object=ob):
        bpy.ops.object.surfacedeform_bind(modifier=sd.name)
    return ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", default="zeros")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "gnm_head.blend"))
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv())
    clear()
    m = G.GNM()
    ident = identity_vector(m, a.identity)
    head_ob, joints, _ = make_mesh(m, ident)
    rig, jb = make_rig(m, joints, head_ob)
    rest, tgt, mid = make_gaze(rig, jb)
    add_controls(rig)
    add_shape_keys(m, head_ob, rig, tgt)
    sub = head_ob.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 1       # equal levels: Surface Deform (lashes) needs a stable target
    sub.uv_smooth = "PRESERVE_BOUNDARIES"
    make_lashes(m, head_ob, (jb[2], jb[3]))
    make_tearline(m, head_ob, (jb[2], jb[3]))
    bpy.context.scene["gnm_eye_mid"] = list(mid)
    bpy.context.scene["gnm_identity"] = a.identity
    if not a.no_save:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=a.out)
        print("saved", a.out)


if __name__ == "__main__":
    main()
