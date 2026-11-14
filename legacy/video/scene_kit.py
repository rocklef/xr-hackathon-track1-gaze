"""Shared helpers for the application scenes (Blender 5.2, EEVEE): cast import, HDRI, lights, cameras,
Google Scanned Objects GLB props, gaze aiming. Imported by video/scenes/*.py."""
import math
import os
import sys

import bpy
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "video", "build")
CAST_DIR = os.path.join(BUILD, "cast")
GSO_DIR = os.path.join(ROOT, "_downloads", "gso", "raw")
HDRI_DIR = os.path.join(os.path.dirname(bpy.app.binary_path), f"{bpy.app.version[0]}.{bpy.app.version[1]}",
                        "datafiles", "studiolights", "world")
sc = bpy.context.scene


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def keys(f, ks):
    if f <= ks[0][0]:
        return ks[0][1]
    for (f0, v0), (f1, v1) in zip(ks, ks[1:]):
        if f <= f1:
            k = ease((f - f0) / (f1 - f0))
            if isinstance(v0, (tuple, list, Vector)):
                return Vector(v0).lerp(Vector(v1), k)
            return v0 + (v1 - v0) * k
    return ks[-1][1]


def reset():
    """Empty the current scene. Run scene scripts with --factory-startup. NEVER call
    wm.read_factory_settings here: inside a session that loaded user prefs it disables all extensions and
    Blender's wheel sync deletes every extension wheel (it did once; a normal Blender start restores them)."""
    global sc
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    return sc


def eevee(res=(1920, 1080), samples=48, fps=30):
    s = bpy.context.scene
    s.render.engine = "BLENDER_EEVEE"
    try:
        s.eevee.taa_render_samples = samples
        s.eevee.use_raytracing = True
        s.eevee.use_shadows = True
    except AttributeError:
        pass
    s.render.resolution_x, s.render.resolution_y = res
    s.render.resolution_percentage = 100
    s.render.fps = fps
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    s.view_settings.view_transform = "AgX"
    try:
        s.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    s.view_settings.exposure = 0.0
    s.render.film_transparent = False


def world_hdri(name="interior.exr", strength=0.8, rot=0.0, bg_strength=None):
    w = bpy.data.worlds.new("WLD_" + name)
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(os.path.join(HDRI_DIR, name))
    mp = nt.nodes.new("ShaderNodeMapping")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp.inputs["Rotation"].default_value[2] = rot
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], env.inputs[0])
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    nt.links.new(env.outputs[0], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])
    return w


def area(name, loc, look_at, energy, size=1.0, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, "AREA")
    d.energy, d.size, d.color = energy, size, color
    d.shape = "DISK"
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    aim(o, look_at)
    return o


def aim(o, target, track="-Z"):
    d = Vector(target) - Vector(o.location)
    o.rotation_euler = d.to_track_quat(track, "Y").to_euler()


def camera(name, loc, look_at, lens=50, fstop=None, focus=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_start = 0.01
    if fstop:
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        if focus is not None:
            cd.dof.focus_object = focus
    o = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    aim(o, look_at)
    bpy.context.scene.camera = o
    return o


def empty(name, loc):
    o = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    return o


def append_cast(who):
    path = os.path.join(CAST_DIR, f"{who}.blend")
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = [c for c in src.collections if c == f"COL_{who}"]
    col = dst.collections[0]
    bpy.context.scene.collection.children.link(col)
    rig = bpy.data.objects[f"ARM_{who}"]
    tgt = bpy.data.objects[f"TGT_{who}"]
    lids_any_distance(rig, tgt)
    return rig, tgt


def lids_any_distance(rig, tgt, dist=0.35):
    """Lid drivers use the true pitch of the gaze: atan(z / (dist - y)) in the rest-anchor frame,
    so far targets (a road 8 m ahead) don't close the eyes."""
    if rig.animation_data is None:
        return
    for fc in rig.animation_data.drivers:
        if "orbicularis0" not in fc.data_path:
            continue
        drv = fc.driver
        k = drv.expression.split("*")[0].strip()
        if "y" not in [v.name for v in drv.variables]:
            v = drv.variables.new()
            v.name, v.type = "y", "SINGLE_PROP"
            v.targets[0].id_type, v.targets[0].id, v.targets[0].data_path = "OBJECT", tgt, "location[1]"
        drv.expression = f"{k} * atan(z / max({dist} - y, 0.05))"


def eye_mid(rig):
    """World position of the eye midpoint (eye.L/eye.R bone heads)."""
    bpy.context.view_layer.update()
    a = rig.matrix_world @ rig.pose.bones["eye.L"].head
    b = rig.matrix_world @ rig.pose.bones["eye.R"].head
    return (a + b) / 2


def look_at_world(tgt, world_point):
    """Place the gaze target (child of the rest anchor) at a world point."""
    bpy.context.view_layer.update()
    tgt.matrix_world.translation = Vector(world_point)


def key_gaze(tgt, frames_points):
    """frames_points: [(frame, world_point)] -> keyframe the target (smooth per frame)."""
    f0, f1 = frames_points[0][0], frames_points[-1][0]
    for f in range(f0, f1 + 1):
        p = keys(f, [(a, Vector(b)) for a, b in frames_points])
        bpy.context.view_layer.update()
        par = tgt.parent.matrix_world
        tgt.location = par.inverted() @ p
        tgt.keyframe_insert("location", frame=f)


def pose(rig, bone_rot):
    """bone_rot: {bone: (x, y, z) degrees} on the default MPFB rig (XYZ euler)."""
    for b, r in bone_rot.items():
        pb = rig.pose.bones.get(b)
        if pb is None:
            print("no bone", b)
            continue
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = [math.radians(v) for v in r]


def gso(name, loc, rot_z=0.0, scale=1.0):
    """Import one Google Scanned Objects GLB (meshopt-decoded copy) at real size; returns the root object."""
    path = os.path.join(GSO_DIR, f"{name}.glb")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    roots = [o for o in new if o.parent is None]
    root = bpy.data.objects.new(f"GSO_{name}", None)
    bpy.context.scene.collection.objects.link(root)
    for o in roots:
        o.parent = root
    # glTF is Y-up -> Blender importer already converts to Z-up; drop it so its lowest point sits at loc.z
    bpy.context.view_layer.update()
    zs = [(o.matrix_world @ Vector(c)).z for o in new if o.type == "MESH" for c in o.bound_box]
    root.location = Vector(loc) - Vector((0, 0, min(zs) if zs else 0))
    root.rotation_euler[2] = rot_z
    root.scale = (scale,) * 3
    return root


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


def pbr(name, color, rough=0.5, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    return m


def box(name, size, loc, mat, bevel=0.01, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        m = o.modifiers.new("Bevel", "BEVEL")
        m.width, m.segments = bevel, 3
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


def cycles(res=(1920, 1080), samples=24, fps=30):
    s = bpy.context.scene
    eevee(res, 16, fps)
    s.render.engine = "CYCLES"
    s.cycles.samples = samples
    s.cycles.use_denoising = True
    try:
        s.cycles.denoising_use_gpu = True  # denoise on the RTX card instead of the CPU
    except AttributeError:
        pass
    s.render.image_settings.compression = 5  # faster PNG writes
    s.render.use_persistent_data = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type != "CPU"
            s.cycles.device = "GPU"
            break
        except TypeError:
            continue


def render(out_dir, frames=None):
    s = bpy.context.scene
    s.render.filepath = os.path.join(out_dir, "####")
    if frames:
        s.frame_start, s.frame_end = frames
    if os.environ.get("XR_SAVE_BLEND"):
        # build-only mode: save the assembled scene to open in Blender instead of rendering it
        p = os.path.join(BUILD, "blend", os.path.basename(out_dir.rstrip("/\\")) + ".blend")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        s.frame_current = s.frame_start
        bpy.ops.wm.save_as_mainfile(filepath=p, copy=True)
        print(f"[scene] saved {p}", flush=True)
        return
    print(f"[scene] rendering {s.frame_start}-{s.frame_end} -> {out_dir}", flush=True)
    bpy.ops.render.render(animation=True)


def argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


# ------------------------------------------------------------------ posing (MPFB default rig)
def point_bone(rig, name, world_dir):
    """Rotate pose bone `name` so its +Y axis points along world_dir (keeps the head where it is)."""
    from mathutils import Matrix
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    M = rig.matrix_world @ pb.matrix
    cur = (M.to_3x3() @ Vector((0, 1, 0))).normalized()
    q = cur.rotation_difference(Vector(world_dir).normalized())
    R = q.to_matrix() @ M.to_3x3()
    new = Matrix.Translation(M.translation) @ R.to_4x4()
    pb.matrix = rig.matrix_world.inverted() @ new
    bpy.context.view_layer.update()


def arm(rig, side, upper_dir, fore_dir):
    """side 'L'/'R'. Directions in world space (the person faces -Y in their own frame)."""
    rot = rig.matrix_world.to_3x3()
    for b in (f"upperarm01.{side}", f"upperarm02.{side}"):
        point_bone(rig, b, rot @ Vector(upper_dir))
    for b in (f"lowerarm01.{side}", f"lowerarm02.{side}"):
        point_bone(rig, b, rot @ Vector(fore_dir))


def arms_down(rig):
    arm(rig, "L", (0.12, 0.05, -1.0), (0.05, -0.25, -1.0))
    arm(rig, "R", (-0.12, 0.05, -1.0), (-0.05, -0.25, -1.0))


def sit(rig):
    rot = rig.matrix_world.to_3x3()
    for s, x in (("L", 0.08), ("R", -0.08)):
        for b in (f"upperleg01.{s}", f"upperleg02.{s}"):
            point_bone(rig, b, rot @ Vector((x, -1.0, -0.05)))
        for b in (f"lowerleg01.{s}", f"lowerleg02.{s}"):
            point_bone(rig, b, rot @ Vector((0.0, -0.15, -1.0)))


def bone_world(rig, name, tail=False):
    bpy.context.view_layer.update()
    pb = rig.pose.bones[name]
    return rig.matrix_world @ (pb.tail if tail else pb.head)


def key_bone_rot(rig, name, frames_vals):
    """frames_vals [(frame, (x, y, z) degrees)] keyed on the pose bone (XYZ euler, eased between keys)."""
    pb = rig.pose.bones[name]
    pb.rotation_mode = "XYZ"
    f0, f1 = frames_vals[0][0], frames_vals[-1][0]
    for f in range(f0, f1 + 1, 2):
        v = keys(f, [(a, Vector(b)) for a, b in frames_vals])
        pb.rotation_euler = [math.radians(c) for c in v]
        pb.keyframe_insert("rotation_euler", frame=f)


# ------------------------------------------------------------------ props
def vr_headset(rig):
    """White VR headset with a black visor and a strap, parented to the head bone."""
    mid = eye_mid(rig)
    shell = pbr("MAT_HS_Shell", (0.86, 0.87, 0.89), 0.38, coat=0.4)
    visor = pbr("MAT_HS_Visor", (0.01, 0.012, 0.016), 0.08, coat=1.0)
    strap = pbr("MAT_HS_Strap", (0.09, 0.09, 0.1), 0.8)
    parts = [box("HS_Body", (0.185, 0.105, 0.085), mid + Vector((0, -0.075, 0.0)), shell, bevel=0.03),
             box("HS_Visor", (0.17, 0.088, 0.012), mid + Vector((0, -0.12, 0.0)), visor, bevel=0.012)]
    bpy.ops.mesh.primitive_torus_add(major_radius=0.1, minor_radius=0.011, location=mid + Vector((0, 0.02, 0.025)),
                                     rotation=(-0.18, 0, 0))
    t = bpy.context.active_object
    t.name = "HS_Strap"
    t.scale = (0.9, 1.2, 1.0)
    t.data.materials.append(strap)
    parts.append(t)
    led = box("HS_LED", (0.012, 0.004, 0.004), mid + Vector((0.06, -0.127, 0.03)), emissive("MAT_HS_LED", (0.3, 0.9, 1.0), 25), 0)
    parts.append(led)
    head = rig.pose.bones["head"]
    bpy.context.view_layer.update()
    for p in parts:
        mw = p.matrix_world.copy()
        p.parent = rig
        p.parent_type = "BONE"
        p.parent_bone = "head"
        p.matrix_world = mw
    return parts


def image_screen(name, w, h, loc, look_from, seq_dir, nframes, strength=1.6):
    """Emissive screen plane playing a PNG sequence (0001.png...), facing look_from."""
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = (w, h, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    face_toward(o, look_from)
    m = bpy.data.materials.new("MAT_" + name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(os.path.join(seq_dir, "0001.png"))
    img.source = "SEQUENCE"
    tex.image = img
    tex.image_user.frame_duration = nframes
    tex.image_user.frame_start = 1
    tex.image_user.use_auto_refresh = True
    em.inputs["Strength"].default_value = strength
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs[0])
    o.data.materials.append(m)
    return o


def face_toward(o, look_from):
    """Upright plane (local +Y up) whose front (+Z) faces look_from; texture reads correctly from there."""
    d = Vector(look_from) - Vector(o.location)
    o.rotation_euler = (math.radians(90), 0.0, math.atan2(d.x, -d.y))


def screen_point(o, u, v):
    """World point at UV (u right, v down, 0..1) on a 1x1 plane scaled by its object (applied scale)."""
    bpy.context.view_layer.update()
    xs = [c.co.x for c in o.data.vertices]
    ys = [c.co.y for c in o.data.vertices]
    x = min(xs) + (max(xs) - min(xs)) * u
    y = max(ys) - (max(ys) - min(ys)) * v
    return o.matrix_world @ Vector((x, y, 0.002))
