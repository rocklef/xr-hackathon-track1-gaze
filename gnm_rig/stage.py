"""Shared render stage for the GNM head: camera, lights, world, render settings and helpers to pose the rig.
Imported by the look tests, the eye CLI (eyes_cli.py) and the texture renders."""
import math
import os

import bpy
from mathutils import Vector

STUDIO = None  # resolved lazily: Blender's bundled studio HDRIs


def hdri_dir():
    base = os.path.dirname(bpy.app.binary_path)
    for root, dirs, files in os.walk(base):
        if root.endswith(os.path.join("studiolights", "world")):
            return root
    return None


def cycles(res=(1920, 1080), samples=64, fps=30):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.render.resolution_x, s.render.resolution_y = res
    s.render.resolution_percentage = 100
    s.render.fps = fps
    s.cycles.samples = samples
    s.cycles.use_denoising = True
    try:
        s.cycles.denoising_use_gpu = True
    except AttributeError:
        pass
    s.render.use_persistent_data = True
    # eyes: refraction through cornea + tear film needs deeper transmission paths
    s.cycles.max_bounces = 16
    s.cycles.transmission_bounces = 16
    s.cycles.glossy_bounces = 8
    s.cycles.transparent_max_bounces = 32
    s.cycles.caustics_refractive = True
    s.cycles.blur_glossy = 0.3
    s.view_settings.view_transform = "AgX"
    s.view_settings.look = "AgX - Base Contrast"
    s.render.image_settings.compression = 5
    prefs = bpy.context.preferences.addons["cycles"].preferences
    # XR_CYCLES_DEVICE=CUDA when OptiX kernels fail on an older NVIDIA driver (default: try OptiX, then CUDA)
    for dev in os.environ.get("XR_CYCLES_DEVICE", "OPTIX,CUDA").split(","):
        try:
            prefs.compute_device_type = dev
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type == dev
            s.cycles.device = "GPU"
            break
        except TypeError:
            continue


def world(name="studio.exr", strength=0.6, rot=0.0):
    w = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    nt = w.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    d = hdri_dir()
    if d and os.path.exists(os.path.join(d, name)):
        env.image = bpy.data.images.load(os.path.join(d, name), check_existing=True)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value[2] = rot
    tc = nt.nodes.new("ShaderNodeTexCoord")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs[0])


def aim(o, target):
    d = Vector(target) - o.location
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def area(name, loc, target, power, size, color=(1, 1, 1), shape="SQUARE"):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size, ld.color = power, size, color
    ld.shape = shape
    o = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(o)
    o.location = Vector(loc)
    aim(o, target)
    return o


def camera(name, loc, target, lens=85, fstop=None, focus=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36
    cd.clip_start = 0.01
    o = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(o)
    o.location = Vector(loc)
    aim(o, target)
    if fstop:
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        cd.dof.focus_distance = (Vector(focus or target) - Vector(loc)).length
    bpy.context.scene.camera = o
    return o


def eye_mid():
    return Vector(bpy.context.scene["gnm_eye_mid"])


def portrait_lights(mid, warm=True):
    """Soft key from front-left, fill right, two cool rims, as in a beauty portrait."""
    k = (1.0, 0.95, 0.9) if warm else (1, 1, 1)
    area("KEY", mid + Vector((-0.45, -0.70, 0.45)), mid, 16, 0.55, k, shape="DISK")      # soft key, above-left
    area("FILL", mid + Vector((0.75, -0.50, -0.05)), mid, 2.2, 1.2)        # gentle fill (about 1:4)
    area("RIM", mid + Vector((0.55, 0.50, 0.35)), mid, 10, 0.3, (0.85, 0.92, 1.0))
    area("CATCH", mid + Vector((0.0, -0.9, 0.12)), mid, 1.2, 0.18, shape="DISK")         # small frontal catchlight


def backdrop(mid, color=(0.03, 0.035, 0.045)):
    bpy.ops.mesh.primitive_plane_add(size=4, location=mid + Vector((0, 0.6, 0)), rotation=(math.radians(90), 0, 0))
    p = bpy.context.object
    p.name = "SET_Backdrop"
    mat = bpy.data.materials.new("MAT_Backdrop")
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    p.data.materials.append(mat)
    return p


# ---------------------------------------------------------------- rig posing
def rig():
    return bpy.data.objects["GNM_Rig"]


def target():
    return bpy.data.objects["GAZE_Target"]


def set_gaze(h_deg, v_deg, dist=0.35):
    """Point both eyes at a target dist metres away. Viewer's frame: h_deg > 0 = toward screen right
    (the subject's left), v_deg > 0 = up."""
    t = target()
    x = dist * math.tan(math.radians(h_deg))
    z = dist * math.tan(math.radians(v_deg))
    # GAZE_Rest axes = world axes at rest: -Y is forward (toward the camera), +X is screen right
    t.location = (x, -dist, z)


def key_gaze(frame, h_deg, v_deg, dist=0.35, interp="BEZIER"):
    set_gaze(h_deg, v_deg, dist)
    target().keyframe_insert("location", frame=frame)


def refresh():
    """Custom-property edits do not tag the depsgraph; force drivers to re-evaluate."""
    rig().update_tag()
    sc = bpy.context.scene
    sc.frame_set(sc.frame_current)
    bpy.context.view_layer.update()


def set_ctrl(**kw):
    r = rig()
    for k, v in kw.items():
        r[k] = float(v)
    refresh()


def key_ctrl(frame, **kw):
    r = rig()
    for k, v in kw.items():
        r[k] = float(v)
        r.keyframe_insert(f'["{k}"]', frame=frame)
