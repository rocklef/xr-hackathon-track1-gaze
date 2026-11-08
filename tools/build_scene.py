"""Build scene.blend for XR Hackathon Track 1 (run once, needs the MPFB add-on).

    blender -b --addons bl_ext.user_default.mpfb -P tools/build_scene.py

Produces scene.blend in the project root. Evaluators do NOT need MPFB: the
saved .blend is self-contained (mesh, textures packed, armature, constraints).
"""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

from bl_ext.user_default.mpfb.services.humanservice import HumanService

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_BLEND = os.path.join(ROOT, "scene.blend")

# ---- calibration constants (metres) -------------------------------------
FIXATION_DIST = 0.35   # fixation target ahead of eye midpoint
CAMERA_DIST = 0.40     # locked render camera ahead of eye midpoint
MOTION_BOUND = 0.05    # +-5 cm target excursion for |h|,|v| = 1
LENS_MM = 85.0         # 36 mm sensor -> ~17 cm field of view at 40 cm
RES = 1024
LID_UPPER = 1.0        # upper lid follows the vertical gaze angle 1:1
LID_LOWER = 0.5        # lower lid follows 50%


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)


def collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def move_to(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def build_human():
    info = HumanService._create_default_human_info_dict()
    info.update(
        name="Human",
        rig="default",
        eyes="high-poly.mhclo",
        eyebrows="eyebrow001.mhclo",
        eyelashes="eyelashes01.mhclo",
        skin_mhmat="young_caucasian_male.mhmat",
        skin_material_type="ENHANCED_SSS",
        eyes_material_type="PROCEDURAL_EYES",
        hair="short02.mhclo",
    )
    settings = HumanService.get_default_deserialization_settings()
    settings["subdiv_levels"] = 1
    body = HumanService.deserialize_from_dict(info, settings)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    eyes = next(o for o in bpy.data.objects if o.type == "MESH" and o.name.endswith("high-poly"))
    rig.name, body.name, eyes.name = "ARM_Human", "SM_Head_Body", "SM_Eyes"
    for o in bpy.data.objects:
        if o.name.endswith("eyebrow001"):
            o.name = "SM_Eyebrows"
        elif o.name.endswith("short02"):
            o.name = "SM_Hair"
        elif o.name.endswith("eyelashes01"):
            o.name = "SM_Eyelashes"
    return body, rig, eyes


def fit_sphere(pts):
    """Least-squares sphere fit -> (center, radius)."""
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(sol[3] + c @ c)
    return Vector(c), r


def calibrate_eyes(rig, eyes):
    """Put each eye bone's head at its eyeball's true centre, point it straight
    forward (-Y), and give the eyeball 100% weight to that bone so it rotates
    rigidly about its centre."""
    mw = eyes.matrix_world
    co = np.array([tuple(mw @ v.co) for v in eyes.data.vertices])
    out = {}
    for side, bone in ((+1, "eye.L"), (-1, "eye.R")):
        idx = np.where(co[:, 0] * side > 0)[0]
        # The eyeball is a sclera sphere + a cornea bulge; fit on the back
        # half (sclera only) so the bulge doesn't bias the centre.
        ymid = np.median(co[idx, 1])
        back = idx[co[idx, 1] > ymid]
        c, r = fit_sphere(co[back])
        # Optical axis: direction from the sphere centre to the centroid of the
        # cornea bulge (vertices that stick out beyond the sclera sphere).
        # MakeHuman's eye is modelled slightly tilted, so this is NOT exactly -Y.
        rel = co[idx] - np.array(c)
        dist = np.linalg.norm(rel, axis=1)
        bulge = rel[dist > r * 1.02]
        axis = bulge.mean(0)
        axis /= np.linalg.norm(axis)
        # pupil depth = how far the front surface reaches along that axis
        depth = float((rel @ axis).max())
        print(f"EYECAL {bone}: r={r*1000:.2f}mm bulge={len(bulge)} axis={np.round(axis,4)} depth={depth*1000:.2f}mm")
        out[bone] = dict(center=c, radius=r, verts=idx.tolist(), axis=Vector(axis), depth=depth)

    # weights: eyeball verts -> 100% own eye bone, nothing else
    for g in list(eyes.vertex_groups):
        eyes.vertex_groups.remove(g)
    for bone, d in out.items():
        g = eyes.vertex_groups.new(name=bone)
        g.add(d["verts"], 1.0, "REPLACE")

    # bones
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    inv = rig.matrix_world.inverted()
    for bone, d in out.items():
        eb = rig.data.edit_bones[bone]
        eb.use_connect = False
        eb.head = inv @ d["center"]
        eb.tail = inv @ (d["center"] + d["axis"] * 0.025)  # bone +Y = line of sight
        eb.roll = 0.0
    bpy.ops.object.mode_set(mode="OBJECT")
    return out


def main():
    clear_scene()
    scene = bpy.context.scene
    col_char = collection("COL_Character")
    col_rig = collection("COL_Rig")
    col_light = collection("COL_Lights")

    body, rig, eyes = build_human()
    for o in list(bpy.data.objects):
        move_to(o, col_char)

    eyecal = calibrate_eyes(rig, eyes)
    cL, cR = eyecal["eye.L"]["center"], eyecal["eye.R"]["center"]
    mid = (cL + cR) / 2
    fwd, right, up = Vector((0, -1, 0)), Vector((1, 0, 0)), Vector((0, 0, 1))
    # "right" = image right as seen by the camera (= subject's left, +X)

    # fixation target + its rest anchor
    anchor = bpy.data.objects.new("EMPTY_FixationRest", None)
    anchor.location = mid + fwd * FIXATION_DIST
    anchor.empty_display_type = "ARROWS"
    anchor.empty_display_size = 0.02
    col_rig.objects.link(anchor)
    target = bpy.data.objects.new("TGT_Fixation", None)
    target.empty_display_type = "SPHERE"
    target.empty_display_size = 0.008
    target.parent = anchor
    col_rig.objects.link(target)
    lim = target.constraints.new("LIMIT_LOCATION")  # hard +-5 cm bounds
    for ax in "xyz":
        setattr(lim, f"use_min_{ax}", True)
        setattr(lim, f"use_max_{ax}", True)
        setattr(lim, f"min_{ax}", -MOTION_BOUND if ax != "y" else 0.0)
        setattr(lim, f"max_{ax}", MOTION_BOUND if ax != "y" else 0.0)
    lim.owner_space = "LOCAL"

    for bone in ("eye.L", "eye.R"):
        pb = rig.pose.bones[bone]
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        con = pb.constraints.new("DAMPED_TRACK")
        con.name = "Fixation"
        con.target = target
        con.track_axis = "TRACK_Y"

    # lid-gaze coupling: eyelids follow the vertical gaze angle (pitch) like
    # real lids do, so looking up doesn't bury the pupil under the upper lid.
    # Driver: lid_pitch = k * atan(target_z / fixation_dist). Only the vertical
    # component is used, so horizontal gaze / convergence never moves the lids.
    # (Lid bones' local +X points to world -X, so +rotation_x raises the lid.)
    for side in ("L", "R"):
        for lid, k in ((f"orbicularis03.{side}", LID_UPPER), (f"orbicularis04.{side}", LID_LOWER)):
            pb = rig.pose.bones[lid]
            for c in list(pb.constraints):
                pb.constraints.remove(c)
            pb.rotation_mode = "XYZ"
            pb.driver_remove("rotation_euler", 0)
            fc = pb.driver_add("rotation_euler", 0)
            drv = fc.driver
            drv.type = "SCRIPTED"
            var = drv.variables.new()
            var.name = "z"
            var.type = "SINGLE_PROP"
            var.targets[0].id_type = "OBJECT"
            var.targets[0].id = target
            var.targets[0].data_path = "location[2]"
            drv.expression = f"{k} * atan(z / {FIXATION_DIST})"  # simple expr: no auto-exec needed

    # camera, locked
    cam_data = bpy.data.cameras.new("CAM_Gaze")
    cam_data.lens = LENS_MM
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.01
    cam = bpy.data.objects.new("CAM_Gaze", cam_data)
    cam.location = mid + fwd * CAMERA_DIST
    cam.rotation_euler = (math.radians(90), 0, 0)  # looks along +Y at the face
    cam.lock_location = (True, True, True)
    cam.lock_rotation = (True, True, True)
    col_rig.objects.link(cam)
    scene.camera = cam

    # constant studio lighting (no animation, fixed values)
    def area(name, loc, energy, size):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy
        ld.size = size
        ld.shape = "DISK"
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        d = (mid - Vector(loc))
        lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        col_light.objects.link(lo)
    area("LGT_Key", mid + Vector((0.3, -0.55, 0.55)), 30, 0.6)
    area("LGT_Fill", mid + Vector((-0.5, -0.45, 0.3)), 12, 0.8)
    area("LGT_Rim", mid + Vector((0.0, 0.5, 0.4)), 20, 0.5)
    world = bpy.data.worlds.new("WLD_Studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.36, 0.38, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
    scene.world = world

    # render settings
    r = scene.render
    r.resolution_x = r.resolution_y = RES
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.film_transparent = False
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 256
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.frame_set(1)

    # hide helpers from render
    for o in (anchor, target):
        o.hide_render = True

    # calibration record used by the wrapper
    scene["gaze_cfg"] = {
        "fixation_dist_m": FIXATION_DIST,
        "camera_dist_m": CAMERA_DIST,
        "motion_bound_m": MOTION_BOUND,
        "eye_midpoint": list(mid),
        "eye_center_L": list(cL),
        "eye_center_R": list(cR),
        "eye_radius_L": eyecal["eye.L"]["radius"],
        "eye_radius_R": eyecal["eye.R"]["radius"],
        "pupil_depth_L": eyecal["eye.L"]["depth"],
        "optical_axis_rest_L": list(eyecal["eye.L"]["axis"]),
        "pupil_depth_R": eyecal["eye.R"]["depth"],
        "optical_axis_rest_R": list(eyecal["eye.R"]["axis"]),
        "axes": {"forward": [0, -1, 0], "right_image": [1, 0, 0], "up": [0, 0, 1]},
    }

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND, compress=True)
    print("SAVED", OUT_BLEND)
    print("CAL", dict(scene["gaze_cfg"]))


main()
