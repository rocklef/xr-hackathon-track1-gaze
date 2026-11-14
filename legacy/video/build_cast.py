"""Build the video's cast: diverse, clothed MakeHuman (MPFB) people, each with the SAME calibrated eye rig
as scene.blend (eyeball centre from a sphere fit, bone along the optical axis, Damped Track to a target,
lid-gaze drivers). Used only for the demo video's application scenes.

    blender -b -P video/build_cast.py -- --who maya        (needs user prefs for MPFB: no --factory-startup)
    -> video/build/cast/maya.blend
"""
import argparse
import os
import sys

import bpy
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_scene as bs  # noqa: E402  (main() is guarded; we reuse calibrate_eyes)
from bl_ext.user_default.mpfb.services.humanservice import HumanService  # noqa: E402

CAST = {
    # East Asian woman, ~25 - VR headset scene
    "maya": dict(phen=dict(gender=0.0, age=0.5, muscle=0.45, weight=0.45, height=0.45,
                           race=dict(asian=0.9, caucasian=0.1, african=0.0)),
                 skin="young_asian_female.mhmat", hair="braid01.mhclo", brows="eyebrow010.mhclo",
                 lashes="eyelashes02.mhclo", clothes=["female_elegantsuit01.mhclo", "shoes02.mhclo"],
                 tint=(0.93, 0.80, 0.68), iris=((0.20, 0.11, 0.05), (0.07, 0.04, 0.02))),
    # South Asian man, ~30 - driver
    "arjun": dict(phen=dict(gender=1.0, age=0.55, muscle=0.6, weight=0.5, height=0.55,
                            race=dict(asian=0.35, caucasian=0.45, african=0.2)),
                  skin="young_caucasian_male.mhmat", hair="short04.mhclo", brows="eyebrow001.mhclo",
                  lashes="eyelashes01.mhclo", clothes=["male_casualsuit01.mhclo", "shoes01.mhclo"],
                  tint=(0.50, 0.34, 0.23), iris=((0.16, 0.09, 0.04), (0.05, 0.03, 0.015))),
    # older Caucasian man, ~75 - types with his eyes
    "arthur": dict(phen=dict(gender=1.0, age=0.92, muscle=0.4, weight=0.55, height=0.5,
                             race=dict(asian=0.0, caucasian=1.0, african=0.0)),
                   skin="old_caucasian_male.mhmat", hair="short02.mhclo", brows="eyebrow005.mhclo",
                   lashes="eyelashes01.mhclo", clothes=["male_casualsuit05.mhclo", "shoes03.mhclo"],
                   tint=(0.95, 0.84, 0.76), iris=((0.30, 0.40, 0.48), (0.10, 0.14, 0.18)), grey_hair=True),
    # African girl, ~7 - tablet eye test
    "zuri": dict(phen=dict(gender=0.0, age=0.12, muscle=0.5, weight=0.5, height=0.5,
                           race=dict(asian=0.0, caucasian=0.0, african=1.0)),
                 skin="young_african_female.mhmat", hair="afro01.mhclo", brows="eyebrow009.mhclo",
                 lashes="eyelashes02.mhclo", clothes=["male_casualsuit06.mhclo", "shoes05.mhclo"],
                 tint=None, iris=((0.14, 0.08, 0.035), (0.05, 0.03, 0.015))),
}


def set_inputs(node, values):
    for k, v in values.items():
        if k in node.inputs:
            node.inputs[k].default_value = v


def lookdev(who, c, body, eyes):
    """Same treatment as tools/lookdev.py: multiply skin tint, darker irises, optional grey hair."""
    g = next((n for n in body.material_slots[0].material.node_tree.nodes if n.type == "GROUP"), None)
    if g is not None and c.get("tint"):
        mix = g.node_tree.nodes.get("Mix")
        if mix is not None:
            mix.blend_type = "MULTIPLY"
        set_inputs(g, {"colorMixIn": (*c["tint"], 1.0), "colorMixInStrength": 1.0, "Roughness": 0.5,
                       "SSS strength": 0.18, "SSS radius scale": 0.07})
    ge = next((n for n in eyes.material_slots[0].material.node_tree.nodes if n.type == "GROUP"), None)
    if ge is not None:
        set_inputs(ge, {"EyeWhiteColor": (0.86, 0.82, 0.78, 1.0), "IrisMajorColor": (*c["iris"][0], 1.0),
                        "IrisMinorColor": (*c["iris"][1], 1.0), "Clearcoat": 0.8, "Clearcoat Roughness": 0.02})
    if c.get("grey_hair"):
        hair = next(o for o in bpy.data.objects if o.name.endswith(c["hair"].replace(".mhclo", "")))
        for slot in hair.material_slots:
            nt = slot.material.node_tree
            for n in [n for n in nt.nodes if n.type == "TEX_IMAGE"]:
                for l in list(n.outputs["Color"].links):
                    hsv = nt.nodes.new("ShaderNodeHueSaturation")
                    hsv.inputs["Saturation"].default_value = 0.12
                    hsv.inputs["Value"].default_value = 2.2
                    nt.links.new(n.outputs["Color"], hsv.inputs["Color"])
                    nt.links.new(hsv.outputs["Color"], l.to_socket)


def build(who):
    c = CAST[who]
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    info = HumanService._create_default_human_info_dict()
    info["phenotype"].update({k: v for k, v in c["phen"].items() if k != "race"})
    info["phenotype"]["race"] = c["phen"]["race"]
    info.update(name=who, rig="default", eyes="high-poly.mhclo", eyebrows=c["brows"], eyelashes=c["lashes"],
                skin_mhmat=c["skin"], skin_material_type="ENHANCED_SSS", eyes_material_type="PROCEDURAL_EYES",
                hair=c["hair"], clothes=list(c["clothes"]))
    settings = HumanService.get_default_deserialization_settings()
    settings["subdiv_levels"] = 1
    body = HumanService.deserialize_from_dict(info, settings)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    eyes = next(o for o in bpy.data.objects if o.type == "MESH" and o.name.endswith("high-poly"))
    lookdev(who, c, body, eyes)
    rig.name, body.name, eyes.name = f"ARM_{who}", f"SM_{who}_Body", f"SM_{who}_Eyes"
    col = bpy.data.collections.new(f"COL_{who}")
    bpy.context.scene.collection.children.link(col)
    for o in list(bpy.data.objects):
        for uc in list(o.users_collection):
            uc.objects.unlink(o)
        col.objects.link(o)
        if o.type == "MESH" and not o.name.startswith("SM_"):
            o.name = f"SM_{who}_{o.name.split('.')[-1]}"

    cal = bs.calibrate_eyes(rig, eyes)
    mid = (cal["eye.L"]["center"] + cal["eye.R"]["center"]) / 2
    anchor = bpy.data.objects.new(f"EMPTY_{who}_GazeRest", None)
    anchor.location = mid + Vector((0, -bs.FIXATION_DIST, 0))
    col.objects.link(anchor)
    tgt = bpy.data.objects.new(f"TGT_{who}", None)
    tgt.empty_display_type = "SPHERE"
    tgt.empty_display_size = 0.01
    tgt.parent = anchor
    col.objects.link(tgt)
    for bone in ("eye.L", "eye.R"):
        pb = rig.pose.bones[bone]
        for k in list(pb.constraints):
            pb.constraints.remove(k)
        con = pb.constraints.new("DAMPED_TRACK")
        con.target, con.track_axis = tgt, "TRACK_Y"
    for side in ("L", "R"):
        for lid, k in ((f"orbicularis03.{side}", bs.LID_UPPER), (f"orbicularis04.{side}", bs.LID_LOWER)):
            pb = rig.pose.bones[lid]
            pb.rotation_mode = "XYZ"
            fc = pb.driver_add("rotation_euler", 0)
            drv = fc.driver
            drv.type = "SCRIPTED"
            v = drv.variables.new()
            v.name, v.type = "z", "SINGLE_PROP"
            v.targets[0].id_type, v.targets[0].id, v.targets[0].data_path = "OBJECT", tgt, "location[2]"
            drv.expression = f"{k} * atan(z / {bs.FIXATION_DIST})"
    # the gaze anchor travels with the person when the rig object is moved / rotated
    anchor.parent = rig
    anchor.matrix_parent_inverse = rig.matrix_world.inverted()
    rig["cast"] = who
    rig["eye_mid"] = list(mid)
    out = os.path.join(ROOT, "video", "build", "cast", f"{who}.blend")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("CAST SAVED", out, [o.name for o in col.objects])


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--who", default="all")
a = ap.parse_args(argv)
for w in (CAST if a.who == "all" else [a.who]):
    build(w)
