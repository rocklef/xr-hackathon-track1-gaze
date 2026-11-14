"""Look development pass: portrait lighting, backdrop, grading, skin/eye
shading and strand hair. Runs on scene.blend after build_scene.py and before
calibrate_pupil.py:

    blender -b --factory-startup scene.blend -P tools/lookdev.py

Everything is constant (no animation), so the "constant lighting" rule holds.
"""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scene = bpy.context.scene
cfg = scene["gaze_cfg"]
MID = Vector(cfg["eye_midpoint"])
SEED = 7


def col(name):
    return bpy.data.collections.get(name) or bpy.data.collections.new(name)


COL_LIGHT = col("COL_Lights")
COL_SET = col("COL_Set")
if COL_SET.name not in scene.collection.children:
    scene.collection.children.link(COL_SET)


# ---------------------------------------------------------------- lighting
def lights():
    for o in list(COL_LIGHT.objects):
        bpy.data.objects.remove(o, do_unlink=True)

    def area(name, off, energy, size, color=(1, 1, 1), shape="DISK", aim=None):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.shape, ld.color = energy, size, shape, color
        lo = bpy.data.objects.new(name, ld)
        lo.location = MID + Vector(off)
        tgt = aim if aim is not None else MID
        lo.rotation_euler = (tgt - lo.location).to_track_quat("-Z", "Y").to_euler()
        COL_LIGHT.objects.link(lo)
        return lo

    # camera is on -Y. Key: camera-right & above (Rembrandt-ish, soft)
    area("LGT_Key", (0.42, -0.36, 0.52), 34.0, 0.30, (1.0, 0.95, 0.88))
    area("LGT_Fill", (-0.60, -0.30, -0.12), 2.5, 0.6, (0.9, 0.94, 1.0))
    area("LGT_Rim", (-0.30, 0.40, 0.38), 14.0, 0.35, (0.95, 0.97, 1.0))
    area("LGT_Kicker", (0.40, 0.30, 0.06), 5.0, 0.30, (0.85, 0.9, 1.0))
    # soft light on the backdrop for a gradient halo behind the head
    area("LGT_Backdrop", (0.0, 0.25, -0.05), 1.6, 0.6, (0.8, 0.9, 1.0),
         aim=MID + Vector((0, 0.75, 0.05)))


def backdrop():
    for o in list(COL_SET.objects):
        if o.name.startswith("SM_Backdrop"):
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=MID + Vector((0, 0.75, 0)))
    bd = bpy.context.active_object
    bd.name = "SM_Backdrop"
    bd.rotation_euler = (math.radians(90), 0, 0)
    for c in list(bd.users_collection):
        c.objects.unlink(bd)
    COL_SET.objects.link(bd)
    m = bpy.data.materials.new("MAT_Backdrop")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.03, 0.038, 0.048, 1)
    b.inputs["Roughness"].default_value = 0.9
    bd.data.materials.append(m)
    bd.visible_shadow = False


def world():
    w = scene.world
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    path = os.path.join(os.path.dirname(bpy.app.binary_path), f"{bpy.app.version[0]}.{bpy.app.version[1]}",
                        "datafiles", "studiolights", "world", "studio.exr")
    env.image = bpy.data.images.load(path, check_existing=True)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.12
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    # camera rays see a dark backdrop colour; everything else sees the HDRI
    dark = nt.nodes.new("ShaderNodeBackground")
    dark.inputs["Color"].default_value = (0.03, 0.035, 0.04, 1)
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(bg.outputs[0], mix.inputs[1])
    nt.links.new(dark.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])


def grading():
    vs = scene.view_settings
    vs.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Punchy"):
        try:
            vs.look = look
            break
        except TypeError:
            continue
    vs.exposure = -0.55
    scene.cycles.samples = 256
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.film_transparent = False


# ---------------------------------------------------------------- shading
def set_inputs(node, values):
    for k, v in values.items():
        if k in node.inputs:
            node.inputs[k].default_value = v


def skin():
    m = bpy.data.objects["SM_Head_Body"].material_slots[0].material
    g = next(n for n in m.node_tree.nodes if n.type == "GROUP")
    # tint by MULTIPLY (keeps texture detail) -> warm medium-brown skin tone
    mix = g.node_tree.nodes.get("Mix")
    if mix is not None:
        mix.blend_type = "MULTIPLY"
    set_inputs(g, {
        "colorMixIn": (0.80, 0.66, 0.56, 1.0), "colorMixInStrength": 1.0,
        "Roughness": 0.5, "Contrast": 0.1, "Brightness": -0.03,
        "Clearcoat": 0.03, "Clearcoat Roughness": 0.45,
        "Pore scale": 2600.0, "Pore detail": 3.0, "Pore strength": 0.38,
        "SSS strength": 0.18, "SSS radius scale": 0.07,
    })


def eyes():
    m = bpy.data.objects["SM_Eyes"].material_slots[0].material
    g = next(n for n in m.node_tree.nodes if n.type == "GROUP")
    set_inputs(g, {
        "EyeWhiteColor": (0.86, 0.82, 0.78, 1.0),
        "IrisMajorColor": (0.42, 0.26, 0.11, 1.0),
        "IrisMinorColor": (0.16, 0.09, 0.035, 1.0),
        "IrisSection4Color": (0.05, 0.03, 0.015, 1.0),
        "Clearcoat": 0.8, "Clearcoat Roughness": 0.02,
        "IrisBumpStrength": 0.45,
    })


def brows_lashes():
    for name, color in (("SM_Eyebrows", (0.06, 0.04, 0.03)), ("SM_Eyelashes", (0.02, 0.015, 0.012))):
        m = bpy.data.objects[name].material_slots[0].material
        mix = m.node_tree.nodes.get("diffuseIntensity")
        if mix and len(mix.inputs) > 2:
            # tint the card texture darker/richer
            mix.inputs[0].default_value = 0.6
            mix.inputs[2].default_value = (*color, 1.0)


# ---------------------------------------------------------------- strand hair
def strand_hair():
    """Buzz-cut: ~70k short Cycles hair strands grown from the hair-cap mesh,
    with density taken from the cap texture's alpha (natural hairline).
    The cap itself becomes a dark 'scalp shadow' layer under the strands."""
    cap = bpy.data.objects["SM_Hair"]
    dg = bpy.context.evaluated_depsgraph_get()
    ev = cap.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    mw = ev.matrix_world
    nm = mw.to_3x3().inverted().transposed()
    uvl = me.uv_layers.active.data
    co = np.array([tuple(mw @ v.co) for v in me.vertices])
    tris = np.array([t.vertices[:] for t in me.loop_triangles])
    tloops = np.array([t.loops[:] for t in me.loop_triangles])
    uvs = np.array([tuple(uvl[i].uv) for i in range(len(uvl))])
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    fn = np.cross(b - a, c - a)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    ev.to_mesh_clear()

    # hair-coverage mask from the cap texture alpha
    mat = cap.material_slots[0].material
    img = next(n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image)
    W, H = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)
    alpha = px[:, :, 3]

    # Grow strands from the SKULL (body mesh) where the cap covers it, so the
    # cut hugs the head instead of following the cap's puffy volume.
    from mathutils.bvhtree import BVHTree
    cap_bvh = BVHTree.FromPolygons([tuple(v) for v in co], [tuple(t) for t in tris])
    body = bpy.data.objects["SM_Head_Body"]
    bev = body.evaluated_get(dg)
    bme = bev.to_mesh()
    bme.calc_loop_triangles()
    bmw = bev.matrix_world
    bco = np.array([tuple(bmw @ v.co) for v in bme.vertices])
    btr = np.array([t.vertices[:] for t in bme.loop_triangles])
    bev.to_mesh_clear()
    ba, bb, bc = bco[btr[:, 0]], bco[btr[:, 1]], bco[btr[:, 2]]
    cen = (ba + bb + bc) / 3
    # Sculpted male hairline around the skull, as a function of azimuth
    # (0 = front, pi = back) -> minimum height above the eye line.
    sk = np.array([0.0, MID.y + 0.075, MID.z])
    def zmin(a):
        A = [0.00, 0.45, 0.85, 1.10, 1.30, 1.40, 1.46, 1.95, 2.25, 2.70, math.pi]
        Z = [0.046, 0.049, 0.057, 0.050, 0.020, 0.000, 0.042, 0.042, -0.030, -0.065, -0.072]
        return np.interp(a, A, Z)
    rel_c = cen - sk
    a_c = np.abs(np.arctan2(rel_c[:, 0], -rel_c[:, 1]))
    cand = np.where(rel_c[:, 2] > zmin(a_c) - 0.01)[0]
    ba, bb, bc = ba[cand], bb[cand], bc[cand]
    barea = np.linalg.norm(np.cross(bb - ba, bc - ba), axis=1) / 2
    bfn = np.cross(bb - ba, bc - ba)
    bfn /= np.linalg.norm(bfn, axis=1, keepdims=True) + 1e-12

    rng = np.random.default_rng(SEED)
    N = 480000
    pick = rng.choice(len(barea), size=N, p=barea / barea.sum())
    r1, r2 = rng.random(N), rng.random(N)
    s = np.sqrt(r1)
    P = ba[pick] * (1 - s)[:, None] + bb[pick] * (s * (1 - r2))[:, None] + bc[pick] * (s * r2)[:, None]
    Nn = bfn[pick].copy()
    rel = P - sk
    phi = np.arctan2(rel[:, 0], -rel[:, 1])
    az = np.abs(phi)
    # soft, slightly irregular (and not mirror-symmetric) hairline edge
    wob = 0.0016 * np.sin(phi * 19.0 + 1.3) + 0.001 * np.sin(phi * 7.0 + 0.4) + rng.normal(0, 0.0014, N)
    dens = np.clip((rel[:, 2] - zmin(az) + wob) / 0.007, 0, 1)
    dens[np.abs(rel[:, 0]) > 0.071] = 0          # never on the ears
    keep = rng.random(N) < dens
    P, Nn = P[keep], Nn[keep]
    head_c = np.array(MID) + np.array([0, 0.035, 0.03])
    flip = ((P - head_c) * Nn).sum(1) < 0
    Nn[flip] *= -1

    # comb direction: back over the top, down on the sides/back
    up = P - head_c
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    side = np.clip(1.0 - up[:, 2] * 1.3, 0, 1)
    comb = np.c_[rng.normal(0, 0.18, len(P)), np.full(len(P), 1.0), -0.15 - 0.9 * side]
    comb -= Nn * (comb * Nn).sum(1, keepdims=True)
    comb /= np.linalg.norm(comb, axis=1, keepdims=True) + 1e-9
    L = (0.005 + 0.008 * (1 - side)) * rng.uniform(0.75, 1.15, len(P))
    jitter = rng.normal(0, 0.25, (len(P), 3))
    PTS = 5
    pos = np.zeros((len(P), PTS, 3))
    for i in range(PTS):
        t = i / (PTS - 1)
        d = Nn * (0.35 - 0.25 * t) + comb * (0.8 + 0.6 * t) + jitter * 0.25
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pos[:, i] = P + Nn * 0.0004 if i == 0 else pos[:, i - 1] + d * (L / (PTS - 1))[:, None]

    old = bpy.data.objects.get("SM_HairStrands")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    cv = bpy.data.hair_curves.new("SM_HairStrands")
    cv.add_curves([PTS] * len(P))
    cv.position_data.foreach_set("vector", pos.reshape(-1).astype(np.float32))
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    rr = np.repeat(np.linspace(0.00005, 0.000018, PTS)[None, :], len(P), 0).reshape(-1)
    rad.data.foreach_set("value", rr.astype(np.float32))
    ob = bpy.data.objects.new("SM_HairStrands", cv)
    bpy.data.collections["COL_Character"].objects.link(ob)

    hm = bpy.data.materials.new("MAT_HairStrands")
    hm.use_nodes = True
    nt = hm.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    hb = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    hb.parametrization = "MELANIN"
    hb.inputs["Melanin"].default_value = 0.985
    hb.inputs["Melanin Redness"].default_value = 0.08
    hb.inputs["Roughness"].default_value = 0.38
    hb.inputs["Radial Roughness"].default_value = 0.45
    hb.inputs["Random Roughness"].default_value = 0.2
    if "Random Color" in hb.inputs:
        hb.inputs["Random Color"].default_value = 0.15
    nt.links.new(hb.outputs[0], out.inputs["Surface"])
    cv.materials.append(hm)

    # cap -> dark scalp-shadow layer (alpha from original texture)
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            for l in list(n.inputs["Base Color"].links):
                mat.node_tree.links.remove(l)
            n.inputs["Base Color"].default_value = (0.018, 0.013, 0.01, 1)
            n.inputs["Roughness"].default_value = 0.85
    cap.hide_render = True          # strands replace the card hair
    cap.hide_set(True)
    print(f"[lookdev] hair strands: {len(P)}")


def eyes_camera():
    """Second locked camera at the SAME 40 cm position, long lens on the eyes."""
    main_cam = bpy.data.objects["CAM_Gaze"]
    old = bpy.data.objects.get("CAM_GazeEyes")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    cd = bpy.data.cameras.new("CAM_GazeEyes")
    cd.lens, cd.sensor_fit, cd.sensor_width, cd.clip_start = 140.0, "HORIZONTAL", 36.0, 0.01
    cam = bpy.data.objects.new("CAM_GazeEyes", cd)
    cam.matrix_world = main_cam.matrix_world.copy()
    cam.lock_location = cam.lock_rotation = (True, True, True)
    for c in main_cam.users_collection:
        c.objects.link(cam)
    cfg["eyes_camera"] = {"name": "CAM_GazeEyes", "res": [1024, 512]}


def main():
    eyes_camera()
    lights()
    backdrop()
    world()
    grading()
    skin()
    eyes()
    strand_hair()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_mainfile(compress=True)
    print("[lookdev] saved")


main()
