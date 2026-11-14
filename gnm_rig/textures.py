"""Bake GNM-UV texture sets (UDIM): skin albedo / normal / roughness on tile 1001, iris+sclera albedo on 1002 (left
eye) and 1004 (right eye), plus simple teeth (1006) and tongue (1007) tiles.

blender -b --factory-startup gnm_rig/out/gnm_head.blend -P gnm_rig/textures.py -- [--ids T01,T02] [--size 4096]
(eye, teeth and tongue tiles are written by `python gnm_rig/eye_tiles.py`, which needs Pillow)

Every skin layer is generated in 3D object space on the GNM surface and baked through GNM's own UVs, so the
maps are continuous across UV seams by construction. Region masks come from GNM vertex groups (skin_masks.py).
Output: textures/<ID>_<name>/<ID>_{albedo,normal,roughness}.<UDIM>.png  (+ preview.png)
"""
import argparse
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gnm_model as G  # noqa: E402
import skin_masks as SM  # noqa: E402
from presets import PRESETS, lin  # noqa: E402

OUT = os.path.join(G.ROOT, "textures")


# ------------------------------------------------------------------ node helpers
class NT:
    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def n(self, kind, **inputs):
        node = self.nt.nodes.new(kind)
        node.location = (self.x, 0)
        self.x += 40
        for k, v in inputs.items():
            if k.startswith("_"):
                setattr(node, k[1:], v)
            else:
                self.set(node, k, v)
        return node

    def set(self, node, key, v):
        sock = node.inputs[key]
        if hasattr(v, "bl_idname") or (isinstance(v, tuple) and len(v) == 2 and hasattr(v[0], "bl_idname")):
            out = v if not isinstance(v, tuple) else v[0].outputs[v[1]]
            if hasattr(out, "outputs"):
                out = out.outputs[0]
            self.nt.links.new(out, sock)
        else:
            sock.default_value = v

    # math sugar ------------------------------------------------------------
    def math(self, op, a, b=0.0, c=None, clamp=False):
        m = self.n("ShaderNodeMath", _operation=op, _use_clamp=clamp)
        self._in(m, 0, a)
        self._in(m, 1, b)
        if c is not None:
            self._in(m, 2, c)
        return m

    def _in(self, node, idx, v):
        if isinstance(v, (int, float)):
            node.inputs[idx].default_value = v
        else:
            out = v[0].outputs[v[1]] if isinstance(v, tuple) else v.outputs[0]
            self.nt.links.new(out, node.inputs[idx])

    def mix(self, a, b, fac, blend="MIX"):
        m = self.n("ShaderNodeMix", _data_type="RGBA", _blend_type=blend)
        self._in(m, 0, fac)
        for idx, v in ((6, a), (7, b)):
            if isinstance(v, tuple) and len(v) in (3, 4) and isinstance(v[0], float):
                m.inputs[idx].default_value = (*v[:3], 1.0)
            else:
                self._in(m, idx, v)
        return (m, 2)

    def smooth(self, lo, hi, v):
        m = self.n("ShaderNodeMapRange", _interpolation_type="SMOOTHSTEP")
        self._in(m, 0, v)
        m.inputs[1].default_value, m.inputs[2].default_value = lo, hi
        m.inputs[3].default_value, m.inputs[4].default_value = 0.0, 1.0
        return m

    def attr(self, name):
        a = self.n("ShaderNodeAttribute", _attribute_name=name, _attribute_type="GEOMETRY")
        return (a, "Fac")


def build_skin_material(mat, P):
    """Returns NT with nodes: albedo colour socket, roughness value, height value."""
    t = NT(mat)
    tc = t.n("ShaderNodeTexCoord")
    pos = (tc, "Object")

    def noise(scale, detail=4.0, rough=0.5, vec=pos):
        nz = t.n("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough)
        t.nt.links.new(vec[0].outputs[vec[1]] if isinstance(vec, tuple) else vec.outputs[0], nz.inputs["Vector"])
        return (nz, "Fac")

    def voronoi(scale, vec=pos, rand=1.0):
        v = t.n("ShaderNodeTexVoronoi", Scale=scale, Randomness=rand)
        t.nt.links.new(vec[0].outputs[vec[1]] if isinstance(vec, tuple) else vec.outputs[0], v.inputs["Vector"])
        sep = t.n("ShaderNodeSeparateColor")
        t.nt.links.new(v.outputs["Color"], sep.inputs[0])
        return (v, "Distance"), (sep, 0)

    def stretched(sx, sy, sz):
        mp = t.n("ShaderNodeMapping")
        t.nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        mp.inputs["Scale"].default_value = (sx, sy, sz)
        return (mp, "Vector")

    M = {k: t.attr(k) for k in ("m_lips", "m_red", "m_brow", "m_undereye", "m_tzone", "m_forehead", "m_crow",
                                "m_beard", "m_scalp", "m_face", "m_ears", "m_socket", "m_margin")}
    base = lin(P["base"])
    # 1. low-frequency mottling (luminance + slight hue)
    mott = t.smooth(0.15, 0.85, noise(42, 6, 0.55))
    mott_c = t.mix((base[0] * 0.93, base[1] * 0.91, base[2] * 0.91), (base[0] * 1.05, base[1] * 1.05, base[2] * 1.03), mott)
    mott2 = t.smooth(0.1, 0.9, noise(150, 3, 0.5))
    mott_c = t.mix(mott_c, t.mix(mott_c, (0.94, 0.9, 0.9), 1.0, "MULTIPLY"), t.math("MULTIPLY", mott2, 0.5))
    # 2. haemoglobin redness: cheeks, nose, ears + blotches
    redn = t.math("MULTIPLY_ADD", M["m_red"], P["red"] * 0.75, t.math("MULTIPLY", t.math("SUBTRACT", noise(16, 3), 0.45), P["red"] * 0.5))
    redn = t.math("MULTIPLY", redn, 1.0, clamp=True)
    c = t.mix(mott_c, t.mix(mott_c, (1.0, 0.66, 0.62), 1.0, "MULTIPLY"), redn)
    # 3. under-eye shadow (slightly violet)
    c = t.mix(c, t.mix(c, (0.8, 0.72, 0.82), 1.0, "MULTIPLY"), t.math("MULTIPLY", M["m_undereye"], P["undereye"]))
    # 4. lips: vermilion colour with fine vertical texture
    lipn = noise(220, 3, 0.6, stretched(3.0, 3.0, 0.4))
    lip_c = t.mix(lin(P["lip"]), t.mix(lin(P["lip"]), (0.8, 0.7, 0.7), 1.0, "MULTIPLY"), t.smooth(0.3, 0.7, lipn))
    lip_c = t.mix(lip_c, t.mix(mott_c, (0.86, 0.66, 0.64), 1.0, "MULTIPLY"), 0.3)
    c = t.mix(c, lip_c, t.math("POWER", M["m_lips"], 1.25))
    # 5. inner socket / mouth lining: pinker and darker
    c = t.mix(c, t.mix(c, (0.75, 0.45, 0.45), 1.0, "MULTIPLY"), t.math("MULTIPLY", M["m_socket"], 0.85))
    c = t.mix(c, (0.62, 0.30, 0.30), t.math("MULTIPLY", M["m_margin"], 0.75))          # wet pink waterline
    # 6. freckles (face) and age spots
    fd, fr = voronoi(520)
    fspot = t.math("MULTIPLY", t.smooth(0.42, 0.12, fd), t.smooth(1 - 0.35 * P["freckle"], 1.0, fr))
    fspot = t.math("MULTIPLY", fspot, t.math("MULTIPLY", M["m_face"], P["freckle"] * 0.9))
    c = t.mix(c, t.mix(c, (0.72, 0.55, 0.42), 1.0, "MULTIPLY"), fspot)
    ad, ar = voronoi(95)
    aspot = t.math("MULTIPLY", t.smooth(0.5, 0.15, ad), t.smooth(0.82, 0.95, ar))
    aspot = t.math("MULTIPLY", aspot, P["age"] * 0.7)
    c = t.mix(c, t.mix(c, (0.74, 0.6, 0.48), 1.0, "MULTIPLY"), aspot)
    # 7. pores (~0.4 mm), stronger on nose/cheeks
    pd, _ = voronoi(2400)
    pore = t.smooth(0.32, 0.0, pd)
    pore_k = t.math("MULTIPLY", t.math("MULTIPLY_ADD", t.math("ADD", M["m_tzone"], M["m_red"]), 0.35, 0.65), P["pore"])
    c = t.mix(c, t.mix(c, (0.74, 0.66, 0.64), 1.0, "MULTIPLY"), t.math("MULTIPLY", pore, t.math("MULTIPLY", pore_k, 0.75)))
    # 8. stubble / buzz-cut: hair-follicle dots + soft shadow
    hair = lin(P["hair"])
    sd, sr = voronoi(1500)
    dots = t.math("MULTIPLY", t.smooth(0.30, 0.05, sd), t.smooth(0.15, 0.3, sr))
    beard = t.math("MULTIPLY", M["m_beard"], P["beard"])
    c = t.mix(c, t.mix(c, (0.78, 0.74, 0.74), 1.0, "MULTIPLY"), t.math("MULTIPLY", beard, 0.6))
    c = t.mix(c, hair, t.math("MULTIPLY", dots, beard))
    scalp = t.math("MULTIPLY", M["m_scalp"], P["scalp"])
    hd, hr = voronoi(2100)
    hdots = t.math("MULTIPLY", t.smooth(0.38, 0.08, hd), t.smooth(0.05, 0.2, hr))
    c = t.mix(c, t.mix(c, hair, 1.0, "MULTIPLY"), t.math("MULTIPLY", scalp, 0.55))
    c = t.mix(c, hair, t.math("MULTIPLY", hdots, scalp))
    # 9. eyebrows: an anatomical arched band (per texel, from 3D position) filled with fine strands
    sep = t.n("ShaderNodeSeparateXYZ")
    t.nt.links.new(tc.outputs["Object"], sep.inputs[0])
    ax = t.math("ABSOLUTE", (sep, 0))
    sb = t.math("DIVIDE", t.math("SUBTRACT", ax, 0.008), 0.052)
    sc_ = t.math("MINIMUM", t.math("MAXIMUM", sb, 0.0), 1.0)
    rise = t.math("SINE", t.math("MULTIPLY", t.math("MINIMUM", t.math("DIVIDE", sc_, 0.62), 1.0), 1.5708))
    fall = t.math("POWER", t.math("DIVIDE", t.math("MAXIMUM", t.math("SUBTRACT", sc_, 0.62), 0.0), 0.38), 1.3)
    yc = t.math("SUBTRACT", t.math("ADD", 0.3155, t.math("MULTIPLY", rise, 0.007)), t.math("MULTIPLY", fall, 0.010))
    hw = t.math("ADD", 0.0056, t.math("MULTIPLY", sc_, -0.0040))
    dist = t.math("ABSOLUTE", t.math("SUBTRACT", (sep, 2), yc))
    band = t.math("SUBTRACT", 1.0, t.math("DIVIDE", t.math("MAXIMUM", t.math("SUBTRACT", dist, t.math("MULTIPLY", hw, 0.45)), 0.0),
                                           t.math("MULTIPLY", hw, 0.55)), clamp=True)
    band = t.math("MULTIPLY", band, band)
    ends = t.math("MULTIPLY", t.smooth(-0.04, 0.07, sb), t.smooth(1.06, 0.9, sb))
    front = t.smooth(-0.07, -0.09, (sep, 1))
    brow_m = t.math("MULTIPLY", t.math("MULTIPLY", band, ends), front)
    strands = noise(1.0, 3, 0.55, stretched(22, 1900, 900))
    s_on = t.smooth(0.40, 0.55, strands)
    bden = t.math("MULTIPLY", brow_m, P["brow_k"])
    c = t.mix(c, t.mix(c, lin(P["brow"]), 0.5), t.math("MULTIPLY", bden, 0.3))   # skin tint under the brow hairs
    albedo = c
    # ---- roughness
    rough = t.math("ADD", 0.56, t.math("MULTIPLY", M["m_tzone"], -0.08))
    rough = t.math("ADD", rough, t.math("MULTIPLY", M["m_lips"], -0.2))
    rough = t.math("ADD", rough, t.math("MULTIPLY", M["m_margin"], -0.4))
    rough = t.math("ADD", rough, t.math("MULTIPLY", scalp, 0.1))
    rough = t.math("ADD", rough, t.math("MULTIPLY", t.math("SUBTRACT", noise(300, 3), 0.5), 0.12))
    rough = t.math("ADD", rough, t.math("MULTIPLY", pore, 0.08), clamp=True)
    # ---- height for the normal map: pores, fine crosshatch, forehead lines, crow's feet
    fine = t.math("MULTIPLY", t.math("SUBTRACT", noise(700, 4, 0.7), 0.5), 0.15)
    h = t.math("SUBTRACT", fine, t.math("MULTIPLY", pore, t.math("MULTIPLY", pore_k, 0.9)))
    wav = t.n("ShaderNodeTexWave", _wave_type="BANDS", _bands_direction="Z", Scale=38, Distortion=6, Detail=3)
    t.nt.links.new(tc.outputs["Object"], wav.inputs["Vector"])
    lines = t.math("POWER", t.math("SUBTRACT", 1.0, (wav, "Fac")), 6)
    h = t.math("SUBTRACT", h, t.math("MULTIPLY", lines, t.math("MULTIPLY", M["m_forehead"], 1.4 * P["age"])))
    crow = t.smooth(0.55, 0.75, noise(1.0, 2, 0.5, stretched(500, 60, 120)))
    h = t.math("SUBTRACT", h, t.math("MULTIPLY", crow, t.math("MULTIPLY", M["m_crow"], 1.1 * P["age"])))
    lipl = t.smooth(0.45, 0.65, noise(1.0, 2, 0.5, stretched(900, 900, 60)))
    h = t.math("SUBTRACT", h, t.math("MULTIPLY", lipl, t.math("MULTIPLY", M["m_lips"], 0.22)))
    return t, albedo, rough, h


def add_mask_attributes(ob, masks):
    for k, w in masks.items():
        if k in ob.data.attributes:
            ob.data.attributes.remove(ob.data.attributes[k])
        a = ob.data.attributes.new(k, "FLOAT", "POINT")
        a.data.foreach_set("value", w.astype(np.float32))


def bake_object(masks):
    src = bpy.data.objects["GNM_Head"]
    me = src.data.copy()
    ob = bpy.data.objects.new("BAKE_Skin", me)
    bpy.context.scene.collection.objects.link(ob)
    if me.shape_keys:
        ob.shape_key_clear()
    add_mask_attributes(ob, masks)          # before removing faces: bmesh keeps the attributes
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index != 0], context="FACES")
    bm.to_mesh(me)
    bm.free()
    me.materials.clear()
    for o in bpy.context.scene.objects:
        o.hide_render = o is not ob
    return ob


def new_image(name, size, non_color):
    img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def bake(ob, mat, t, out_socket, kind, img, margin=24):
    nodes = t.nt.nodes
    for nd in [n for n in nodes if n.bl_idname in ("ShaderNodeOutputMaterial", "ShaderNodeEmission",
                                                    "ShaderNodeBsdfPrincipled", "ShaderNodeBump", "ShaderNodeTexImage")]:
        nodes.remove(nd)
    out = nodes.new("ShaderNodeOutputMaterial")
    if kind == "NORMAL":
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = 0.00025
        t._in(bump, "Height", out_socket) if False else t.nt.links.new(
            out_socket[0].outputs[out_socket[1]] if isinstance(out_socket, tuple) else out_socket.outputs[0], bump.inputs["Height"])
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        t.nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        t.nt.links.new(bsdf.outputs[0], out.inputs[0])
    else:
        em = nodes.new("ShaderNodeEmission")
        src = out_socket[0].outputs[out_socket[1]] if isinstance(out_socket, tuple) else out_socket.outputs[0]
        t.nt.links.new(src, em.inputs["Color"])
        t.nt.links.new(em.outputs[0], out.inputs[0])
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    nodes.active = tex
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    sc = bpy.context.scene
    sc.render.bake.margin = margin
    sc.render.bake.margin_type = "EXTEND"
    if kind == "NORMAL":
        sc.render.bake.normal_space = "TANGENT"
        bpy.ops.object.bake(type="NORMAL")
    else:
        bpy.ops.object.bake(type="EMIT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(p["id"] for p in PRESETS))
    ap.add_argument("--size", type=int, default=4096)
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 4
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev in os.environ.get("XR_CYCLES_DEVICE", "OPTIX,CUDA").split(","):
        try:
            prefs.compute_device_type = dev
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type == dev
            sc.cycles.device = "GPU"
            break
        except TypeError:
            continue
    m = G.GNM()
    masks = SM.build(m)
    ob = bake_object(masks)
    for P in PRESETS:
        if P["id"] not in a.ids.split(","):
            continue
        d = os.path.join(OUT, f"{P['id']}_{P['name']}")
        os.makedirs(d, exist_ok=True)
        mat = bpy.data.materials.new(f"AUTHOR_{P['id']}")
        ob.data.materials.clear()
        ob.data.materials.append(mat)
        for kind, sock_i, non_color in (("albedo", 1, False), ("roughness", 2, True), ("normal", 3, True)):
            t, alb, rough, h = build_skin_material(mat, P)
            sock = {"albedo": alb, "roughness": rough, "normal": h}[kind]
            img = new_image(f"{P['id']}_{kind}", a.size, non_color)
            bake(ob, mat, t, sock, "NORMAL" if kind == "normal" else "EMIT", img)
            img.filepath_raw = os.path.join(d, f"{P['id']}_{kind}.1001.png")
            img.file_format = "PNG"
            img.save()
            print(P["id"], kind, "baked", flush=True)
        print(P["id"], "done ->", d, flush=True)


if __name__ == "__main__":
    main()
