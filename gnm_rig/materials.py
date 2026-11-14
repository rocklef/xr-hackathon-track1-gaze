"""Apply a baked GNM texture set (UDIM: <ID>_{albedo,normal,roughness}.<UDIM>.png) to the GNM_Head render materials.
One tiled image per channel feeds every region (skin 1001, eyes 1002/1004, teeth 1006, tongue 1007)."""
import glob
import os

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(ROOT, "textures")


def set_dir(tex_id):
    d = glob.glob(os.path.join(TEX, f"{tex_id}_*"))
    if not d:
        raise FileNotFoundError(f"no texture set {tex_id} in {TEX}")
    return d[0]


def tiled(path_1001, non_color):
    p = path_1001.replace(".1001.", ".<UDIM>.")
    name = os.path.basename(p)
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.load(p, check_existing=True)
        img.source = "TILED"
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def _reset(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return nt


def _tex(nt, img, uv):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = img
    t.interpolation = "Cubic"
    nt.links.new(uv.outputs["UV"], t.inputs["Vector"])
    return t


def apply(tex_id, head_name="GNM_Head", sss=1.0):
    d = set_dir(tex_id)
    base = os.path.join(d, f"{tex_id}_")
    alb = tiled(base + "albedo.1001.png", False)
    nrm = tiled(base + "normal.1001.png", True)
    rgh = tiled(base + "roughness.1001.png", True)
    head = bpy.data.objects[head_name]
    mats = {m.name: m for m in head.data.materials}

    # ---- skin
    nt = _reset(mats["MAT_skin"])
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    a, n, r = _tex(nt, alb, uv), _tex(nt, nrm, uv), _tex(nt, rgh, uv)
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = 0.8
    nt.links.new(n.outputs["Color"], nm.inputs["Color"])
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(r.outputs["Color"], b.inputs["Roughness"])
    nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    b.subsurface_method = "RANDOM_WALK_SKIN"
    b.inputs["Subsurface Weight"].default_value = 0.5 * sss
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.36, 0.18)
    b.inputs["Subsurface Scale"].default_value = 0.0035
    b.inputs["Subsurface IOR"].default_value = 1.4
    b.inputs["Specular IOR Level"].default_value = 0.5
    b.inputs["Sheen Weight"].default_value = 0.12
    b.inputs["Sheen Roughness"].default_value = 0.35
    b.inputs["Coat Weight"].default_value = 0.04
    b.inputs["Coat Roughness"].default_value = 0.25
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(b.outputs[0], out.inputs[0])

    # ---- eye interiors (sclera + iris + pupil), one per eye, same tiled image (tiles 1002 / 1004)
    for s in ("L", "R"):
        nt = _reset(mats[f"MAT_eye_interior_{s}"])
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = "UVMap"
        a = _tex(nt, alb, uv)
        en = _tex(nt, nrm, uv)
        enm = nt.nodes.new("ShaderNodeNormalMap")
        enm.inputs["Strength"].default_value = 1.0
        nt.links.new(en.outputs["Color"], enm.inputs["Color"])
        # lids softly shadow the eyeball: darken by ambient occlusion within ~3 mm (the "eye occlusion" look)
        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = 8
        ao.inputs["Distance"].default_value = 0.0018
        aom = nt.nodes.new("ShaderNodeMapRange")
        aom.inputs[3].default_value, aom.inputs[4].default_value = 0.72, 1.0
        nt.links.new(ao.outputs["AO"], aom.inputs[0])
        dark = nt.nodes.new("ShaderNodeMix")
        dark.data_type, dark.blend_type = "RGBA", "MULTIPLY"
        dark.inputs[0].default_value = 1.0
        nt.links.new(a.outputs["Color"], dark.inputs[6])
        nt.links.new(aom.outputs[0], dark.inputs[7])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(dark.outputs[2], b.inputs["Base Color"])
        nt.links.new(enm.outputs["Normal"], b.inputs["Normal"])
        b.inputs["Roughness"].default_value = 0.45
        b.inputs["Subsurface Weight"].default_value = 0.3        # sclera/iris stroma translucency
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.25)
        b.inputs["Subsurface Scale"].default_value = 0.0012
        b.inputs["Specular IOR Level"].default_value = 0.25
        # pupil = light trap: UV radius < 0.0495 from the tile centre (pupil vertices keep these UVs when dilating)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(uv.outputs["UV"], sep.inputs[0])
        fu = nt.nodes.new("ShaderNodeMath"); fu.operation = "FRACT"
        nt.links.new(sep.outputs[0], fu.inputs[0])
        du = nt.nodes.new("ShaderNodeMath"); du.operation = "SUBTRACT"; du.inputs[1].default_value = 0.5
        nt.links.new(fu.outputs[0], du.inputs[0])
        dv = nt.nodes.new("ShaderNodeMath"); dv.operation = "SUBTRACT"; dv.inputs[1].default_value = 0.5
        nt.links.new(sep.outputs[1], dv.inputs[0])
        cmb = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(du.outputs[0], cmb.inputs[0]); nt.links.new(dv.outputs[0], cmb.inputs[1])
        ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
        nt.links.new(cmb.outputs[0], ln.inputs[0])
        msk = nt.nodes.new("ShaderNodeMapRange")
        msk.inputs[1].default_value, msk.inputs[2].default_value = 0.0485, 0.0465
        nt.links.new(ln.outputs["Value"], msk.inputs[0])
        # subsurface only on the sclera (iris detail stays crisp): UV radius 0.18 -> 0.20
        sssm = nt.nodes.new("ShaderNodeMapRange")
        sssm.inputs[1].default_value, sssm.inputs[2].default_value = 0.18, 0.20
        sssm.inputs[3].default_value, sssm.inputs[4].default_value = 0.03, 0.35
        nt.links.new(ln.outputs["Value"], sssm.inputs[0])
        nt.links.new(sssm.outputs[0], b.inputs["Subsurface Weight"])
        black = nt.nodes.new("ShaderNodeBsdfDiffuse")
        black.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1)
        mixp = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(msk.outputs[0], mixp.inputs[0])
        nt.links.new(b.outputs[0], mixp.inputs[1])
        nt.links.new(black.outputs[0], mixp.inputs[2])
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(mixp.outputs[0], out.inputs[0])

        # ---- cornea: clear wet glass; shadow rays pass through so the iris is lit (no caustic blackout)
        nt = _reset(mats[f"MAT_cornea_{s}"])
        g = nt.nodes.new("ShaderNodeBsdfPrincipled")
        g.inputs["Base Color"].default_value = (1, 1, 1, 1)
        g.inputs["Roughness"].default_value = 0.0
        g.inputs["IOR"].default_value = 1.376
        g.inputs["Transmission Weight"].default_value = 1.0
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        lp = nt.nodes.new("ShaderNodeLightPath")
        mx = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Shadow Ray"], mx.inputs[0])
        nt.links.new(g.outputs[0], mx.inputs[1])
        nt.links.new(tr.outputs[0], mx.inputs[2])
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.target = "CYCLES"
        nt.links.new(mx.outputs[0], out.inputs[0])
        # viewport / EEVEE: transparent cornea with a wet specular highlight (glass would render opaque there)
        fr = nt.nodes.new("ShaderNodeFresnel")
        fr.inputs["IOR"].default_value = 1.376
        gl = nt.nodes.new("ShaderNodeBsdfGlossy")
        gl.inputs["Roughness"].default_value = 0.02
        tr2 = nt.nodes.new("ShaderNodeBsdfTransparent")
        mx2 = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(fr.outputs[0], mx2.inputs[0])
        nt.links.new(tr2.outputs[0], mx2.inputs[1])
        nt.links.new(gl.outputs[0], mx2.inputs[2])
        out2 = nt.nodes.new("ShaderNodeOutputMaterial")
        out2.target = "EEVEE"
        nt.links.new(mx2.outputs[0], out2.inputs[0])
        mats[f"MAT_cornea_{s}"].surface_render_method = "BLENDED"

    # ---- tear meniscus: clear wet film (Cycles: refractive; shadows pass through)
    tl = bpy.data.materials.get("MAT_tearline")
    if tl:
        nt = _reset(tl)
        g = nt.nodes.new("ShaderNodeBsdfPrincipled")
        g.inputs["Base Color"].default_value = (1.0, 0.97, 0.96, 1)
        g.inputs["Roughness"].default_value = 0.02
        g.inputs["IOR"].default_value = 1.336
        g.inputs["Transmission Weight"].default_value = 1.0
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        lp = nt.nodes.new("ShaderNodeLightPath")
        mx = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Shadow Ray"], mx.inputs[0])
        nt.links.new(g.outputs[0], mx.inputs[1])
        nt.links.new(tr.outputs[0], mx.inputs[2])
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(mx.outputs[0], out.inputs[0])

    for nm_, rough in (("MAT_teeth", 0.25), ("MAT_tongue", 0.4)):
        nt = _reset(mats[nm_])
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = "UVMap"
        a = _tex(nt, alb, uv)
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = rough
        b.inputs["Subsurface Weight"].default_value = 0.3
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(b.outputs[0], out.inputs[0])
    return d
