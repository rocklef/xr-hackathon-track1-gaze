"""Rubric item 1 (Texture photorealism): render the GNM head with the generated UV texture sets.

    blender -b --factory-startup gnm_rig/out/gnm_head.blend -P render_textures.py -- --tex all
    blender -b --factory-startup gnm_rig/out/heads/T03.blend -P render_textures.py -- --tex T03 --views portrait,eyes,macro

Options (after --): --tex T01,T02|all  --views portrait,front,eyes,macro  --samples 128  --res 1920x1080  --out DIR
Implementation: gnm_rig/render_textures.py (materials: gnm_rig/materials.py).
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnm_rig", "render_textures.py"), run_name="__main__")
