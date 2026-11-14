"""Rubric item 2 (Kinematic accuracy): one script, three inputs -> rendered frames + video (each <= 15 s).

    python eyes_cli.py --input left       # both eyes move to the left, naturally
    python eyes_cli.py --input right      # both eyes move to the right
    python eyes_cli.py --input motion     # a natural sequence (glances, upward eye-roll, reading glances, blinks)

Options: --tex T01..T08, --res 1920x1080, --samples 48, --fps 30, --out DIR, --blender PATH (or env XR_BLENDER).
Implementation: gnm_rig/eyes_cli.py (Blender side: gnm_rig/animate.py, motion model: gnm_rig/motion.py).
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnm_rig", "eyes_cli.py"), run_name="__main__")
