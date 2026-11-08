"""One-command pipeline (plain Python 3.9+, run from anywhere).

    python tools/pipeline.py                 # render + verify + contact sheet (uses shipped scene.blend)
    python tools/pipeline.py --all           # rebuild scene from MPFB, calibrate, render, verify, sheet
    python tools/pipeline.py --steps build calibrate

Blender is found via $BLENDER, then PATH, then the default install folders.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STEPS = ["build", "lookdev", "calibrate", "render", "verify", "sheet"]


def find_blender():
    cand = [os.environ.get("BLENDER"), shutil.which("blender")]
    cand += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe"), reverse=True)
    cand += ["/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"]
    for c in cand:
        if c and os.path.isfile(c):
            return c
    sys.exit("Blender not found: set BLENDER=/path/to/blender")


def run(cmd):
    print("\n>>", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        sys.exit(f"step failed ({r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", nargs="+", choices=STEPS)
    a = ap.parse_args()
    steps = STEPS if a.all else (a.steps or ["render", "verify", "sheet"])
    blender = None
    if {"build", "calibrate", "render"} & set(steps):
        blender = find_blender()
        print("Blender:", blender)
    blend = os.path.join(ROOT, "scene.blend")
    for s in STEPS:
        if s not in steps:
            continue
        if s == "build":   # needs the MPFB extension + MakeHuman CC0 system assets
            run([blender, "-b", "--addons", "bl_ext.user_default.mpfb", "-P", "tools/build_scene.py"])
        elif s == "lookdev":
            run([blender, "-b", "--factory-startup", blend, "-P", "tools/lookdev.py"])
        elif s == "calibrate":
            run([blender, "-b", "--factory-startup", blend, "-P", "tools/calibrate_pupil.py"])
        elif s == "render":
            run([blender, "-b", "--factory-startup", blend, "-P", "wrapper/eyes.py", "--", "--sequence", "--directions"])
        elif s == "verify":
            run([sys.executable, "tools/verify.py"])
        elif s == "sheet":
            run([sys.executable, "tools/contact_sheet.py"])


if __name__ == "__main__":
    main()
