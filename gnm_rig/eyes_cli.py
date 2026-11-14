"""Eye-motion renderer for the GNM head (rubric: one script, three inputs, frames + video).

    python gnm_rig/eyes_cli.py --input left       # both eyes move to the left, naturally (saccades, blinks, head follow)
    python gnm_rig/eyes_cli.py --input right      # both eyes move to the right
    python gnm_rig/eyes_cli.py --input motion     # a natural sequence: look around, eye-roll upwards, reading glances,
                                                  # blinks and a double blink (12 s)
Options: --tex T01..T08 (texture set), --head PATH, --res 1920x1080, --samples 48, --fps 30, --out DIR,
         --blender PATH (or env XR_BLENDER); env XR_CYCLES_DEVICE=CUDA if OptiX fails on your driver

Renders PNG frames with Blender (Cycles, GPU) and encodes an H.264 MP4 (each video is <= 15 s).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HEAD = os.path.join(HERE, "out", "gnm_head.blend")
MAX_SECONDS = 15.0


def find_blender(explicit=None):
    if explicit or os.environ.get("XR_BLENDER"):
        return explicit or os.environ["XR_BLENDER"]
    cands = sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe"), reverse=True)
    for c in [shutil.which("blender")] + cands:
        if c and os.path.exists(c):
            return c
    sys.exit("Blender not found: pass --blender PATH")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, choices=["left", "right", "motion"])
    ap.add_argument("--tex", default="T02")
    ap.add_argument("--head", default=None, help="head .blend (default: out/heads/<tex>.blend if built, else out/gnm_head.blend)")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(ROOT, "renders_eyes_cli"))
    ap.add_argument("--blender", default=None)
    a = ap.parse_args()
    blender = find_blender(a.blender)
    head = a.head or os.path.join(HERE, "out", "heads", f"{a.tex}.blend")
    if not os.path.exists(head):
        head = HEAD
    if not os.path.exists(head):
        subprocess.run([blender, "-b", "--factory-startup", "-P", os.path.join(HERE, "build_head.py")], check=True)
    a.out = os.path.abspath(a.out)
    frames = os.path.join(a.out, a.input, "frames")
    shutil.rmtree(frames, ignore_errors=True)
    os.makedirs(frames, exist_ok=True)
    cmd = [blender, "-b", "--factory-startup", head, "-P", os.path.join(HERE, "animate.py"), "--",
           "--input", a.input, "--tex", a.tex, "--res", a.res, "--samples", str(a.samples), "--fps", str(a.fps),
           "--frames-dir", frames]
    print("rendering:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    n = len(glob.glob(os.path.join(frames, "*.png")))
    mp4 = os.path.join(a.out, f"eyes_{a.input}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(frames, "%04d.png"),
                    "-t", str(MAX_SECONDS), "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", mp4], check=True)
    print(f"{n} frames -> {mp4} ({n / a.fps:.1f} s)")


if __name__ == "__main__":
    main()
