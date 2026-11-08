"""Make the submission zip in the exact required layout.

    python tools/package.py CollegeName_TeamName      -> dist/CollegeName_TeamName.zip
"""
import os
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
name = sys.argv[1] if len(sys.argv) > 1 else "CollegeName_TeamName"
FILES = ["scene.blend", "wrapper/eyes.py", "metadata.json", "README.md", "contact_sheet.png"]
FILES += [f"renders/frame_{k:02d}.png" for k in range(10)]
FILES += [f"checks/{n}.png" for n in ("center", "left", "right", "up", "down", "reset")]
FILES += ["checks/directions.json", "checks/report.json"]
FILES += [f"tools/{f}" for f in ("build_scene.py", "calibrate_pupil.py", "pupil_detect.py",
                                  "verify.py", "contact_sheet.py", "pipeline.py", "package.py")]

missing = [f for f in FILES if not os.path.isfile(os.path.join(ROOT, f))]
if missing:
    sys.exit(f"missing: {missing}")
os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
out = os.path.join(ROOT, "dist", f"{name}.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(os.path.join(ROOT, f), f"{name}/{f}")
print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB, {len(FILES)} files)")
