"""Independent, image-based QA for Track 1 outputs (plain Python: numpy + Pillow).

    python tools/verify.py            # checks/ (5-direction) + renders/ + metadata.json

It does NOT trust metadata for the direction test: pupils are detected from
the PNG pixels (darkest round blob inside each eye), then compared with the
centre render. The metadata pupil centres are then checked against the
detected ones (reprojection error).
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pupil_detect import detect  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIN_SHIFT_PX = 5.0      # a direction must move the pupil at least this much
RESET_TOL_PX = 1.5
ANGLE_TOL_DEG = 15.0
REQUIRED = ["frame", "theta", "input", "eye_rotation_L", "eye_rotation_R",
            "pupil_center_2D_L", "pupil_center_2D_R", "camera_matrix"]


def detect_pupil(img, approx):
    """Pupil centre from pixels only (see tools/pupil_detect.py)."""
    return detect(np.asarray(img.convert("RGB"), dtype=np.float32), approx)


def check_directions(d):
    ref = json.load(open(os.path.join(d, "directions.json")))
    c_img = Image.open(os.path.join(d, "center.png"))
    centre = {s: detect_pupil(c_img, ref["center"][f"pupil_center_2D_{s}"]) for s in "LR"}
    expect = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1), "reset": (0, 0)}
    rows, all_ok = [], True
    for name, (ex, ey) in expect.items():
        img = Image.open(os.path.join(d, f"{name}.png"))
        ok_eyes = []
        for s in "LR":
            p = detect_pupil(img, centre[s])
            dx, dy = p[0] - centre[s][0], p[1] - centre[s][1]
            if name == "reset":
                ok = math.hypot(dx, dy) <= RESET_TOL_PX
            else:
                along = dx * ex + dy * ey
                across = abs(dx * ey) + abs(dy * ex)
                ok = along >= MIN_SHIFT_PX and across < along / 2
            meta = ref[name][f"pupil_center_2D_{s}"]
            err = math.hypot(p[0] - meta[0], p[1] - meta[1])
            ok_eyes.append(dict(eye=s, dx=round(dx, 2), dy=round(dy, 2), pass_=bool(ok), meta_err_px=round(err, 2)))
        passed = all(e["pass_"] for e in ok_eyes)
        all_ok &= passed
        rows.append(dict(direction=name.upper(), passed=passed, eyes=ok_eyes))
    return all_ok, rows


def check_sequence(root):
    meta = json.load(open(os.path.join(root, "metadata.json")))
    problems = []
    if len(meta) != 10:
        problems.append(f"expected 10 records, got {len(meta)}")
    centre_ref = json.load(open(os.path.join(root, "checks", "directions.json")))["center"]
    c_img = Image.open(os.path.join(root, "checks", "center.png"))
    centre = {s: detect_pupil(c_img, centre_ref[f"pupil_center_2D_{s}"]) for s in "LR"}
    rows = []
    reproj_max = [0.0]
    for k, r in enumerate(meta):
        miss = [key for key in REQUIRED if key not in r]
        if miss:
            problems.append(f"frame {k}: missing {miss}")
        theta = 2 * math.pi * k / 10
        if r["frame"] != k or abs(r["theta"] - theta) > 1e-3:
            problems.append(f"frame {k}: frame/theta mismatch")
        if abs(r["input"]["h"] - math.sin(theta)) > 1e-3 or abs(r["input"]["v"] - math.cos(theta)) > 1e-3:
            problems.append(f"frame {k}: h,v != sin,cos")
        # camera model consistency: K [R|t] must reproject the 3D pupils onto the 2D ones
        if "camera_projection_P" in r:
            P = np.array(r["camera_projection_P"])
            Kc = np.array(r["camera_matrix"]); RT = np.array(r["camera_extrinsics_RT"])
            if np.abs(Kc @ RT - P).max() > 1e-2 * max(1.0, np.abs(P).max()):
                problems.append(f"frame {k}: P != K[R|t]")
            for s_ in "LR":
                X = np.r_[r[f"pupil_center_3D_{s_}"], 1.0]
                u = P @ X
                e = np.hypot(u[0] / u[2] - r[f"pupil_center_2D_{s_}"][0], u[1] / u[2] - r[f"pupil_center_2D_{s_}"][1])
                if e > 0.5:
                    problems.append(f"frame {k} eye {s_}: P reprojection error {e:.2f}px")
                reproj_max[0] = max(reproj_max[0], e)
        img = Image.open(os.path.join(root, "renders", f"frame_{k:02d}.png"))
        if img.size != (1024, 1024):
            problems.append(f"frame {k}: size {img.size}")
        eyes = []
        for s in "LR":
            p = detect_pupil(img, r[f"pupil_center_2D_{s}"])
            dx, dy = p[0] - centre[s][0], p[1] - centre[s][1]
            ang = math.degrees(math.atan2(dx, -dy)) % 360       # clockwise from top
            dang = (ang - math.degrees(theta) + 180) % 360 - 180
            err = math.hypot(p[0] - r[f"pupil_center_2D_{s}"][0], p[1] - r[f"pupil_center_2D_{s}"][1])
            if abs(dang) > ANGLE_TOL_DEG:
                problems.append(f"frame {k} eye {s}: image angle {ang:.1f} vs theta {math.degrees(theta):.1f}")
            eyes.append(dict(eye=s, angle_deg=round(ang, 1), angle_err_deg=round(dang, 1),
                             shift_px=round(math.hypot(dx, dy), 2), meta_err_px=round(err, 2)))
        rows.append(dict(frame=k, theta_deg=round(math.degrees(theta), 1), eyes=eyes))
    print(f"  camera model: max |K[R|t]*pupil3D - pupil2D| = {reproj_max[0]:.4f} px")
    return not problems, problems, rows


def main():
    report = {}
    ok_d, rows_d = check_directions(os.path.join(ROOT, "checks"))
    report["directions"] = dict(passed=ok_d, rows=rows_d)
    print("== 5-direction check (pixels vs centre render) ==")
    for r in rows_d:
        e = "  ".join(f"{x['eye']}: dx={x['dx']:+6.2f} dy={x['dy']:+6.2f} metaErr={x['meta_err_px']:.2f}px" for x in r["eyes"])
        print(f"  {r['direction']:<6} {'PASS' if r['passed'] else 'FAIL'}   {e}")
    print(f"  => {sum(r['passed'] for r in rows_d)}/5 directions + reset pass" if ok_d else "  => FAIL")

    if os.path.exists(os.path.join(ROOT, "metadata.json")):
        ok_s, probs, rows_s = check_sequence(ROOT)
        report["sequence"] = dict(passed=ok_s, problems=probs, rows=rows_s)
        print("== clockwise sequence (image angle vs theta_k) ==")
        for r in rows_s:
            e = "  ".join(f"{x['eye']}: ang={x['angle_deg']:6.1f} err={x['angle_err_deg']:+5.1f} shift={x['shift_px']:5.2f}px metaErr={x['meta_err_px']:.2f}" for x in r["eyes"])
            print(f"  k={r['frame']} theta={r['theta_deg']:5.1f}  {e}")
        print("  => PASS" if ok_s else "  => FAIL: " + "; ".join(probs))
    else:
        ok_s = False
    with open(os.path.join(ROOT, "checks", "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    sys.exit(0 if ok_d and ok_s else 1)


if __name__ == "__main__":
    main()
