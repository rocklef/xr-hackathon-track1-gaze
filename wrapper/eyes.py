"""eyes.py - deterministic gaze control + rendering for scene.blend.

Inside Blender (scene.blend open):
    import eyes
    eyes.move(h, v)      # h, v in [-1, 1]; +h = image right, +v = up
    eyes.reset()         # back to centre fixation
    rec = eyes.render("renders/frame_00.png")   # PNG + metadata record

Headless (what the evaluator runs):
    blender -b scene.blend -P wrapper/eyes.py -- --sequence
    blender -b scene.blend -P wrapper/eyes.py -- --move 1 0 --out left.png
    blender -b scene.blend -P wrapper/eyes.py -- --directions

Geometry (all from scene["gaze_cfg"], measured at build time):
  * M = midpoint of the two eyeball centres.
  * Fixation target T = M + 0.35 m forward + (h*0.05 m) right + (v*0.05 m) up.
  * Each eye bone's pivot is its eyeball's fitted sphere centre; a Damped Track
    constraint turns it to look at T, so both eyes converge on T.
  * Camera locked at M + 0.40 m forward, facing the face; 1024x1024 PNG.
"""
import argparse
import json
import math
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

RIG = "ARM_Human"
EYES = "SM_Eyes"
TARGET = "TGT_Fixation"
CAMERA = "CAM_Gaze"
BONES = {"L": "eye.L", "R": "eye.R"}

_state = {"h": 0.0, "v": 0.0}


# --------------------------------------------------------------------------
def _cfg():
    return bpy.context.scene["gaze_cfg"]


def _clamp(x):
    return max(-1.0, min(1.0, float(x)))


def move(h, v):
    """Move both pupils synchronously. h,v are normalised in [-1, 1]."""
    h, v = _clamp(h), _clamp(v)
    bound = float(_cfg()["motion_bound_m"])
    tgt = bpy.data.objects[TARGET]
    tgt.location = (h * bound, 0.0, v * bound)  # local to the rest anchor
    _state["h"], _state["v"] = h, v
    bpy.context.view_layer.update()
    return h, v


def reset():
    """Centre fixation (straight ahead, 35 cm)."""
    return move(0.0, 0.0)


# --------------------------------------------------------------------------
def _pixel(cam, scene, p):
    co = world_to_camera_view(scene, cam, p)
    r = scene.render
    w = r.resolution_x * r.resolution_percentage / 100
    hgt = r.resolution_y * r.resolution_percentage / 100
    # image coordinates: origin top-left, x right, y down (OpenCV style)
    return [round(co.x * w, 3), round((1.0 - co.y) * hgt, 3)]


def camera_matrices(scene=None, cam=None):
    """Return (K 3x3, RT 3x4 world->camera, OpenCV convention)."""
    scene = scene or bpy.context.scene
    cam = cam or scene.camera
    r = scene.render
    w = r.resolution_x * r.resolution_percentage / 100
    hgt = r.resolution_y * r.resolution_percentage / 100
    cd = cam.data
    sensor = cd.sensor_width if cd.sensor_fit != "VERTICAL" else cd.sensor_height
    f_px = cd.lens / sensor * (w if cd.sensor_fit != "VERTICAL" else hgt)
    K = [[f_px, 0.0, w / 2 - cd.shift_x * w], [0.0, f_px, hgt / 2 + cd.shift_y * w], [0.0, 0.0, 1.0]]
    # Blender camera: -Z forward, +Y up. OpenCV: +Z forward, +Y down.
    flip = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_wc = cam.matrix_world.to_3x3().normalized().transposed()
    t_wc = -(R_wc @ cam.matrix_world.translation)
    R = flip @ R_wc
    t = flip @ t_wc
    RT = [list(R[i]) + [t[i]] for i in range(3)]
    return K, RT


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _eye_state(side):
    """World rotation of the eye relative to rest, gaze vector, pupil point."""
    scene = bpy.context.scene
    rig = bpy.data.objects[RIG]
    pb = rig.pose.bones[BONES[side]]
    rest = (rig.matrix_world @ pb.bone.matrix_local).to_3x3().normalized()
    pose = (rig.matrix_world @ pb.matrix).to_3x3().normalized()
    delta = pose @ rest.inverted()  # world-frame rotation applied to the eye
    gaze = (pose @ Vector((0, 1, 0))).normalized()  # bone +Y = line of sight

    # pupil = calibrated entrance-pupil point, rigidly attached to the eye:
    #   eyeball centre (fixed pivot) + R_eye @ offset   (see tools/calibrate_pupil.py)
    # falls back to the front-surface point on the optical axis if uncalibrated.
    centre = rig.matrix_world @ pb.head
    cfg = _cfg()
    key = f"pupil_offset_local_{side}"
    if key in cfg:
        R_eye = (rig.matrix_world.to_3x3() @ pb.matrix.to_3x3()).normalized()
        p = centre + R_eye @ Vector(tuple(cfg[key]))
    else:
        p = centre + gaze * float(cfg[f"pupil_depth_{side}"])
    e = delta.to_euler("XYZ")
    return {
        "rotation_deg": [round(math.degrees(a), 4) for a in e],
        "gaze": [round(x, 6) for x in gaze],
        "pupil_3D": [round(x, 6) for x in p],
        "pupil_2D": _pixel(scene.camera, scene, p),
    }


def metadata(frame=None, theta=None):
    scene = bpy.context.scene
    L, R = _eye_state("L"), _eye_state("R")
    K, RT = camera_matrices(scene)
    tgt = bpy.data.objects[TARGET].matrix_world.translation
    vergence = math.degrees(Vector(L["gaze"]).angle(Vector(R["gaze"])))
    rec = {
        "frame": frame,
        "theta": None if theta is None else round(theta, 4),
        "input": {"h": round(_state["h"], 4), "v": round(_state["v"], 4)},
        "eye_rotation_L": L["rotation_deg"],
        "eye_rotation_R": R["rotation_deg"],
        "pupil_center_2D_L": L["pupil_2D"],
        "pupil_center_2D_R": R["pupil_2D"],
        "camera_matrix": [[round(x, 4) for x in row] for row in K],   # intrinsics K (OpenCV)
        "camera_extrinsics_RT": [[round(x, 6) for x in row] for row in RT],  # world -> camera
        "camera_projection_P": [[round(x, 4) for x in row] for row in _matmul(K, RT)],  # K [R|t]
        # extras (useful for gaze-estimator training)
        "gaze_vector_L": L["gaze"],
        "gaze_vector_R": R["gaze"],
        "pupil_center_3D_L": L["pupil_3D"],
        "pupil_center_3D_R": R["pupil_3D"],
        "fixation_target_3D": [round(x, 6) for x in tgt],
        "vergence_deg": round(vergence, 4),
        "units": {"rotation": "deg, XYZ euler, world frame, relative to rest (straight-ahead) pose",
                  "pupil_2D": "px, origin top-left, x right, y down",
                  "3D": "m, Blender world (Z up, face looks along -Y)",
                  "camera": "OpenCV: x_px ~ K [R|t] X_world"},
    }
    return rec


def setup_device():
    """Prefer GPU Cycles (OptiX/CUDA); fall back to CPU. Same seed/samples."""
    scene = bpy.context.scene
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type == kind]
                if devs:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    scene.cycles.device = "GPU"
                    return kind
            except TypeError:
                continue
    except Exception:
        pass
    scene.cycles.device = "CPU"
    return "CPU"


def render(path=None, frame=None, theta=None):
    """Render the locked camera view to PNG and return its metadata record."""
    scene = bpy.context.scene
    rec = metadata(frame, theta)
    if path:
        path = os.path.abspath(bpy.path.abspath(path))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        rec["image"] = os.path.basename(path)
    return rec


# --------------------------------------------------------------------------
def run_sequence(out_root, n=10):
    """10 clockwise renders, theta_k = 2*pi*k/n, h = sin, v = cos (k=0 is top)."""
    rdir = os.path.join(out_root, "renders")
    records = []
    for k in range(n):
        theta = 2 * math.pi * k / n
        move(math.sin(theta), math.cos(theta))
        rec = render(os.path.join(rdir, f"frame_{k:02d}.png"), frame=k, theta=theta)
        records.append(rec)
        print(f"[eyes] frame {k} theta={theta:.4f} h={rec['input']['h']} v={rec['input']['v']} "
              f"pL={rec['pupil_center_2D_L']} pR={rec['pupil_center_2D_R']}", flush=True)
    reset()
    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(records, f, indent=2)
    return records


def run_directions(out_root):
    """Centre/Left/Right/Up/Down renders for the 5-direction pass/fail check."""
    ddir = os.path.join(out_root, "checks")
    cases = [("center", 0, 0), ("left", -1, 0), ("right", 1, 0), ("up", 0, 1), ("down", 0, -1)]
    recs = {}
    for name, h, v in cases:
        move(h, v)
        recs[name] = render(os.path.join(ddir, f"{name}.png"))
    reset()
    recs["reset"] = render(os.path.join(ddir, "reset.png"))
    with open(os.path.join(ddir, "directions.json"), "w") as f:
        json.dump(recs, f, indent=2)
    return recs


def _main(argv):
    ap = argparse.ArgumentParser(prog="eyes.py")
    ap.add_argument("--sequence", action="store_true", help="10 clockwise renders + metadata.json")
    ap.add_argument("--directions", action="store_true", help="centre/L/R/U/D/reset check renders")
    ap.add_argument("--move", nargs=2, type=float, metavar=("H", "V"))
    ap.add_argument("--out", default=None, help="PNG path for a single --move render")
    ap.add_argument("--root", default=None, help="output root (default: folder of scene.blend)")
    ap.add_argument("--samples", type=int, default=None)
    a = ap.parse_args(argv)
    root = a.root or os.path.dirname(bpy.data.filepath) or os.getcwd()
    if a.samples:
        bpy.context.scene.cycles.samples = a.samples
    print("[eyes] device:", setup_device(), flush=True)
    if a.sequence:
        run_sequence(root)
    if a.directions:
        run_directions(root)
    if a.move:
        move(*a.move)
        rec = render(a.out)
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    _main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
