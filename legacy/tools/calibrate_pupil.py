"""Calibrate the apparent (entrance) pupil position, then save it into scene.blend.

    blender -b scene.blend -P tools/calibrate_pupil.py

Why: the pupil is seen through the refracting cornea and is slightly
decentred, so the visible pupil is not exactly the front-surface point on the
optical axis. We render catchlight-free views at several gaze directions
(lights off, uniform world), detect each pupil from the pixels, and fit a 3D
offset o (in the eye bone's local frame, mirrored between L and R) such that
    project(eye_centre + R_eye @ o) == detected pupil        (least squares)
The result is stored as scene["gaze_cfg"]["pupil_offset_local_L/R"]; the
wrapper then reports pupil_center_2D from that physically-placed point.
Lid-gaze coupling keeps the pupil uncovered, so up-gaze poses are included.
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "wrapper"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eyes  # noqa: E402
from pupil_detect import detect_green  # noqa: E402

TMP = os.path.join(ROOT, "_calib")
POSES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-0.7, -0.7), (0.7, -0.7),
         (-0.7, 0.7), (0.7, 0.7), (-1, -0.4), (1, -0.4), (-1, 0.4), (1, 0.4)]
MIRROR = np.array([-1.0, 1.0, 1.0])  # bone-local X flips between left and right eye


def load_rgb(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    return a * 255.0


def eye_frame(side):
    rig = bpy.data.objects[eyes.RIG]
    pb = rig.pose.bones[eyes.BONES[side]]
    c = rig.matrix_world @ pb.head
    R = (rig.matrix_world.to_3x3() @ pb.matrix.to_3x3()).normalized()
    return c, R


def main():
    scene = bpy.context.scene
    cfg = scene["gaze_cfg"]
    eyes.setup_device()
    lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    saved = {o.name: o.hide_render for o in lights}
    for o in lights:
        o.hide_render = True
    # catchlight-free: uniform grey world, lights off
    world_old, samples_old = scene.world, scene.cycles.samples
    flat = bpy.data.worlds.new("WLD_CalibFlat")
    flat.use_nodes = True
    flat.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    flat.node_tree.nodes["Background"].inputs[1].default_value = 1.2
    scene.world, scene.cycles.samples = flat, 48
    # chroma-key: paint the pupil pure green for calibration only
    eye_grp = next(n for n in bpy.data.objects[eyes.EYES].material_slots[0].material.node_tree.nodes
                   if n.type == "GROUP")
    pupil_old = tuple(eye_grp.inputs["PupilColor"].default_value)
    eye_grp.inputs["PupilColor"].default_value = (0.0, 1.0, 0.0, 1.0)
    hidden = [o for o in bpy.data.objects if o.name.startswith("SM_Backdrop") and not o.hide_render]
    for ob in hidden:
        ob.hide_render = True

    obs = []  # (side, centre, R, detected_px)
    os.makedirs(TMP, exist_ok=True)
    for i, (h, v) in enumerate(POSES):
        eyes.move(h, v)
        path = os.path.join(TMP, f"cal_{i:02d}.png")
        eyes.render(path)
        rgb = load_rgb(path)
        for side in "LR":
            c, R = eye_frame(side)
            d0 = float(cfg[f"pupil_depth_{side}"])
            guess = eyes._pixel(scene.camera, scene, c + R @ Vector((0, d0, 0)))
            det = detect_green(rgb, guess)
            if det is None:
                print(f"[calib] pose {i} eye {side}: no pupil found")
                continue
            obs.append((side, c.copy(), R.copy(), np.array(det)))

    def residuals(oL):
        res = []
        for side, c, R, det in obs:
            o = oL if side == "L" else oL * MIRROR
            p = eyes._pixel(scene.camera, scene, c + R @ Vector(tuple(o)))
            res.extend([p[0] - det[0], p[1] - det[1]])
        return np.array(res)

    # Gauss-Newton on the 3 offset parameters (metres)
    o = np.array([0.0, float(cfg["pupil_depth_L"]), 0.0])
    r0 = residuals(o)
    rms_surface = float(np.sqrt((r0 ** 2).mean() * 2))
    for _ in range(8):
        r = residuals(o)
        J = np.zeros((len(r), 3))
        for k in range(3):
            dv = np.zeros(3)
            dv[k] = 1e-4
            J[:, k] = (residuals(o + dv) - r) / 1e-4
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        o = o + step
        if np.linalg.norm(step) < 1e-7:
            break
    rf = residuals(o)
    per_obs = np.hypot(rf[0::2], rf[1::2])
    rms = float(np.sqrt((per_obs ** 2).mean()))
    print(f"[calib] n={len(obs)}  surface-point RMS={rms_surface:.2f}px  ->  fitted RMS={rms:.2f}px  "
          f"max={per_obs.max():.2f}px")
    print(f"[calib] pupil offset (eye-local, mm): x={o[0]*1000:+.2f} depth={o[1]*1000:.2f} z={o[2]*1000:+.2f}")

    for l in lights:
        l.hide_render = saved[l.name]
    scene.world, scene.cycles.samples = world_old, samples_old
    eye_grp.inputs["PupilColor"].default_value = pupil_old
    bpy.data.worlds.remove(flat)
    for ob in hidden:
        ob.hide_render = False
    cfg["pupil_offset_local_L"] = [float(x) for x in o]
    cfg["pupil_offset_local_R"] = [float(x) for x in o * MIRROR]
    cfg["pupil_calibration"] = {"rms_px": rms, "max_px": float(per_obs.max()),
                                "rms_px_surface_point": rms_surface, "n_observations": len(obs)}
    eyes.reset()
    with open(os.path.join(TMP, "calibration.json"), "w") as f:
        json.dump(dict(offset_L=o.tolist(), rms=rms, rms_surface=rms_surface,
                       residuals_px=per_obs.tolist(), poses=POSES), f, indent=1)
    bpy.ops.wm.save_mainfile()
    print("[calib] saved scene.blend")


main()
