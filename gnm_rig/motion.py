"""Natural eye-motion synthesis (pure NumPy): turns a short script of fixations into per-frame channels.

Human-like rules used:
  * saccades follow the "main sequence": duration ~ 21 ms + 2.2 ms/deg, with a minimum-jerk position profile;
  * fixations are never perfectly still: slow drift plus micro-saccades (0.2-0.5 deg every 0.4-0.9 s);
  * blinks every ~2-6 s (fast close ~90 ms, brief hold, slower reopen ~190 ms); big gaze shifts (>25 deg) often
    carry a blink; occasional half-blinks;
  * eye-head coordination: for shifts over ~10 deg the head turns ~25% of the way, starting ~50 ms after the eyes
    with a smooth lag, while the eyes counter-rotate so gaze stays on target (vestibulo-ocular reflex);
  * eyelids follow vertical gaze (done by the rig drivers); brows lift for upward eye-rolls.
Channels per frame: gaze_h, gaze_v (deg, world), head_yaw, head_pitch (deg), eye_h, eye_v (deg, in head),
blink, brow_raise, wide, squint, pupil.
"""
import numpy as np

FPS = 30


def min_jerk(t):
    t = np.clip(t, 0, 1)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def saccade_duration(amp_deg):
    return 0.021 + 0.0022 * amp_deg


# --------------------------------------------------------------------------- scripts for the 3 inputs
def script(kind):
    """List of events. ('fix', t, h, v) = start a saccade at time t to (h, v); ('roll', t, dur, pts);
    ('blink', t[, amp]); ('brow', t, dur, amount); ('end', t)."""
    if kind in ("left", "right"):
        s = -1 if kind == "left" else 1
        return [("fix", 0.0, 0, 0), ("blink", 0.55),
                ("fix", 1.1, 26 * s, -2), ("fix", 1.75, 24 * s, 2.5),            # land + small corrective
                ("fix", 2.6, 29 * s, -4), ("blink", 3.35, 0.55),
                ("fix", 3.9, 22 * s, 1),
                ("fix", 4.75, 0, 0), ("blink", 4.8),                              # return with a blink
                ("end", 6.0)]
    if kind == "motion":
        return [("fix", 0.0, 0, 0),
                ("fix", 0.7, -24, 3), ("fix", 1.5, 23, 2), ("blink", 1.52),       # look left, then right
                ("fix", 2.3, 18, 14),                                              # up-right
                ("fix", 3.0, -18, 8),
                ("roll", 3.6, 1.0, [(-18, 8), (-16, 22), (0, 30), (16, 22), (20, 6)]),   # eye-roll upward
                ("brow", 3.5, 1.4, 0.75),
                ("blink", 4.75), ("fix", 4.95, 0, -3),
                ("fix", 5.8, -12, -22), ("fix", 6.35, 4, -21), ("fix", 6.9, 16, -22),    # reading glances, down
                ("blink", 7.45, 0.5),
                ("fix", 7.9, 0, 0), ("blink", 8.6), ("blink", 8.95),                   # double blink
                ("fix", 9.6, -9, 4), ("fix", 10.4, 6, -1),
                ("fix", 11.0, 0, 0),
                ("end", 12.0)]
    if kind == "expression":
        return [("fix", 0.0, 0, 0), ("blink", 0.6),
                ("ctrl", 1.0, 1.7, "brow_raise", 1.0), ("ctrl", 1.0, 1.7, "wide", 0.9),      # surprise
                ("ctrl", 1.0, 1.9, "pupil", 1.2), ("fix", 1.05, 0, 7),
                ("fix", 2.9, -20, 1), ("fix", 3.6, 20, 2), ("blink", 3.62),                  # look left / right
                ("ctrl", 4.5, 2.0, "squint", 0.85), ("ctrl", 4.5, 2.0, "brow_down", 0.9),     # suspicious glance
                ("fix", 4.6, 15, -3), ("fix", 5.5, 9, -2),
                ("roll", 6.7, 1.0, [(-16, 6), (-12, 20), (0, 27), (12, 20), (16, 5)]),       # eye-roll
                ("brow", 6.6, 1.4, 0.8),
                ("blink", 7.9), ("fix", 8.3, 0, -14), ("ctrl", 8.3, 1.2, "pupil", -0.6),     # look down
                ("fix", 9.3, 0, 0), ("blink", 9.6),
                ("end", 10.5)]
    raise ValueError(kind)


def synthesize(kind, fps=FPS, seed=3):
    rng = np.random.default_rng(seed)
    ev = script(kind)
    T = [e[1] for e in ev if e[0] == "end"][0]
    n = int(round(T * fps))
    t = np.arange(n) / fps
    gh, gv = np.zeros(n), np.zeros(n)
    # --- fixations / saccades
    fixes = sorted([e for e in ev if e[0] in ("fix", "roll")], key=lambda e: e[1])
    cur = (0.0, 0.0)
    for i, e in enumerate(fixes):
        t0 = e[1]
        t_next = fixes[i + 1][1] if i + 1 < len(fixes) else T
        sel = t >= t0
        if e[0] == "fix":
            tgt = (float(e[2]), float(e[3]))
            amp = np.hypot(tgt[0] - cur[0], tgt[1] - cur[1])
            dur = saccade_duration(amp)
            p = min_jerk((t[sel] - t0) / max(dur, 1e-3))
            gh[sel] = cur[0] + (tgt[0] - cur[0]) * p
            gv[sel] = cur[1] + (tgt[1] - cur[1]) * p
            cur = tgt
        else:  # smooth eye-roll along a path (pursuit-like, slower than a saccade)
            dur, pts = e[2], np.array(e[3], dtype=float)
            u = min_jerk((t[sel] - t0) / dur) * (len(pts) - 1)
            k = np.clip(u.astype(int), 0, len(pts) - 2)
            f = (u - k)[:, None]
            path = pts[k] * (1 - f) + pts[k + 1] * f
            gh[sel], gv[sel] = path[:, 0], path[:, 1]
            cur = tuple(pts[-1])
        # fixation jitter until the next event: drift + micro-saccades
        fsel = (t >= t0 + 0.12) & (t < t_next)
        if fsel.any():
            tt = t[fsel]
            drift = np.cumsum(rng.normal(0, 0.035, (fsel.sum(), 2)), 0)
            drift -= np.linspace(0, 1, fsel.sum())[:, None] * drift[-1]       # stays anchored
            micro = np.zeros((fsel.sum(), 2))
            tm = t0 + 0.12 + rng.uniform(0.35, 0.8)
            off = np.zeros(2)
            while tm < t_next - 0.1:
                off = off * 0.3 + rng.normal(0, 0.3, 2)
                micro[tt >= tm] = off
                tm += rng.uniform(0.4, 0.9)
            gh[fsel] += drift[:, 0] + micro[:, 0]
            gv[fsel] += drift[:, 1] + micro[:, 1]
    # --- head follows large gaze shifts (25%), lagged + smoothed
    head_goal_h = np.where(np.abs(gh) > 10, 0.25 * gh, 0.08 * gh)
    head_goal_v = np.where(np.abs(gv) > 10, 0.18 * gv, 0.05 * gv)
    lag = int(round(0.05 * fps))
    head_goal_h = np.r_[np.full(lag, head_goal_h[0]), head_goal_h[:-lag]] if lag else head_goal_h
    head_goal_v = np.r_[np.full(lag, head_goal_v[0]), head_goal_v[:-lag]] if lag else head_goal_v
    hy, hp = np.zeros(n), np.zeros(n)
    vh = vp = 0.0
    w = 2 * np.pi * 1.6                         # critically damped follower (~0.25 s)
    dt = 1.0 / fps
    for i in range(1, n):
        for arr, goal, vel_name in ((hy, head_goal_h, "vh"), (hp, head_goal_v, "vp")):
            v = vh if vel_name == "vh" else vp
            a = w * w * (goal[i] - arr[i - 1]) - 2 * w * v
            v = v + a * dt
            arr[i] = arr[i - 1] + v * dt
            if vel_name == "vh":
                vh = v
            else:
                vp = v
    hy += 0.4 * np.sin(2 * np.pi * 0.21 * t + 1.0)   # tiny postural sway
    hp += 0.3 * np.sin(2 * np.pi * 0.17 * t)
    # --- blinks
    blink = np.zeros(n)
    blink_times = [(e[1], e[2] if len(e) > 2 else 1.0) for e in ev if e[0] == "blink"]
    for tb, amp in blink_times:
        tc, th, to = 0.09, 0.03, 0.19
        x = t - tb
        b = np.where(x < 0, 0, np.where(x < tc, min_jerk(x / tc),
                     np.where(x < tc + th, 1, np.where(x < tc + th + to, 1 - min_jerk((x - tc - th) / to), 0))))
        blink = np.maximum(blink, amp * b)
    # --- brows / wide (eye-roll), pupil breathing
    brow = np.zeros(n)
    for e in ev:
        if e[0] == "brow":
            t0, dur, amt = e[1], e[2], e[3]
            x = (t - t0) / dur
            brow = np.maximum(brow, amt * np.sin(np.pi * np.clip(x, 0, 1)) ** 0.7)
    up = np.clip((gv - 12) / 15, 0, 1)
    brow = np.maximum(brow, 0.35 * up)
    wide = 0.25 * up
    pupil = -1.0 + 0.15 * np.sin(2 * np.pi * 0.12 * t)    # daylight pupil (~3.5 mm) + hippus
    squint = np.zeros(n)
    brow_down = np.zeros(n)
    chans = {"brow_raise": brow, "wide": wide, "pupil": pupil, "squint": squint, "brow_down": brow_down}
    for e in ev:
        if e[0] == "ctrl":
            t0, dur, name, amt = e[1], e[2], e[3], e[4]
            x = (t - t0) / dur
            env = np.where(x < 0.18, min_jerk(x / 0.18), np.where(x < 0.75, 1.0, 1 - min_jerk((x - 0.75) / 0.25)))
            env = np.where((x < 0) | (x > 1), 0, env)
            chans[name] += amt * env
    eye_h, eye_v = gh - hy, gv - hp
    return dict(t=t, gaze_h=gh, gaze_v=gv, head_yaw=hy, head_pitch=hp, eye_h=eye_h, eye_v=eye_v, blink=blink,
                brow_raise=np.clip(brow, 0, 1), wide=np.clip(wide, 0, 1), squint=np.clip(squint, 0, 1),
                brow_down=np.clip(brow_down, 0, 1), pupil=np.clip(pupil, -1, 2), fps=fps)


if __name__ == "__main__":
    for k in ("left", "right", "motion", "expression"):
        d = synthesize(k)
        print(k, len(d["t"]), "frames", f"{len(d['t']) / FPS:.1f}s", "max |eye|",
              round(float(np.abs(d["eye_h"]).max()), 1), round(float(np.abs(d["eye_v"]).max()), 1),
              "blinks peak", round(float(d["blink"].max()), 2))
