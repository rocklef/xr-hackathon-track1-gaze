"""Open scene.blend in Blender's UI as a live demo (does not modify the file).

    blender scene.blend --python tools/demo_view.py

* Viewport switches to Material Preview, looking through CAM_Gaze.
* The fixation target (TGT_Fixation) is animated around the +-5 cm circle,
  clockwise from the top, and playback starts, so the eyes (and lids)
  follow it live. Press Space to pause, Esc to stop.
"""
import math

import bpy

N = 120  # frames per loop


def setup():
    scene = bpy.context.scene
    tgt = bpy.data.objects["TGT_Fixation"]
    bound = float(scene["gaze_cfg"]["motion_bound_m"])
    tgt.animation_data_clear()
    for f in range(N + 1):
        th = 2 * math.pi * f / N
        tgt.location = (math.sin(th) * bound, 0.0, math.cos(th) * bound)
        tgt.keyframe_insert("location", frame=f + 1)
    scene.frame_start, scene.frame_end = 1, N
    scene.frame_set(1)
    tgt.hide_viewport = False
    tgt.empty_display_size = 0.006

    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            sp = area.spaces.active
            sp.shading.type = "MATERIAL"
            sp.region_3d.view_perspective = "CAMERA"
            sp.show_region_toolbar = False
            sp.show_region_ui = False
            for reg in area.regions:
                if reg.type == "WINDOW":
                    with bpy.context.temp_override(window=win, area=area, region=reg):
                        bpy.ops.view3d.view_center_camera()
            with bpy.context.temp_override(window=win, area=area):
                bpy.ops.screen.animation_play()
            return None
    return None


bpy.app.timers.register(setup, first_interval=1.5)
