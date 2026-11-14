# GNM head: textures, eye motion and expressions

This folder rebuilds the Track 1 deliverables on **Google's GNM v3 head model** (Apache-2.0). That is the
topology the evaluation uses ("textures will be directly wrapped around GNM Head Model").

| Rubric item | Points | Deliverable | Where |
|---|---|---|---|
| Texture photorealism | 60 | 8 texture sets on GNM's UV layout (skin, both eyes, teeth, tongue), a render script, and renders | `../textures/`, `../render_textures.py` (wraps `render_textures.py` here), `../renders_textures/` |
| Kinematic accuracy | 20 | One script with three inputs (`left`, `right`, `motion`) that renders frames and a video for each, each ≤ 15 s | `../eyes_cli.py` (wraps `eyes_cli.py` here), `../renders_eyes_cli/` |
| Code & UV portability | 20 | A `.blend` that loads the GNM head and plays an expression on the Timeline (eyelids, iris/pupil, eyebrows, gaze) | `../gnm_expression_timeline.blend` (repo root) |

## Quick start

```bash
# 0. one-off: GNM model (53 MB, Apache-2.0) -> _downloads/gnm/model/gnm_head.npz
#    https://huggingface.co/google/gnm-v3/resolve/main/v3_0/gnm_head.npz
# 1. build the rigged head (mesh + UDIM UVs + rig + expression shape keys + eyelashes)
blender -b --factory-startup -P gnm_rig/build_head.py
# 2. textures: bake skin maps in Blender, then eyes, brows (system Python + Pillow/SciPy)
blender -b --factory-startup gnm_rig/out/gnm_head.blend -P gnm_rig/textures.py
python gnm_rig/eye_tiles.py
python gnm_rig/brow_hairs.py
# 3. renders with each texture set
blender -b --factory-startup gnm_rig/out/gnm_head.blend -P gnm_rig/render_textures.py -- --tex all
# 4. eye-motion videos
python gnm_rig/eyes_cli.py --input left
python gnm_rig/eyes_cli.py --input right
python gnm_rig/eyes_cli.py --input motion
```

## Textures (GNM UV, UDIM)

Each set is `textures/<ID>_<name>/<ID>_{albedo,normal,roughness}.<UDIM>.png`:

| Tile | Region | Resolution |
|---|---|---|
| 1001 | Skin (head, face, ears, neck, inner lids and mouth lining) | 4096² albedo, normal and roughness |
| 1002 | Left eye: sclera, iris, pupil | 2048² |
| 1004 | Right eye: a **separate** iris map, same colour family with a different pattern | 2048² |
| 1003, 1005 | Corneas (clear, no map needed) | — |
| 1006, 1007 | Teeth and gums, tongue | 512² |

In Blender, load one channel as a single tiled image (`<ID>_albedo.<UDIM>.png`) on GNM's UV map, and every region
gets its texture.

- **Seam-free by construction.** Every skin layer is generated in 3D on the GNM surface and baked through GNM's own
  UVs, so neighbouring texels across a UV seam come from the same 3D point.
- **Skin layers.** Melanin base tone and mottling; haemoglobin redness on the nose, cheeks and ears; under-eye
  shadow; vermilion lips with fine lines; pores at about 0.4 mm (stronger in the T-zone); freckles and age spots;
  stubble and buzz-cut follicles; forehead lines and crow's feet that scale with age. Region masks come from GNM's
  own vertex groups (`skin_masks.py`).
- **Eyebrows.** About 900 individual hairs per brow. They are grown with an anatomical flow field (upward at the
  inner end, herringbone through the body, outward at the tail) and drawn into UV space (`brow_hairs.py`).
- **Irises.** Radial stroma fibres, collarette, crypts, pigment flecks, a limbal ring and a sclera with veins
  (`iris.py`). They follow GNM's radial eye UV: pupil r < 0.05, iris r < 0.185.

| ID | Set | Iris |
|---|---|---|
| T01 | fair, young female | blue |
| T02 | fair male, stubble | green |
| T03 | olive / Mediterranean male | brown |
| T04 | East Asian female | dark brown |
| T05 | South Asian male | dark brown |
| T06 | African female | very dark |
| T07 | elderly Caucasian male, wrinkles and age spots | grey-blue |
| T08 | Latina female | hazel |

`identities.py` runs GNM's official semantic **IdentitySampler** (its conditional-VAE decoder, ported to NumPy) to give
each set a matching face shape for the renders. The textures themselves fit every GNM identity, because the UV
layout is shared.

## Rig and expressions

- **Rig.** Uses GNM's 4 joints (`neck`, `head`, `eye.L`, `eye.R`) and its skinning weights. Each eye bone sits at the
  eyeball centre on GNM's optical axis. Both eyes track `GAZE_Target` (Damped Track), so they converge naturally.
- **Expressions.** GNM's eye-region expression space has 100 unnamed PCA components per eye. `expressions.py`
  solves for semantic shapes inside that space using a ridge-regularised least-squares fit to eyelid and brow
  targets: blink, wide, squint, lids-up, lids-down, brow-raise and brow-down per side. GNM's own `pupils_000`
  component gives the pupil size. Because each shape is a combination of scan-derived components, the whole lid,
  brow and skin move the way real faces do.
- **Lids follow gaze automatically.** GNM's pose correctives don't move the lids for eye rotation, so drivers add
  lids-up and lids-down from the gaze pitch.
- **Controls.** Custom properties on `GNM_Rig`, all keyframeable: `blink_L`, `blink_R`, `wide`, `squint`,
  `brow_raise`, `brow_down`, `pupil` and `auto_lids`. Gaze is set by moving `GAZE_Target`.
- **Eyelashes.** 140 tapered, curled strands per eye. They are rooted on GNM's lid margins and bound with Surface
  Deform, so they ride along with blinks.

## Natural eye motion (`motion.py`)

- **Saccades** follow the main sequence (21 ms + 2.2 ms/°) with a minimum-jerk profile.
- **Fixations** have drift and micro-saccades.
- **Blinks** close fast (about 90 ms) and reopen more slowly (about 190 ms). Large gaze shifts carry a blink, and
  there are occasional half-blinks.
- **The head follows** large shifts by about 25%, with a 50 ms lag, while the eyes counter-rotate (vestibulo-ocular
  reflex).
- **Brows lift** during upward eye-rolls.

`left` and `right` are 6 s each, `motion` is 12 s, and `expression` (the Timeline file) is 10.5 s. Directions are
from the viewer's side.
