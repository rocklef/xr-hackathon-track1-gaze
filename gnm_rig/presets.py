"""The texture set catalogue: 8 diverse skin + iris presets (colours in sRGB 0-255)."""

PRESETS = [
    dict(id="T01", name="fair_young_female", base=(226, 182, 158), lip=(196, 112, 112), brow=(92, 66, 48),
         hair=(110, 80, 58), red=0.55, freckle=0.55, age=0.0, beard=0.0, scalp=0.35, brow_k=0.75, pore=0.6,
         undereye=0.35, iris="blue", iris_seed=11),
    dict(id="T02", name="fair_male_stubble", base=(214, 166, 140), lip=(172, 98, 92), brow=(70, 50, 36),
         hair=(58, 42, 30), red=0.65, freckle=0.15, age=0.15, beard=0.85, scalp=0.75, brow_k=0.9, pore=0.8,
         undereye=0.45, iris="green", iris_seed=22),
    dict(id="T03", name="olive_mediterranean_male", base=(190, 136, 104), lip=(150, 84, 74), brow=(38, 28, 22),
         hair=(30, 22, 18), red=0.4, freckle=0.0, age=0.25, beard=1.0, scalp=0.9, brow_k=1.0, pore=0.9,
         undereye=0.55, iris="brown", iris_seed=33),
    dict(id="T04", name="east_asian_female", base=(222, 178, 146), lip=(190, 108, 104), brow=(34, 26, 22),
         hair=(26, 20, 18), red=0.35, freckle=0.0, age=0.0, beard=0.0, scalp=0.9, brow_k=0.85, pore=0.45,
         undereye=0.35, iris="dark_brown", iris_seed=44),
    dict(id="T05", name="south_asian_male", base=(160, 108, 76), lip=(120, 70, 62), brow=(24, 18, 15),
         hair=(20, 15, 13), red=0.3, freckle=0.0, age=0.1, beard=0.95, scalp=0.95, brow_k=1.0, pore=0.8,
         undereye=0.75, iris="dark_brown", iris_seed=55),
    dict(id="T06", name="african_female", base=(104, 66, 46), lip=(88, 50, 44), brow=(14, 11, 10),
         hair=(12, 10, 9), red=0.2, freckle=0.0, age=0.0, beard=0.0, scalp=0.95, brow_k=0.8, pore=0.6,
         undereye=0.6, iris="very_dark", iris_seed=66),
    dict(id="T07", name="elderly_caucasian_male", base=(218, 168, 146), lip=(176, 112, 108), brow=(170, 165, 158),
         hair=(185, 182, 176), red=0.85, freckle=0.0, age=1.0, beard=0.6, scalp=0.4, brow_k=0.8, pore=1.0,
         undereye=0.7, iris="grey_blue", iris_seed=77),
    dict(id="T08", name="latina_female", base=(198, 146, 112), lip=(168, 92, 86), brow=(44, 32, 25),
         hair=(40, 28, 22), red=0.45, freckle=0.25, age=0.05, beard=0.0, scalp=0.9, brow_k=0.9, pore=0.55,
         undereye=0.45, iris="hazel", iris_seed=88),
]

BY_ID = {p["id"]: p for p in PRESETS}


def lin(c):
    """sRGB 0-255 -> linear 0-1."""
    out = []
    for v in c:
        v = v / 255.0
        out.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return tuple(out)
