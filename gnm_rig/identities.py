"""GNM's semantic IdentitySampler (a conditional-VAE decoder, Apache-2.0) evaluated in plain NumPy.

The official sampler (gnm/shape/semantic_sampler.py) runs a small Keras MLP:
  concat(latent[64], one-hot gender[2] + ethnicity[4]) -> 64 -> 128 -> 256 -> 512 (ReLU) -> 253 identity coeffs.
We export its weights once (from identity_decoder_model.h5) to identity_decoder.npz and evaluate it here, so no
TensorFlow is needed. Labels may be blended (weights per class), as in the official blend_* helpers.

python gnm_rig/identities.py --export PATH/identity_decoder_model.h5     (one-off)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "identity_decoder.npz")
GENDER = {"female": 0, "male": 1}
ETHNICITY = {"middle_eastern": 0, "asian": 1, "white": 2, "black": 3}

# texture set -> identity labels (gender, {ethnicity: weight}, seed)
MATCH = {
    "T01": ("female", {"white": 1.0}, 3),
    "T02": ("male", {"white": 1.0}, 5),
    "T03": ("male", {"middle_eastern": 0.7, "white": 0.3}, 2),
    "T04": ("female", {"asian": 1.0}, 4),
    "T05": ("male", {"asian": 0.5, "middle_eastern": 0.5}, 6),
    "T06": ("female", {"black": 1.0}, 1),
    "T07": ("male", {"white": 1.0}, 9),
    "T08": ("female", {"middle_eastern": 0.5, "white": 0.5}, 7),
}


def export(h5_path):
    import h5py
    f = h5py.File(h5_path, "r")
    w = {}
    for i in range(4, 9):
        g = f["model_weights"][f"dense_{i}"][f"dense_{i}"]
        w[f"W{i}"] = np.array(g["kernel:0"])
        w[f"b{i}"] = np.array(g["bias:0"])
    np.savez(NPZ, **w)
    return NPZ


def sample(gender="female", ethnicity=None, seed=0, temperature=0.8):
    w = np.load(NPZ)
    lab = np.zeros(6, dtype=np.float32)
    lab[GENDER[gender]] = 1.0
    eth = ethnicity or {"white": 1.0}
    tot = sum(eth.values())
    for k, v in eth.items():
        lab[2 + ETHNICITY[k]] = v / tot
    z = np.random.default_rng(seed).normal(size=64).astype(np.float32) * temperature
    x = np.concatenate([z, lab])
    for i in range(4, 8):
        x = np.maximum(0, x @ w[f"W{i}"] + w[f"b{i}"])
    return (x @ w["W8"] + w["b8"]).astype(np.float64)


def for_texture(tex_id):
    g, e, seed = MATCH[tex_id]
    return sample(g, e, seed)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--export":
        print("exported", export(sys.argv[2]))
    else:
        for k in MATCH:
            v = for_texture(k)
            print(k, MATCH[k][0], MATCH[k][1], "|id| max", round(float(np.abs(v).max()), 2))
