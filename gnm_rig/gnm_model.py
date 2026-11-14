"""Minimal NumPy evaluator for Google's GNM v3 head model (Apache-2.0).

Re-implements the forward pass of github.com/google/GNM (gnm/shape/gnm_xnp.py, gnm_common.py) so it can run
inside Blender's Python without extra packages:

    vertices = LBS( template + B_id @ identity + B_expr @ expression + correctives(R), joints(identity), R )

GNM space is metres, +Y up, the face looking down +Z. `to_blender()` converts to Blender (Z up, face toward -Y).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NPZ = os.environ.get("GNM_NPZ", os.path.join(ROOT, "_downloads", "gnm", "model", "gnm_head.npz"))

# UDIM tiles (gnm/shape/gnm_udim.py): one tile per textured region.
TILES = {"skin": 1001, "left_eye_interior": 1002, "left_eye_exterior": 1003, "right_eye_interior": 1004,
         "right_eye_exterior": 1005, "upper_teeth_and_gums": 1006, "lower_teeth_and_gums": 1006, "tongue": 1007}

# GNM -> Blender: (x, y, z) -> (x, -z, y)  (proper rotation, det = +1)
G2B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def to_blender(p):
    return np.asarray(p) @ G2B.T


def rodrigues(aa):
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    aa = np.asarray(aa, dtype=np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, aa / np.maximum(th, 1e-12), 0.0)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    K = np.stack([np.stack([z, -kz, ky], -1), np.stack([kz, z, -kx], -1), np.stack([-ky, kx, z], -1)], -2)
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


class GNM:
    def __init__(self, path=NPZ):
        d = np.load(path, allow_pickle=False)
        self.d = d
        self.T = d["template_vertex_positions"].astype(np.float64)
        self.J0 = d["template_joint_positions"].astype(np.float64)
        self.B_id = d["vertex_identity_basis"]
        self.J_id = d["joint_identity_basis"]
        self.B_ex = d["expression_basis"]
        self.W = d["skinning_weights"].astype(np.float64)            # (J, V)
        self.parents = [int(i) for i in d["joint_parent_indices"]]
        self.P = d["pose_correctives_regressor"].astype(np.float64)  # (9J, 3V)
        self.quads = d["quads"]
        self.quad_uvs = d["quad_uvs"]
        self.tris = d["triangles"]
        self.joint_names = [str(s) for s in d["joint_names"]]
        self.expr_names = [str(s) for s in d["expression_names"]]
        self.id_names = [str(s) for s in d["identity_names"]]
        self.groups = {str(n): d["vertex_groups"][i] for i, n in enumerate(d["vertex_group_names"])}
        self.nv = self.T.shape[0]

    # ---- parameter helpers
    def expr_index(self, prefix):
        return np.array([i for i, n in enumerate(self.expr_names) if n.startswith(prefix)])

    def zeros(self):
        return np.zeros(self.B_id.shape[0]), np.zeros(self.B_ex.shape[0]), np.zeros((len(self.parents), 3))

    # ---- forward pass
    def bind(self, identity, expression):
        v = self.T + np.tensordot(identity, self.B_id, 1) + np.tensordot(expression, self.B_ex, 1)
        j = self.J0 + np.tensordot(identity, self.J_id, 1)
        return v, j

    def correctives(self, R):
        f = (R - np.eye(3)).reshape(-1)
        return (f @ self.P).reshape(self.nv, 3)

    def world_transforms(self, joints, R, translation=np.zeros(3)):
        G = []
        for i, p in enumerate(self.parents):
            t = joints[i] + translation if p < 0 else joints[i] - joints[p]
            L = np.eye(4)
            L[:3, :3], L[:3, 3] = R[i], t
            G.append(L if p < 0 else G[p] @ L)
        return np.stack(G)

    def __call__(self, identity=None, expression=None, rotations=None, translation=np.zeros(3)):
        i0, e0, r0 = self.zeros()
        identity = i0 if identity is None else identity
        expression = e0 if expression is None else expression
        rotations = r0 if rotations is None else rotations
        v, j = self.bind(identity, expression)
        R = rodrigues(rotations)
        v = v + self.correctives(R)
        G = self.world_transforms(j, R, translation)
        # LBS about the bind joints: x' = sum_j w_j * G_j @ (x - j_j)
        A = G.copy()
        A[:, :3, 3] = G[:, :3, 3] - np.einsum("jab,jb->ja", G[:, :3, :3], j)
        M = np.einsum("jv,jab->vab", self.W, A)
        return np.einsum("vab,vb->va", M[:, :3, :3], v) + M[:, :3, 3], G

    # ---- topology helpers
    def face_regions(self):
        """Region name per quad (by majority of its vertices' groups)."""
        g = self.groups
        regions = np.full(len(self.quads), "skin", dtype=object)
        for side in ("left", "right"):
            eye = g[f"{side}_eye"] > 0.5
            interior, exterior = g["eye_interiors"] > 0.5, g["eye_exteriors"] > 0.5
            regions[(eye & interior)[self.quads].mean(1) > 0.5] = f"{side}_eye_interior"
            regions[(eye & exterior)[self.quads].mean(1) > 0.5] = f"{side}_eye_exterior"
        for name in ("upper_teeth_and_gums", "lower_teeth_and_gums", "tongue"):
            regions[(g[name] > 0.5)[self.quads].mean(1) > 0.5] = name
        return regions

    def quad_uvs_udim(self):
        """Per-region UVs shifted into their UDIM tile (u + tile - 1001)."""
        reg = self.face_regions()
        off = np.array([TILES[r] - 1001 for r in reg], dtype=np.float64)
        uv = self.quad_uvs.astype(np.float64).copy()
        uv[..., 0] += off[:, None]
        return uv, reg
