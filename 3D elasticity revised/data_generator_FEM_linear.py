# make_linear_data_fast.py
# Linear (isotropic) FEM data generator (hex8, 2x2x2 Gauss).
# Pre-factorizes K_bc once (fixed E, nu, BC), then solves K_bc u = F_bc
# for each new body-force density f. Output tensors: f, u, bc_mask, bc_val.


import os, glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import splu   # LU factorization for repeated solves
import matplotlib.pyplot as plt
import random

# -----------------------
# User config
# -----------------------
N = 32                  # grid points per axis
num_samples = 1      # how many samples to generate
output_dir = "./data_linear_twopeak_64"
os.makedirs(output_dir, exist_ok=True)

# Material (linear isotropic)
E = 1.0
nu = 0.30

# Force field settings (Gaussian blob(s) in x-direction by default)
force_sigma = 0.05
num_blobs = 2
force_mag_range = (0.1, 0.2)  # (min, max) density magnitude per blob

# -----------------------
# Helpers / Indexing
# -----------------------
dx = 1.0 / (N - 1)
ndof = (N**3) * 3
TARGET_LUMPED_FORCE_TOTAL = 0.0005  # total ∑ f*dx^3 across all components

def node_id(i, j, k):
    return i * N * N + j * N + k

# def generate_gaussian_force():
#     # Random Gaussian blobs in x-direction (component 0)
#     x, y, z = np.meshgrid(np.linspace(0, 1, N),
#                           np.linspace(0, 1, N),
#                           np.linspace(0, 1, N), indexing='ij')
#     f = np.zeros((3, N, N, N), dtype=np.float32)
#     s2 = 2.0 * (force_sigma ** 2)
#     for _ in range(num_blobs):
#         cx, cy, cz = np.random.uniform(0.1, 0.9, 3)
#         mag = np.random.uniform(*force_mag_range)
#         g = mag * np.exp(-((x - cx)**2 + (y - cy)**2 + (z - cz)**2) / s2)
#         f[0] += g.astype(np.float32)
#     return f

def generate_gaussian_force():
    """
    Generate Gaussian body-force density then rescale so that the *lumped* force
    (sum over nodes of f * dx^3) hits a desired target.

    Options:
      - Scalar total target (default): set TARGET_LUMPED_FORCE_TOTAL = 1.0 (float)
        -> sum_c sum_{i,j,k} f[c,i,j,k] * dx^3 = TARGET_LUMPED_FORCE_TOTAL

      - Per-component targets: set TARGET_LUMPED_FORCE_PER_COMP = (Tx, Ty, Tz)
        -> for each component c: sum_{i,j,k} f[c,i,j,k] * dx^3 = Tc
    """
    # --- original Gaussian construction ---
    x, y, z = np.meshgrid(np.linspace(0, 1, N),
                          np.linspace(0, 1, N),
                          np.linspace(0, 1, N), indexing='ij')
    f = np.zeros((3, N, N, N), dtype=np.float32)
    s2 = 2.0 * (force_sigma ** 2)
    num_blobs = random.randint(2, 2)
    for _ in range(num_blobs):
        cx, cy, cz = np.random.uniform(0.1, 0.9, 3)
        mag = np.random.uniform(*force_mag_range)
        g = mag * np.exp(-((x - cx)**2 + (y - cy)**2 + (z - cz)**2) / s2)
        f[2] += g.astype(np.float32)  # x-direction only (as before)

    # --- rescale to hit target lumped force ---
    vol = dx**3
    # current lumped totals per component (float64 for accuracy)
    curr = vol * f.astype(np.float64).reshape(3, -1).sum(axis=1)

    # If user provided per-component targets, honor them; else use scalar total = 1.0
    if 'TARGET_LUMPED_FORCE_PER_COMP' in globals():
        target = np.array(TARGET_LUMPED_FORCE_PER_COMP, dtype=np.float64).reshape(3)
        f64 = f.astype(np.float64, copy=False)
        for c in range(3):
            if curr[c] > 0.0:
                scale = target[c] / curr[c]
                # keep f as float32
                f[c] = (f64[c] * scale).astype(np.float32)
            # if curr[c] == 0 and target[c] != 0, we leave it as 0 (no Gaussian in that comp)
    else:
        tgt_total = float(globals().get('TARGET_LUMPED_FORCE_TOTAL', 1.0))
        curr_total = curr.sum()
        if curr_total > 0.0:
            scale = tgt_total / curr_total
            f = (f.astype(np.float64) * scale).astype(np.float32)
        # if curr_total == 0, nothing to scale

    return f


# def generate_gaussian_force():
#     """
#     Uniform force on the bottom face (k = N-1) in +z with magnitude 1.
#     NOTE: If you lump RHS as F = f * dx**3, this acts like a surface load of ~1*dx.
#           For an *exact* unit traction (per area), set the face value to 1/dx instead.
#     """
#     f = np.zeros((3, N, N, N), dtype=np.float32)
#     # Option A (simple): unit "density" on the face → effective traction ≈ 1*dx
#     f[2, :, :, N-1] = 1.0

#     # --- Option B (exact unit traction): uncomment and comment Option A ---
#     # f[2, :, :, N-1] = 1.0 / dx

#     return f

# def generate_gaussian_force():
#     """
#     Uniform 'force density' on the bottom face (k = N-1) chosen so that
#     the lumped RHS F = f * dx^3 produces a total resultant of exactly 1
#     in the +z direction.
#     """
#     f = np.zeros((3, N, N, N), dtype=np.float32)
#     amp = 1.0 / (float(N * N) * (dx ** 3))  # ensures sum(Fz) == 1 after lumping
#     f[2, :, :, N-1] = amp
#     return f

def generate_bc_mask_and_value():
    # Clamp the k=0 face to zero (all 3 components)
    mask = np.zeros((3, N, N, N), dtype=np.float32)
    value = np.zeros((3, N, N, N), dtype=np.float32)
    mask[:, :, :, 0] = 1.0
    value[:, :, :, 0] = 0.0
    return mask, value

# -------------------------------------------------
# Hex8 linear elasticity: element matrices (2x2x2)
# -------------------------------------------------
def dN_dxi_hex8(xi, eta, zeta):
    c = np.array([[-1, -1, -1],
                  [ 1, -1, -1],
                  [ 1,  1, -1],
                  [-1,  1, -1],
                  [-1, -1,  1],
                  [ 1, -1,  1],
                  [ 1,  1,  1],
                  [-1,  1,  1]], dtype=np.float64)
    dN = np.zeros((3, 8), dtype=np.float64)
    for a in range(8):
        cx, cy, cz = c[a]
        dN[0, a] = 0.125 * cx * (1 + cy * eta) * (1 + cz * zeta)  # d/dxi
        dN[1, a] = 0.125 * cy * (1 + cx * xi)  * (1 + cz * zeta)  # d/deta
        dN[2, a] = 0.125 * cz * (1 + cx * xi)  * (1 + cy * eta)   # d/dzeta
    return dN

def B_matrix_from_dN_dx(dN_dx):
    B = np.zeros((6, 24), dtype=np.float64)
    for a in range(8):
        i = 3 * a
        dN = dN_dx[:, a]
        B[:, i:i+3] = np.array([
            [dN[0], 0.0,   0.0  ],
            [0.0,   dN[1], 0.0  ],
            [0.0,   0.0,   dN[2]],
            [dN[1], dN[0], 0.0  ],
            [0.0,   dN[2], dN[1]],
            [dN[2], 0.0,   dN[0]],
        ], dtype=np.float64)
    return B

def elasticity_C(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    C = np.array([
        [lam+2*mu, lam,       lam,       0,   0,   0],
        [lam,      lam+2*mu,  lam,       0,   0,   0],
        [lam,      lam,       lam+2*mu,  0,   0,   0],
        [0,        0,         0,         mu,  0,   0],
        [0,        0,         0,         0,   mu,  0],
        [0,        0,         0,         0,   0,   mu]
    ], dtype=np.float64)
    return C

def element_stiffness_hex8(dx, E, nu):
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    C = elasticity_C(E, nu)
    Ke = np.zeros((24, 24), dtype=np.float64)
    detJ = (dx/2.0)**3
    scale_grad = 2.0/dx
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN_dxi = dN_dxi_hex8(xi, eta, zeta)     # (3,8)
                dN_dx  = scale_grad * dN_dxi            # (3,8)
                B = B_matrix_from_dN_dx(dN_dx)          # (6,24)
                Ke += (B.T @ C @ B) * detJ
    return Ke

# -----------------------
# Assembly (global K, F)
# -----------------------
class LinearFEMAssembler:
    def __init__(self, N, dx, E, nu):
        self.N = N
        self.dx = dx
        self.E = E
        self.nu = nu
        self.Ke = element_stiffness_hex8(dx, E, nu)  # precompute once
        self.elem_dofs = self._make_connectivity()

    def _make_connectivity(self):
        N = self.N
        elem_dofs = []
        for i in range(N-1):
            for j in range(N-1):
                for k in range(N-1):
                    n = [
                        node_id(i+0, j+0, k+0),
                        node_id(i+1, j+0, k+0),
                        node_id(i+1, j+1, k+0),
                        node_id(i+0, j+1, k+0),
                        node_id(i+0, j+0, k+1),
                        node_id(i+1, j+0, k+1),
                        node_id(i+1, j+1, k+1),
                        node_id(i+0, j+1, k+1),
                    ]
                    dofs = []
                    for nid in n:
                        dofs += [3*nid + 0, 3*nid + 1, 3*nid + 2]
                    elem_dofs.append(dofs)
        return elem_dofs

    def assemble_stiffness_only(self):
        nelem = len(self.elem_dofs)
        max_entries = nelem * 24 * 24
        I = np.empty(max_entries, dtype=np.int32)
        J = np.empty(max_entries, dtype=np.int32)
        V = np.empty(max_entries, dtype=np.float64)
        idx = 0
        Ke = self.Ke
        for dofs in self.elem_dofs:
            rows = np.repeat(dofs, 24)
            cols = np.tile(dofs, 24)
            vals = Ke.ravel()
            I[idx:idx+576] = rows
            J[idx:idx+576] = cols
            V[idx:idx+576] = vals
            idx += 576
        return coo_matrix((V[:idx], (I[:idx], J[:idx])), shape=(ndof, ndof)).tocsc()

    def lump_force(self, f_density):
        # F[dof(node,c)] = f_density[c,i,j,k] * dx^3
        F = np.zeros(ndof, dtype=np.float64)
        vol = self.dx ** 3
        c0 = f_density[0].reshape(-1)
        c1 = f_density[1].reshape(-1)
        c2 = f_density[2].reshape(-1)
        # node order in flatten must match node_id(i,j,k): (i major, then j, then k)
        # np.reshape with default C-order matches (i,j,k) indexing when created with meshgrid(..., indexing='ij')
        F[0::3] = c0 * vol
        F[1::3] = c1 * vol
        F[2::3] = c2 * vol
        return F

def apply_dirichlet_to_stiffness(K_csc, bc_mask):
    """
    Enforce u=bc_val on masked DOFs by zeroing row & column and setting diag=1.
    Returns K_bc (CSC). Columns are zeroed so nonzero bc_val won't leak into free eqs.
    """
    K = K_csc.tolil(copy=True)
    cons = []
    for c in range(3):
        ii, jj, kk = np.where(bc_mask[c] > 0.5)
        for i, j, k in zip(ii, jj, kk):
            cons.append(3*node_id(i, j, k) + c)
    if len(cons) == 0:
        return K_csc  # nothing to do

    # Zero columns first (LIL supports assignment to slices)
    for dof in cons:
        K[:, dof] = 0.0
    # Then zero rows and set diagonal to 1
    for dof in cons:
        K.rows[dof] = [dof]
        K.data[dof] = [1.0]
    return K.tocsc()

def apply_dirichlet_to_rhs(F, bc_mask, bc_val):
    """
    Sets constrained DOF entries of RHS to the prescribed value.
    (With columns zeroed above, this is sufficient even for nonzero bc_val.)
    """
    F = F.copy()
    for c in range(3):
        ii, jj, kk = np.where(bc_mask[c] > 0.5)
        for i, j, k in zip(ii, jj, kk):
            dof = 3*node_id(i, j, k) + c
            F[dof] = float(bc_val[c, i, j, k])
    return F

def visualize_force_displacement(f, u):
    sl = N // 2
    fz = f[2, :, :, sl]
    uz = u[2, :, :, sl]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(fz, cmap='hot', origin='lower')
    axs[0].set_title("Force (z-dir) at Z mid-slice")
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(uz, cmap='viridis', origin='lower')
    axs[1].set_title("Displacement (z-dir) at Z mid-slice")
    plt.colorbar(im1, ax=axs[1])
    for ax in axs:
        ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.tight_layout(); plt.show()

# -----------------------
# Main generation loop
# -----------------------
def main():
    print(f"FAST Linear FEM generator | N={N}, dx={dx:.6f}, E={E}, nu={nu}")
    print(f"Output: {output_dir}")
    t0 = time.time()

    assembler = LinearFEMAssembler(N, dx, E, nu)

    # 1) Build global stiffness once
    print("Assembling global stiffness K ...")
    K = assembler.assemble_stiffness_only()

    # 2) Build BCs once and pre-factorize K_bc
    bc_mask, bc_val = generate_bc_mask_and_value()
    print("Applying Dirichlet BC to K and pre-factorizing ...")
    K_bc = apply_dirichlet_to_stiffness(K, bc_mask)      # CSC
    lu = splu(K_bc)                                      # SuperLU factorization

    t_setup = time.time()
    print(f"Setup time (assemble + BC + factorize): {t_setup - t0:.2f} s")

    # 3) Generate many samples, only new F and a backsolve each time
    for i in tqdm(range(num_samples), desc="Generating linear FEM samples (fast)"):
        # force density and nodal RHS
        f = generate_gaussian_force()                    # [3,N,N,N], float32
        F = assembler.lump_force(f)                      # float64

        # Apply BCs to RHS (cheap)
        F_bc = apply_dirichlet_to_rhs(F, bc_mask, bc_val)

        # Solve via pre-factorized K_bc
        u_flat = lu.solve(F_bc)                          # float64
        u = u_flat.reshape(N, N, N, 3).transpose(3, 0, 1, 2).astype(np.float32)

        # Save
        sample = {
            "f": torch.from_numpy(f),                                        # body-force density
            "u": torch.from_numpy(u),
            "bc_mask": torch.from_numpy(bc_mask.astype(np.float32)),
            "bc_val": torch.from_numpy(bc_val.astype(np.float32)),
        }
        fname = os.path.join(output_dir, f"validation_{i:04d}_N{N}_linear_fem.pt")
        torch.save(sample, fname)

        # Quick residual check on first sample
        if i == 0:
            res = K_bc @ u_flat - F_bc
            rel_res = np.linalg.norm(res) / (np.linalg.norm(F_bc) + 1e-16)
            print(f"[GEN CHECK] sample {i:04d}  rel_residual = {rel_res:.3e}")
            visualize_force_displacement(f, u)

    print(f"Saved {num_samples} samples to: {output_dir}")
    print(f"Total time: {time.time() - t0:.2f} s  |  Avg per sample (after setup): "
          f"{(time.time() - t_setup)/max(1, num_samples):.3f} s")

if __name__ == "__main__":
    main()
