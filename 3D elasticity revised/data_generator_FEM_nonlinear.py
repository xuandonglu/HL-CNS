# ==== DROP-IN: Vectorized bilinear (J2-like) nonlinear FEM data generator ====
# Hex8, 2x2x2 Gauss, small strain.
# - Explicit NU (Poisson)
# - Symmetric Dirichlet enforcement
# - Bilinear algorithmic tangent that preserves bulk modulus (volumetric elastic)
# - Precomputed I/J triplets, only V changes per iteration (faster than nested loops)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer

# ---- CuPy-or-NumPy shim (drop-in) ----
import numpy as _np
try:
    import cupy as cp
    # force-load NVRTC to catch DLL issues upfront
    _ = cp.cuda.runtime.getDeviceCount()
    print("[CuPy] GPU available:", _)
except Exception as e:
    print("[CuPy] Not available -> falling back to NumPy (CPU):", e)
    class _CPShim:
        # delegate everything to NumPy
        def __getattr__(self, name): return getattr(_np, name)
        # provide CuPy-compatible helper
        @staticmethod
        def asnumpy(x): return _np.asarray(x)
        # dtype names often used
        float32 = _np.float32
        uint8   = _np.uint8
        ndarray = _np.ndarray
    cp = _CPShim()
# --------------------------------------

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# -----------------------
# User config
# -----------------------
N = 32
dx = 1.0 / (N - 1)
num_samples = 1
output_dir = "./data_nonlinear_twopeak"
os.makedirs(output_dir, exist_ok=True)

# Material (bilinear J2-like)
E      = 1.0        # elastic modulus
NU     = 0.30       # Poisson's ratio (explicit!)
SIGY   = 2.0e-3     # yield stress
H_iso  = 0.10       # isotropic hardening modulus (fraction of E is common)

# Nonlinear loop
MAX_ITER = 90
TOL      = 1e-40

# Derived sizes
NE   = (N - 1) ** 3
NDOF = (N ** 3) * 3
Hh   = 1.0 / (N - 1)   # element size
VOL  = Hh ** 3

# Force field settings (Gaussian blob(s) in x-direction by default)
force_sigma = 0.05
num_blobs = 2
force_mag_range = (0.1, 0.2)  # (min, max) density magnitude per blob

TARGET_LUMPED_FORCE_TOTAL = 0.002  # total ∑ f*dx^3 across all components

def node_id(i, j, k): return i * N * N + j * N + k

# -----------------------
# Loads & BC
# -----------------------
def generate_gaussian_force():
    # --- original Gaussian construction ---
    x, y, z = np.meshgrid(np.linspace(0, 1, N),
                          np.linspace(0, 1, N),
                          np.linspace(0, 1, N), indexing='ij')
    f = np.zeros((3, N, N, N), dtype=np.float32)
    s2 = 2.0 * (force_sigma ** 2)
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

def generate_bc_mask_and_value():
    """
    Clamp the z=0 face (k=0) to zero (all 3 components).
    NOTE: mask[:, :, :, 0] corresponds to z=0 plane with [3, X, Y, Z] storage.
    """
    mask = cp.zeros((3, N, N, N), dtype=cp.uint8)
    val  = cp.zeros((3, N, N, N), dtype=cp.float32)
    mask[:, :, :, 0] = 1
    val[:,  :, :, 0] = 0.0
    return mask, val

# -----------------------
# Material helpers
# -----------------------
def _C_iso(K, mu):
    """6x6 Voigt for isotropic small strain: C = 3K*Ivol + 2mu*Idev."""
    Ivol = np.array([[1,1,1,0,0,0],
                     [1,1,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]], dtype=float) / 3.0
    Idev = np.eye(6) - Ivol
    return 3.0*K*Ivol + 2.0*mu*Idev

def _C_elastic_and_params(E, nu):
    K  = E / (3.0 * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return _C_iso(K, mu), K, mu

def _C_plastic_like(K, mu, H):
    """
    Cheap J2-like algorithmic tangent:
      volumetric stays elastic (K),
      deviatoric softened with mu_alg = (mu*H)/(3mu + H).
    """
    mu_alg = (mu * H) / (3.0 * mu + H)
    return _C_iso(K, mu_alg)

def _von_mises_voigt(sig6):
    """sig6: (m,6) -> von Mises equivalent stress."""
    sx, sy, sz, txy, tyz, tzx = sig6.T
    return np.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2) + 3.0*(txy**2 + tyz**2 + tzx**2))

# -----------------------
# HEX8 machinery
# -----------------------
def _dN_dxi_hex8(xi, eta, zeta):
    c = np.array([[-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
                  [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]], dtype=float)
    dN = np.zeros((3, 8), dtype=float)
    for a in range(8):
        cx, cy, cz = c[a]
        dN[0, a] = 0.125 * cx * (1 + cy*eta) * (1 + cz*zeta)
        dN[1, a] = 0.125 * cy * (1 + cx*xi)  * (1 + cz*zeta)
        dN[2, a] = 0.125 * cz * (1 + cx*xi)  * (1 + cy*eta)
    return dN

def _B_from_dNdx(dNdx):
    B = np.zeros((6, 24), dtype=float)
    for a in range(8):
        i = 3*a
        dNx, dNy, dNz = dNdx[:, a]
        B[:, i:i+3] = np.array([[dNx, 0,   0  ],
                                [0,   dNy, 0  ],
                                [0,   0,   dNz],
                                [dNy, dNx, 0  ],
                                [0,   dNz, dNy],
                                [dNz, 0,   dNx]], dtype=float)
    return B

def _Ke_hex8(dx, C):
    """2x2x2 Gauss with physical mapping for a cube element of size dx."""
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    Ke = np.zeros((24,24), dtype=float)
    Jinv = np.diag([2.0/dx, 2.0/dx, 2.0/dx])
    detJ = (dx/2.0)**3
    for xi in gp:
        for et in gp:
            for ze in gp:
                dN_dxi = _dN_dxi_hex8(xi, et, ze)
                dN_dx  = Jinv @ dN_dxi
                B = _B_from_dNdx(dN_dx)
                Ke += (B.T @ C @ B) * detJ
    return Ke

# -----------------------
# Assembler (precompute I/J)
# -----------------------
class NonlinearAssembler:
    def __init__(self, N, dx, E, nu, H, sigy):
        self.N   = N
        self.dx  = dx
        self.sigy = sigy

        # material
        self.Ce, self.K_bulk, self.mu = _C_elastic_and_params(E, nu)
        self.Cp = _C_plastic_like(self.K_bulk, self.mu, H)

        # element matrices
        self.Ke_e = _Ke_hex8(dx, self.Ce)
        self.Ke_p = _Ke_hex8(dx, self.Cp)

        # connectivity
        self.elem_dofs = self._make_elem_dofs()

        # Precompute I/J triplets once
        self.I, self.J = self._build_IJ()

        # B at element center for stress prediction
        self.Bc = self._center_B()

    def _make_elem_dofs(self):
        edofs = np.zeros((NE, 24), dtype=np.int32)
        e = 0
        for i in range(self.N-1):
            for j in range(self.N-1):
                for k in range(self.N-1):
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
                        dofs += [3*nid+0, 3*nid+1, 3*nid+2]
                    edofs[e] = np.array(dofs, dtype=np.int32)
                    e += 1
        return edofs

    def _build_IJ(self):
        """Global row/col arrays for all element contributions (constant)."""
        I = np.empty(NE * 576, dtype=np.int32)
        J = np.empty(NE * 576, dtype=np.int32)
        off = 0
        for dofs in self.elem_dofs:
            rows = np.repeat(dofs, 24)
            cols = np.tile(dofs, 24)
            I[off:off+576] = rows
            J[off:off+576] = cols
            off += 576
        return I, J

    def _center_B(self):
        """Strain-displacement matrix at element center (-0- point)."""
        dN_dxi = np.zeros((3,8), dtype=float)
        # at center, derivatives simplify: each is 0.125 * corner sign
        corners = np.array([[-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
                            [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]], dtype=float)
        for a, (cx,cy,cz) in enumerate(corners):
            dN_dxi[0,a] = 0.125 * cx
            dN_dxi[1,a] = 0.125 * cy
            dN_dxi[2,a] = 0.125 * cz
        Jinv = np.diag([2.0/self.dx, 2.0/self.dx, 2.0/self.dx])
        dN_dx = Jinv @ dN_dxi
        return _B_from_dNdx(dN_dx)

    # ---- nonlinear pieces ----
    def _trial_stress_vm(self, u_flat):
        """von Mises at element centers using elastic predictor."""
        ue = u_flat[self.elem_dofs]             # (NE, 24)
        eps = (self.Bc @ ue.T).T                # (NE, 6)
        sig = (self.Ce @ eps.T).T               # (NE, 6)
        return _von_mises_voigt(sig)

    def assemble(self, f_cpu, plastic_flags=None):
        """
        Build global K (CSC) and F (lumped) for the current plastic flags.
        plastic_flags: bool array (NE,) -> choose Ke_p vs Ke_e per element.
        """
        if plastic_flags is None:
            plastic_flags = np.zeros(NE, dtype=bool)

        # Fill values V per element by picking Ke
        V = np.empty(NE * 576, dtype=np.float64)
        off = 0
        Ke_e_flat = self.Ke_e.ravel()
        Ke_p_flat = self.Ke_p.ravel()
        for e in range(NE):
            if plastic_flags[e]:
                V[off:off+576] = Ke_p_flat
            else:
                V[off:off+576] = Ke_e_flat
            off += 576

        K = coo_matrix((V, (self.I, self.J)), shape=(NDOF, NDOF)).tocsc()

        # Lumped body force: F = ∫ f dV ≈ f(node) * vol
        F = np.zeros(NDOF, dtype=np.float64)
        F[0::3] = f_cpu[0].ravel()
        F[1::3] = f_cpu[1].ravel()
        F[2::3] = f_cpu[2].ravel()
        F *= VOL
        return K, F

# Global assembler
ASM = NonlinearAssembler(N, dx, E, NU, H_iso, SIGY)

# -----------------------
# Symmetric Dirichlet
# -----------------------
def apply_dirichlet(K, F, bc_mask_cpu, bc_val_cpu):
    """
    Symmetric Dirichlet: zero columns & rows for constrained DOFs, set diag=1, RHS=bc.
    """
    K = K.tolil(copy=True)

    cons = []
    vals = []
    for c in range(3):
        i, j, k = np.where(bc_mask_cpu[c] > 0.5)
        dofs = 3 * node_id(i, j, k) + c
        cons.append(dofs)
        vals.append(bc_val_cpu[c, i, j, k])
    cons = np.concatenate(cons).astype(np.int32)
    vals = np.concatenate(vals).astype(np.float64)

    # zero columns then rows
    K[:, cons] = 0.0
    K[cons, :] = 0.0
    for d in cons:
        K[d, d] = 1.0
    F[cons] = vals
    return K.tocsc(), F

# -----------------------
# Nonlinear solve (bilinear)
# -----------------------
def solve_bilinear(f_cu, bc_mask_cu, bc_val_cu, max_iter=MAX_ITER, tol=TOL):
    f  = cp.asnumpy(f_cu)
    m  = cp.asnumpy(bc_mask_cu)
    bv = cp.asnumpy(bc_val_cu)

    # Initial (fully elastic)
    K, F = ASM.assemble(f, plastic_flags=None)
    Kb, Fb = apply_dirichlet(K, F, m, bv)
    u = spsolve(Kb, Fb)

    for it in range(max_iter):
        u_old = u.copy()

        # classify elements by trial vm stress at centers
        t1 = default_timer()
        vm = ASM._trial_stress_vm(u)
        plast = vm >= SIGY
        pct = 100.0 * plast.sum() / NE
        print(f"  Iter {it+1}: plastic elements = {plast.sum():,}/{NE:,} ({pct:.2f}%)")

        # rebuild with bilinear tangent (elementwise)
        K, F = ASM.assemble(f, plastic_flags=plast)
        Kb, Fb = apply_dirichlet(K, F, m, bv)
        u = spsolve(Kb, Fb)

        t2 = default_timer()
        iter_time = t2 - t1

        rel = np.linalg.norm(u - u_old) / (np.linalg.norm(u) + 1e-16)
        print(f"  Iter {it+1}: rel update = {rel:.2e} | Time: {iter_time: .2f} s")
        if rel < tol:
            print(f"Converged in {it+1} iterations")
            break
    else:
        print(f"Reached MAX_ITER={max_iter}")

    # reshape to [3, X, Y, Z]
    u3 = u.reshape(N, N, N, 3).transpose(3, 0, 1, 2).astype(np.float32)
    return u3

# -----------------------
# Viz (optional)
# -----------------------
def visualize_force_displacement(f, u):
    sl = N // 2
    fz = cp.asnumpy(f[2, :, :, sl])
    uz = u[2, :, :, sl]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(fz, cmap='hot', origin='lower'); axs[0].set_title("Force (z-dir) at Z mid")
    im1 = axs[1].imshow(uz, cmap='viridis', origin='lower'); axs[1].set_title("Displacement (z-dir) at Z mid")
    plt.colorbar(im0, ax=axs[0]); plt.colorbar(im1, ax=axs[1])
    for ax in axs: ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.tight_layout(); plt.show()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print(f"Nonlinear FEM | N={N}, dx={dx:.6f}, E={E}, nu={NU}, sigy={SIGY}, H={H_iso}")
    print(f"DOFs={NDOF:,}, elements={NE:,}")
    for i in tqdm(range(num_samples), desc="Generating (bilinear nonlinear)"):
        f = generate_gaussian_force()
        bc_mask, bc_val = generate_bc_mask_and_value()

        u = solve_bilinear(f, bc_mask, bc_val)

        sample = {
            "f": torch.tensor(cp.asnumpy(f)),
            "u": torch.tensor(u),
            "bc_mask": torch.tensor(cp.asnumpy(bc_mask)),
            "bc_val": torch.tensor(cp.asnumpy(bc_val)),
        }
        fname = os.path.join(output_dir, f"sample1_{i:04d}_N{N}_bilinear_fem.pt")
        torch.save(sample, fname)

        if i == 0:
            visualize_force_displacement(f, u)

        # residual check with final tangent (classify once more)
        u_flat = u.transpose(1, 2, 3, 0).reshape(-1)
        plast = ASM._trial_stress_vm(u_flat) >= SIGY
        Kc, Fc = ASM.assemble(cp.asnumpy(f), plastic_flags=plast)
        Kc, Fc = apply_dirichlet(Kc, Fc, cp.asnumpy(bc_mask), cp.asnumpy(bc_val))
        res = Kc @ u_flat - Fc
        rel_res = np.linalg.norm(res) / (np.linalg.norm(u_flat) + 1e-16)
        print(f"[GEN CHECK] sample {i:04d}  rel_residual = {rel_res:.3e}")
    print(f"Saved {num_samples} samples to: {output_dir}")
# ==== END DROP-IN ====
