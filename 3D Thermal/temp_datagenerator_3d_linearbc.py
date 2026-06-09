import os
import numpy as np
import torch
from scipy.sparse import diags, lil_matrix, block_diag
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# --- Config ---
N = 32
dx = 1.0 / (N - 1)
batch_size = 10   # Changed to 1 as in original
alpha = 2.0
beta = 2.0
output_dir = "./"
BC = {'top': 1.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0, 'front': 0.0, 'back': 0.0}

# Add missing idx function
def idx(i, j, k):
    return i * N * N + j * N + k

force_sigma = 0.05
force_mag_range = (1000.0, 5000.0)
NUM_BLOBS_RANGE = (1, 1)
TARGET_LUMPED_FORCE_TOTAL = 1.0
def generate_force(num_blobs, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        rand_uniform = rng.uniform
        rand_int = rng.randint
    else:
        rand_uniform = np.random.uniform
        rand_int = np.random.randint

    x, y, z = np.meshgrid(
        np.linspace(0, 1, N),
        np.linspace(0, 1, N),
        np.linspace(0, 1, N),
        indexing='ij'
    )
    f = np.zeros((N, N, N), dtype=np.float32)
    s2 = 2.0 * (force_sigma ** 2)

    if isinstance(num_blobs, (tuple, list)):
        nb = rand_int(num_blobs[0], num_blobs[1] + 1)
    else:
        nb = int(num_blobs)

    for _ in range(nb):
        cx, cy, cz = rand_uniform(0.1, 0.9, 3)
        mag = rand_uniform(*force_mag_range)
        g = mag * np.exp(-((x - cx)**2 + (y - cy)**2 + (z - cz)**2) / s2)
        f += g.astype(np.float32)
    return f

def apply_bc_3d(A, F):
    z_values = np.linspace(1.0, 0.0, N)
    
    A = A.tolil()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                index = idx(i, j, k)
                if i == 0:      # Top boundary
                    A[index, :] = 0
                    A[index, index] = 1
                    F[index] = 1.0
                elif i == N-1:  # Bottom boundary
                    A[index, :] = 0
                    A[index, index] = 1
                    F[index] = 0.0
                elif j == 0:    # Left boundary
                    A[index, :] = 0
                    A[index, index] = 1
                    F[index] = z_values[i]
                elif j == N-1:  # Right boundary
                    A[index, :] = 0
                    A[index, index] = 1
                    F[index] = z_values[i]
                elif k == 0:    # Front boundary
                    A[index, :] = 0
                    A[index, index] = 1
                    F[index] = z_values[i]
                elif k == N-1:  # Back boundary
                    A[index, :] = 0
                    A[index, index] = 1
                    F[index] = z_values[i]
    return A.tocsr(), F

def construct_matrix_3d(kappa):
    """Original single-matrix construction"""
    size = N * N * N
    diag = np.zeros(size)
    off1 = np.zeros(size - 1)
    offN = np.zeros(size - N)
    offNN = np.zeros(size - N * N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                index = idx(i, j, k)
                if i == 0 or i == N-1 or j == 0 or j == N-1 or k == 0 or k == N-1:
                    diag[index] = 1
                else:
                    kc = kappa[i, j, k]
                    kx1 = kappa[i, j, k-1]
                    kx2 = kappa[i, j, k+1]
                    ky1 = kappa[i, j-1, k]
                    ky2 = kappa[i, j+1, k]
                    kz1 = kappa[i-1, j, k]
                    kz2 = kappa[i+1, j, k]
                    wx1 = (kc + kx1) / (2 * dx**2)
                    wx2 = (kc + kx2) / (2 * dx**2)
                    wy1 = (kc + ky1) / (2 * dx**2)
                    wy2 = (kc + ky2) / (2 * dx**2)
                    wz1 = (kc + kz1) / (2 * dx**2)
                    wz2 = (kc + kz2) / (2 * dx**2)
                    diag[index] = -(wx1 + wx2 + wy1 + wy2 + wz1 + wz2)
                    off1[index - 1] = wx1
                    off1[index] = wx2
                    offN[index - N] = wy1
                    offN[index] = wy2
                    offNN[index - N*N] = wz1
                    offNN[index] = wz2
    return diags([diag, off1, off1, offN, offN, offNN, offNN], [0, -1, 1, -N, N, -N*N, N*N]).tolil()

def solve_nonlinear_poisson(f, beta=2.0, mode="linear", nit=60, tol=1e-5, verbose=False):
    f_flat = f.flatten()

    if mode == "linear" or beta == 0.0:
        diag = -6.0 * np.ones(N**3)
        off1 = np.ones(N**3 - 1)
        off1[np.arange(1, N**3) % N == 0] = 0
        offN = np.ones(N**3 - N)
        offNN = np.ones(N**3 - N * N)
        A = diags([diag, off1, off1, offN, offN, offNN, offNN], [0, -1, 1, -N, N, -N*N, N*N]) / dx**2
        A, f_flat_bc = apply_bc_3d(A.copy(), f_flat.copy())
        u = spsolve(A, f_flat_bc).reshape(N, N, N)
        return u

    A0 = diags([-6.0] * N**3, 0) / dx**2
    A0, f_bc = apply_bc_3d(A0.copy(), f_flat.copy())
    u = spsolve(A0, f_bc).reshape(N, N, N)

    for n in range(nit):
        if mode == "decreasing":
            kappa = 1.0 / (1.0 + beta * np.maximum(u, 0.0) ** alpha)
        else:
            kappa = 1.0 + beta * np.maximum(u, 0.0) ** alpha

        A = construct_matrix_3d(kappa)
        A, F = apply_bc_3d(A, f_flat.copy())

        u_new = spsolve(A, F).reshape(N, N, N)
        err = np.max(np.abs(u_new - u))
        if verbose:
            print(f"iter={n+1}, err={err:.3e}")

        u = u_new
        if err < tol:
            break
    return u

def generate_and_save(mode, num_blobs, beta, filename):
    """Modified with progress bar but keeping original single-sample logic"""
    f_list, u_list = [], []
    for _ in tqdm(range(num_samples), desc=f"Generating {mode} samples"):
        f = generate_force(num_blobs)
        u = solve_nonlinear_poisson(f, beta=beta, mode=mode)
        # u1 = solve_nonlinear_poisson(f, beta=0, mode='linear')
        # u2 = solve_nonlinear_poisson(f, beta=beta, mode='decreasing')
        # i = 18
        # plt.contourf(f[i])
        # plt.show()
        # plt.contourf(u1[i])
        # plt.colorbar()
        # plt.show()
        # plt.contourf(u[i])
        # plt.colorbar()
        # plt.show()
        # plt.contourf(u2[i])
        # plt.colorbar()
        # plt.show()
        f_list.append(torch.tensor(f).unsqueeze(0).float())
        u_list.append(torch.tensor(u).unsqueeze(0).float())
    data = {'f': torch.stack(f_list), 'u': torch.stack(u_list)}
    torch.save(data, os.path.join(output_dir, filename))
    print(f"Saved: {filename}")

num_samples = 200
# generate_and_save("linear", 1, beta=0.0, filename=f"data/validation_data_N{N}_linear.pt")
# generate_and_save("increasing", 1, beta=2, filename=f"data/validation_data_N{N}_increase.pt")
# generate_and_save("decreasing", 1, beta=5.0, filename=f"data/validation_data_N{N}_decrease.pt")

num_samples = 500
# generate_and_save("linear", 1, beta=0.0, filename=f"data/training_data_N{N}_linear.pt")
generate_and_save("increasing", 1, beta=2, filename=f"data/training_data_N{N}_increase.pt")
# generate_and_save("decreasing", 1, beta=2, filename=f"data/training_data_N{N}_decrease.pt")

num_samples = 200
# generate_and_save("linear", 2, beta=0.0, filename=f"data/dualforce_data_N{N}_linear.pt")
# generate_and_save("increasing", 2, beta=5.0, filename=f"data/dualforce_data_N{N}_increase.pt")
# generate_and_save("decreasing", 2, beta=5.0, filename=f"data/dualforce_data_N{N}_decrease.pt")

