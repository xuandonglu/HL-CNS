import os
import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import random
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

force_mag_range = (1000.0, 5000.0)
force_sigma = 0.05
def generate_force(num_blobs):
    x, y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing='ij')
    f = np.zeros((N, N), dtype=np.float32)
    s2 = 2.0 * (force_sigma ** 2)
    for _ in range(num_blobs):
        cx, cy = np.random.uniform(0.1, 0.9, 2)
        mag = np.random.uniform(*force_mag_range)
        g = mag * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s2)
        f += g.astype(np.float32)
    return f


def solve_nonlinear_poisson(f, beta=2.0, mode="linear", nit=200):
    def apply_bc(A, F):
        A = A.tolil()
        vertical_bc = np.linspace(1.0, 0.0, N)
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                if i == 0:
                    A[idx, :] = 0; A[idx, idx] = 1; F[idx] = BC['top']
                elif i == N - 1:
                    A[idx, :] = 0; A[idx, idx] = 1; F[idx] = BC['bottom']
                elif j == 0:
                    A[idx, :] = 0; A[idx, idx] = 1; F[idx] = vertical_bc[i]  # left edge: top=1 to bottom=0
                elif j == N - 1:
                    A[idx, :] = 0; A[idx, idx] = 1; F[idx] = vertical_bc[i]  # right edge: top=1 to bottom=0
        return A.tocsr(), F


    def construct_matrix(kappa):
        diag = np.zeros(N * N)
        off1 = np.zeros(N * N - 1)
        offN = np.zeros(N * N - N)
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                if i == 0 or i == N-1 or j == 0 or j == N-1:
                    diag[idx] = 1
                else:
                    kc = kappa[i, j]
                    kl = kappa[i, j - 1]
                    kr = kappa[i, j + 1]
                    ku = kappa[i - 1, j]
                    kd = kappa[i + 1, j]
                    wl = (kc + kl) / (2 * dx * dx)
                    wr = (kc + kr) / (2 * dx * dx)
                    wu = (kc + ku) / (2 * dx * dx)
                    wd = (kc + kd) / (2 * dx * dx)
                    off1[idx - 1] = wl
                    off1[idx] = wr
                    offN[idx - N] = wu
                    offN[idx] = wd
                    diag[idx] = -(wl + wr + wu + wd)
        return diags([diag, off1, off1, offN, offN], [0, -1, 1, -N, N]).tolil()
    
    # ✅ Handle linear case separately (no iteration)
    if mode == "linear" or beta == 0.0:
        diag = -4.0 * np.ones(N * N)
        off1 = np.ones(N * N - 1)
        off1[np.arange(1, N * N) % N == 0] = 0
        offN = np.ones(N * N - N)
        A = diags([diag, off1, off1, offN, offN], [0, -1, 1, -N, N]) / dx**2
        F_flat = f.flatten()
        A, F_flat = apply_bc(A.copy(), F_flat.copy())
        u = spsolve(A, F_flat).reshape(N, N)
        return u    

    # Initial linear solve
    A0 = diags([-4.0] * N * N, 0) / dx**2
    f_flat = f.flatten()
    A0, f_bc = apply_bc(A0.copy(), f_flat.copy())
    u = spsolve(A0, f_bc).reshape(N, N)

    for n in range(nit):
        if mode == "decreasing":
            kappa = 1.0 / (1.0 + beta * np.maximum(u, 0.0) ** alpha)
        else:
            kappa = 1.0 + beta * np.maximum(u, 0.0) ** alpha
        A = construct_matrix(kappa)
        F = f.flatten()
        A, F = apply_bc(A, F)
        u = spsolve(A, F).reshape(N, N)

    return u

def generate_and_save(mode, num_blobs, beta, filename):
    f_list, u_list = [], []
    for _ in range(num_samples):
        f = generate_force(num_blobs)
        u = solve_nonlinear_poisson(f, beta=beta, mode=mode)
        # u1 = solve_nonlinear_poisson(f, beta=0, mode='linear')
        # u2 = solve_nonlinear_poisson(f, beta=beta, mode='decreasing')
        # plt.contourf(f)
        # plt.show()
        # plt.contourf(u1)
        # plt.colorbar()
        # plt.show()
        # plt.contourf(u)
        # plt.colorbar()
        # plt.show()
        # plt.contourf(u2)
        # plt.colorbar()
        # plt.show()
        f_list.append(torch.tensor(f).unsqueeze(0).float())
        u_list.append(torch.tensor(u).unsqueeze(0).float())
    data = {'f': torch.stack(f_list), 'u': torch.stack(u_list)}
    torch.save(data, os.path.join(output_dir, filename))
    print(f"Saved: {filename}")

# Grid & Parameters
N = 32
dx = 1.0 / (N - 1)
alpha = 2.0
output_dir = "./"  # Change if needed
BC = {'top': 1.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0}

num_samples = 200
# generate_and_save("linear", 1, beta=0.0, filename=f"data/validation_data_N{N}_linear.pt")
# generate_and_save("increasing", 1, beta=80, filename=f"data/validation_data_N{N}_increase.pt")
# generate_and_save("decreasing", 1, beta=2, filename=f"data/validation_data_N{N}_decrease.pt")
#
num_samples = 200
# generate_and_save("linear", 2, beta=0.0, filename=f"data/dualforce_data_N{N}_linear.pt")
generate_and_save("increasing", 2, beta=80, filename=f"data/dualforce_data_N{N}_increase.pt")
# generate_and_save("decreasing", 2, beta=80, filename=f"data/dualforce_data_N{N}_decrease.pt")

num_samples = 1000
# generate_and_save("linear", 1, beta=0.0, filename=f"data/training_data_N{N}_linear.pt")
# generate_and_save("increasing", 1, beta=80, filename=f"data/training_data_N{N}_increase.pt")
# generate_and_save("decreasing", 1, beta=80, filename=f"data/training_data_N{N}_decrease.pt")

