
# Revised training and evaluation script for 3D Elasticity
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch. utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def calculate_von_mises_strain(u, dx=1.0):
    """
    Calculate von Mises equivalent strain from displacement field
    Args:
        u: displacement field [3, D, H, W]
        dx: grid spacing
    Returns:
        von_mises: scalar von Mises strain field [D, H, W]
    """
    # Convert to numpy if tensor
    if torch.is_tensor(u):
        u = u.detach().cpu().numpy()
    
    # Calculate strain tensor components
    strain = np.zeros((3, 3) + u.shape[1:])
    
    # Spatial gradients for each component
    for i in range(3):
        for j in range(3):
            # Central difference for strain components
            if i == 0:  # x-derivatives
                strain[i,j] = 0.5 * (np.gradient(u[j], axis=2, edge_order=2) / dx)
            elif i == 1:  # y-derivatives
                strain[i,j] = 0.5 * (np.gradient(u[j], axis=1, edge_order=2) / dx)
            elif i == 2:  # z-derivatives
                strain[i,j] = 0.5 * (np.gradient(u[j], axis=0, edge_order=2) / dx)
            
            # Symmetric part (epsilon_ij = 0.5*(u_i,j + u_j,i))
            if i != j:
                if j == 0:  # x-derivatives
                    strain[i,j] += 0.5 * (np.gradient(u[i], axis=2, edge_order=2) / dx)
                elif j == 1:  # y-derivatives
                    strain[i,j] += 0.5 * (np.gradient(u[i], axis=1, edge_order=2) / dx)
                elif j == 2:  # z-derivatives
                    strain[i,j] += 0.5 * (np.gradient(u[i], axis=0, edge_order=2) / dx)
    
    # Calculate von Mises equivalent strain
    eps_xx = strain[0,0]
    eps_yy = strain[1,1]
    eps_zz = strain[2,2]
    eps_xy = strain[0,1]
    eps_yz = strain[1,2]
    eps_zx = strain[2,0]
    
    von_mises = np.sqrt(0.5 * ((eps_xx - eps_yy)**2 + 
                               (eps_yy - eps_zz)**2 + 
                               (eps_zz - eps_xx)**2 + 
                               6 * (eps_xy**2 + eps_yz**2 + eps_zx**2)))
    
    return von_mises

def compute_stress_divergence(u, E=1.0, nu=0.3, dx=1.0):
    if torch.is_tensor(u):
        u = u.detach().cpu().numpy()

    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    grad_u = np.zeros((3, 3) + u.shape[1:])  # shape: (3, 3, D, H, W)

    for i in range(3):  # u_i
        for j in range(3):  # ∂/∂x_j
            grad_u[i, j] = np.gradient(u[i], dx, axis=j, edge_order=2)

    strain = 0.5 * (grad_u + grad_u.transpose(1, 0, 2, 3, 4))  # symmetric ε_ij

    # Compute stress tensor σ_ij
    trace = strain[0, 0] + strain[1, 1] + strain[2, 2]
    stress = np.zeros_like(strain)
    for i in range(3):
        for j in range(3):
            stress[i, j] = lam * trace * (i == j) + 2 * mu * strain[i, j]

    # Divergence of stress → ∂σ_ij/∂x_j
    div_sigma = np.zeros_like(u)
    for i in range(3):  # output component
        for j in range(3):
            div_sigma[i] += np.gradient(stress[i, j], dx, axis=j, edge_order=2)

    return div_sigma

def plot_stress_divergence(u_pred, f_true, dx=1.0, E=1.0, nu=0.3, slice_dim=2, slice_idx=None):
    """
    Plot residual ∇·σ(u) - f for the predicted displacement.
    Args:
        u_pred: predicted displacement, shape [3, D, H, W]
        f_true: ground-truth force field, same shape
        dx: grid spacing
    """
    if u_pred.ndim == 5: u_pred = u_pred[0]
    if f_true.ndim == 5: f_true = f_true[0]
    if torch.is_tensor(f_true):
        f_true = f_true.detach().cpu().numpy()
    
    div_sigma = compute_stress_divergence(u_pred, E=E, nu=nu, dx=dx)
    residual = div_sigma - f_true
    residual_norm = np.linalg.norm(residual, axis=0)  # shape: [D, H, W]


    # Choose slice
    if slice_idx is None:
        slice_idx = residual_norm.shape[slice_dim] // 2

    if slice_dim == 0:
        slice = residual_norm[slice_idx]
        label = f"Z={slice_idx}"
    elif slice_dim == 1:
        slice = residual_norm[:, slice_idx]
        label = f"Y={slice_idx}"
    else:
        slice = residual_norm[:, :, slice_idx]
        label = f"X={slice_idx}"

    # Plot
    plt.figure(figsize=(7, 6))
    plt.title(f"‖∇·σ(u) − f‖ Residual (Slice {label})")
    im = plt.imshow(slice, cmap='magma')
    plt.colorbar(im, label='Residual magnitude')
    plt.axis('off')
    plt.tight_layout()
    plt.show()



import matplotlib.gridspec as gridspec


import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- helper: normalize any strain layout to Voigt-6 (exx, eyy, ezz, gxy, gyz, gzx) ----
def _voigt6_from_any(eps):
    if not isinstance(eps, torch.Tensor):
        eps = torch.as_tensor(eps)
    if eps.ndim == 5 and eps.shape[1] == 6:
        return eps
    if eps.ndim == 5 and eps.shape[1] == 3:  # only normals
        exx, eyy, ezz = eps[:,0], eps[:,1], eps[:,2]
        z = torch.zeros_like(exx)
        return torch.stack([exx, eyy, ezz, z, z, z], dim=1)
    if eps.ndim == 5 and eps.shape[1] == 9:
        xx, xy, xz, yx, yy, yz, zx, zy, zz = [eps[:,i] for i in range(9)]
        exx, eyy, ezz = xx, yy, zz
        exy = 0.5*(xy + yx); eyz = 0.5*(yz + zy); ezx = 0.5*(zx + xz)
        gxy, gyz, gzx = 2*exy, 2*eyz, 2*ezx
        return torch.stack([exx, eyy, ezz, gxy, gyz, gzx], dim=1)
    if eps.ndim == 6 and eps.shape[1] == 3 and eps.shape[2] == 3:
        exx = eps[:,0,0]; eyy = eps[:,1,1]; ezz = eps[:,2,2]
        exy = 0.5*(eps[:,0,1] + eps[:,1,0])
        eyz = 0.5*(eps[:,1,2] + eps[:,2,1])
        ezx = 0.5*(eps[:,2,0] + eps[:,0,2])
        gxy, gyz, gzx = 2*exy, 2*eyz, 2*ezx
        return torch.stack([exx, eyy, ezz, gxy, gyz, gzx], dim=1)
    raise ValueError(f"Unsupported strain shape {tuple(eps.shape)}")

# ---- HL-CNS VM-from-displacement (PHYSICAL units) ----
def _vm_from_u_physical(u_phys, dx, compute_strain_tensor):
    if not isinstance(u_phys, torch.Tensor):
        u = torch.from_numpy(np.asarray(u_phys))
    else:
        u = u_phys
    if u.ndim == 4:   # [3,D,H,W] -> [1,3,D,H,W]
        u = u.unsqueeze(0)

    eps = compute_strain_tensor(u, dx)     # your function
    e6  = _voigt6_from_any(eps)            # [B,6,D,H,W]
    exx, eyy, ezz, gxy, gyz, gzx = [e6[:,i] for i in range(6)]
    # engineering shear γ -> tensor shear ε = γ/2
    exy, eyz, ezx = 0.5*gxy, 0.5*gyz, 0.5*gzx
    mean = (exx + eyy + ezz)/3.0
    exx_d, eyy_d, ezz_d = exx-mean, eyy-mean, ezz-mean
    e2 = (exx_d**2 + eyy_d**2 + ezz_d**2) + 2.0*(exy**2 + eyz**2 + ezx**2)
    vm = torch.sqrt((2.0/3.0) * e2 + 1e-32)[0]   # [D,H,W]
    return vm.cpu().numpy()

def plot_comparison_3d(u_true_phys, u_pred_phys, *, dx,
                       compute_strain_tensor,
                       yield_strain=None,  # e.g., 0.002; if given we fix vmax to this for comparability
                       save_path=None, slice_dim=2, slice_idx=None, font_size=14):

    # ensure [3,D,H,W] for both
    if isinstance(u_true_phys, torch.Tensor) and u_true_phys.ndim == 5:
        u_true_phys = u_true_phys[0]
    if isinstance(u_pred_phys, torch.Tensor) and u_pred_phys.ndim == 5:
        u_pred_phys = u_pred_phys[0]

    # displacement magnitude (physical)
    disp_true = torch.linalg.norm(u_true_phys, dim=0).detach().cpu().numpy()
    disp_pred = torch.linalg.norm(u_pred_phys, dim=0).detach().cpu().numpy()

    # von Mises strain (physical) using the SAME routine used by your 3D plot
    vm_true = _vm_from_u_physical(u_true_phys, dx, compute_strain_tensor)  # [D,H,W]
    vm_pred = _vm_from_u_physical(u_pred_phys, dx, compute_strain_tensor)

    # choose slice
    if slice_idx is None:
        size = vm_true.shape[slice_dim]
        slice_idx = size // 2
    axis_labels = ['X', 'Y', 'Z']
    slice_label = f"{axis_labels[slice_dim]}={slice_idx}"

    if slice_dim == 0:
        d_true, d_pred = disp_true[slice_idx], disp_pred[slice_idx]
        v_true, v_pred = vm_true[slice_idx], vm_pred[slice_idx]
    elif slice_dim == 1:
        d_true, d_pred = disp_true[:, slice_idx], disp_pred[:, slice_idx]
        v_true, v_pred = vm_true[:, slice_idx], vm_pred[:, slice_idx]
    else:
        d_true, d_pred = disp_true[:, :, slice_idx], disp_pred[:, :, slice_idx]
        v_true, v_pred = vm_true[:, :, slice_idx], vm_pred[:, :, slice_idx]

    # consistent ranges
    vmin_disp = min(float(d_true.min()), float(d_pred.min()))
    vmax_disp = max(float(d_true.max()), float(d_pred.max()))
    if yield_strain is not None:
        vmin_strain, vmax_strain = 0.0, float(yield_strain)   # compare to yield consistently
    else:
        vmin_strain = min(float(v_true.min()), float(v_pred.min()))
        vmax_strain = max(float(v_true.max()), float(v_pred.max()))

    # plot
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Displacement & Strain Comparison ({slice_label})", fontsize=font_size + 2)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.05)

    def plot_field(ax, data, title, cmap, vmin, vmax):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        return im

    # row 1: displacement
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); cax0 = fig.add_subplot(gs[0, 2])
    im0 = plot_field(ax0, d_true, "True Displacement (m)", 'viridis', vmin_disp, vmax_disp)
    im1 = plot_field(ax1, d_pred, "Predicted Displacement (m)", 'viridis', vmin_disp, vmax_disp)
    cbar0 = plt.colorbar(im1, cax=cax0); cbar0.ax.tick_params(labelsize=font_size - 2)

    # row 2: von Mises strain
    ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[1, 1]); cax1 = fig.add_subplot(gs[1, 2])
    im2 = plot_field(ax2, v_true, "True Von Mises Strain", 'plasma', vmin_strain, vmax_strain)
    im3 = plot_field(ax3, v_pred, "Predicted Von Mises Strain", 'plasma', vmin_strain, vmax_strain)
    cbar1 = plt.colorbar(im3, cax=cax1); cbar1.ax.tick_params(labelsize=font_size - 2)
    if yield_strain is not None:
        cbar1.set_label(f"ε_vm (yield={yield_strain:g})")

    fig.subplots_adjust(left=0.06, right=0.95, top=0.93, bottom=0.07, wspace=0.1, hspace=0.15)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()





def compute_strain_tensor(u_batch, dx):
    B, C, D, H, W = u_batch.shape
    grads = [
        torch.gradient(u_batch[:, i], spacing=(dx, dx, dx), dim=(1, 2, 3), edge_order=2)
        for i in range(3)
    ]
    eps = torch.zeros(B, 3, 3, D, H, W, device=u_batch.device)
    for i in range(3):
        for j in range(3):
            eps[:, i, j] = 0.5 * (grads[i][j] + grads[j][i])
    return eps
