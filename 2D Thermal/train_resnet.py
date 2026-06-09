# train_plot_resnet_2d_thermal.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv

# ---------------- Utils ----------------
def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def relative_l2_masked(pred, target, mask, eps=1e-12):
    m = (mask < 0.5)  # True=interior
    num = torch.linalg.vector_norm((pred - target)[m])
    den = torch.linalg.vector_norm(target[m]) + eps
    return (num / den).item()

def _build_deriv_kernels_2d(dtype, device):
    k = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device)
    Kx = k.view(1, 1, 1, 3)
    Ky = k.view(1, 1, 3, 1)
    return Kx, Ky

def grad_2d_scalar(T: torch.Tensor, dx: float):
    Kx, Ky = _build_deriv_kernels_2d(T.dtype, T.device)
    Tx = F.conv2d(F.pad(T, (1, 1, 0, 0), mode='replicate'), Kx) / dx
    Ty = F.conv2d(F.pad(T, (0, 0, 1, 1), mode='replicate'), Ky) / dx
    return Tx, Ty

def create_bc_mask_and_value_like_2d(T: torch.Tensor):
    B, C, H, W = T.shape
    assert C == 1
    bc_mask = torch.zeros_like(T)
    bc_val  = torch.zeros_like(T)

    # top / bottom
    bc_mask[..., 0, :]  = 1.0; bc_val[..., 0, :]  = 1.0
    bc_mask[..., -1, :] = 1.0; bc_val[..., -1, :] = 0.0

    # left / right (vertical gradient)
    v = torch.linspace(1.0, 0.0, H, device=T.device).view(1, 1, H)  # [1,1,H]
    bc_mask[..., :, 0]  = 1.0; bc_val[..., :, 0]  = v
    bc_mask[..., :, -1] = 1.0; bc_val[..., :, -1] = v
    return bc_mask, bc_val

# ---------------- Dataset ----------------
class PackedTensorDataset2D(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path)
        self.f = self.data['f'].float()
        self.u = self.data['u'].float()
        assert self.f.shape == self.u.shape and self.f.ndim == 4

    def __len__(self): return self.f.shape[0]
    def __getitem__(self, i): return self.f[i], self.u[i]

# ---------------- ResNet (2D Thermal) ----------------
class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.short = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        r = self.short(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.relu(y + r)
        return y

class ResNetThermal2D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock2D(32, 48, stride=1)
        self.layer2 = ResidualBlock2D(48, 64, stride=2)
        self.layer3 = ResidualBlock2D(64, 84, stride=2)
        self.layer4 = ResidualBlock2D(84, 112, stride=2)

        self.up1 = nn.ConvTranspose2d(112, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64,  48,  kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(48,  32,  kernel_size=2, stride=2)

        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1)
        )

    def forward(self, f, bc_mask, bc_val):
        x = self.stem(f)
        x = self.layer1(x)
        x = self.layer2(x)  # /2
        x = self.layer3(x)  # /4
        x = self.layer4(x)  # /4
        x = self.up1(x)     # /4
        x = self.up2(x)     # /2
        x = self.up3(x)     # /1
        u = self.head(x)
        return u * (1 - bc_mask) + bc_val * bc_mask

# ---------------- Train ----------------
def train_2d_resnet(
    train_pt: str,
    val_pt: str,
    model_ckpt: str = 'ckpt_resnet2d.pt',
    best_ckpt: str = 'best_resnet2d.pt',
    num_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    use_grad_loss: bool = True,
    device: str = None,
    log_csv_path: str = 'training_log.csv',
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    train_ds = PackedTensorDataset2D(train_pt)
    val_ds   = PackedTensorDataset2D(val_pt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    _, _, H, W = train_ds.f.shape
    N = H
    dx = 1.0 / (N - 1)

    model = ResNetThermal2D(in_ch=3, out_ch=1).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}  |  Total: {all_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2)
    loss_fn = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(num_epochs):
        model.train()

        for f, u in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            f = f.to(device); u = u.to(device)
            bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
            f_s, u_s, bv_s = f, u, bc_val
            net_in = torch.cat([f_s, bc_mask, bv_s], dim=1)
            pred = model(net_in, bc_mask, bv_s)
            interior = (bc_mask < 0.5)
            mse_field = loss_fn(pred[interior], u_s[interior])

            if use_grad_loss:
                Txp, Typ = grad_2d_scalar(pred, dx)
                Txt, Tyt = grad_2d_scalar(u_s, dx)
                mse_grad = 0.5 * (loss_fn(Txp[interior], Txt[interior]) +
                                  loss_fn(Typ[interior], Tyt[interior]))
                loss = 0.5 * mse_field + 0.5 * mse_grad
            else:
                loss = mse_field

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # -------- validation --------
        model.eval()
        tot_loss = tot_rel = 0.0
        with torch.no_grad():
            for f, u in train_loader:
                f = f.to(device);
                u = u.to(device)
                bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
                f_s, u_s, bv_s = f, u, bc_val
                net_in = torch.cat([f_s, bc_mask, bv_s], dim=1)
                pred = model(net_in, bc_mask, bv_s)
                interior = (bc_mask < 0.5)
                loss = loss_fn(pred[interior], u_s[interior])
                tot_loss += float(loss.item())
                tot_rel += relative_l2_masked(pred, u_s, bc_mask)
        avg_train = tot_loss / max(1, len(train_loader))
        avg_train_rel = tot_rel / max(1, len(train_loader))

        val_loss = val_rel = 0.0
        with torch.no_grad():
            for f, u in val_loader:
                f = f.to(device); u = u.to(device)
                bc_mask, bc_val = create_bc_mask_and_value_like_2d(f)
                f_s, u_s, bv_s = f, u, bc_val
                net_in = torch.cat([f_s, bc_mask, bv_s], dim=1)
                pred = model(net_in, bc_mask, bv_s)
                interior = (bc_mask < 0.5)
                loss = loss_fn(pred[interior], u_s[interior])
                val_loss += float(loss.item())
                val_rel  += relative_l2_masked(pred, u_s, bc_mask)
        avg_val = val_loss / max(1, len(val_loader))
        avg_val_rel = val_rel / max(1, len(val_loader))
        sch.step(avg_val)

        # save
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'opt': opt.state_dict()}, model_ckpt)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_ckpt)

        print(f"Epoch {epoch+1}/{num_epochs} | Train:{avg_train:.6e}  Rel:{avg_train_rel:.6e} "
              f"| Val:{avg_val:.6e}  Rel:{avg_val_rel:.6e}")

        # log
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train, avg_val])

if __name__ == "__main__":
    mode = "increase"
    train_pt = f"data/training_data_N32_{mode}.pt"
    val_pt = f"data/validation_data_N32_{mode}.pt"

    for i in range(1, 2):
        model_path = f"model_{mode}/resnet/ckpt_2d_resnet_{mode}_{i}.pt"
        best_model_path = f"model_{mode}/resnet/best_2d_resnet_{mode}_{i}.pt"
        log_path = f"model_{mode}/resnet/resnet_{mode}_{i}.csv"

        train_2d_resnet(
            train_pt=train_pt,
            val_pt=val_pt,
            model_ckpt=model_path,
            best_ckpt=best_model_path,
            num_epochs=100,
            batch_size=10,
            lr=1e-3,
            use_grad_loss=True,
            log_csv_path=log_path
        )
