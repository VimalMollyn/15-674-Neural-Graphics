import os
import torch
from PIL import Image
import numpy as np
import random
from pathlib import Path
import math


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# P6
def quaternion_to_rotation(q):
    # q: (N, 4) as (w, x, y, z)
    # TODO: normalize q, then build R(q) above          -> (N, 3, 3)
    q_normalized = q / torch.linalg.norm(q, dim=-1, keepdim=True)  # (N, 4)

    w, x, y, z = q_normalized[:, 0], q_normalized[:, 1], q_normalized[:, 2], q_normalized[:, 3]

    R_q = torch.zeros((q.shape[0], 3, 3), device=q.device)  # (N, 3, 3)
    R_q[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R_q[:, 0, 1] = 2 * (x * y - w * z)
    R_q[:, 0, 2] = 2 * (x * z + w * y)
    R_q[:, 1, 0] = 2 * (x * y + w * z)
    R_q[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R_q[:, 1, 2] = 2 * (y * z - w * x)
    R_q[:, 2, 0] = 2 * (x * z - w * y)
    R_q[:, 2, 1] = 2 * (y * z + w * x)
    R_q[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R_q 

# P6
def covariance_3d(scale, quat):
    # scale: (N, 3) positive,  quat: (N, 4)
    # TODO: R = quaternion_to_rotation(quat); return R S S^T R^T  -> (N, 3, 3)
    R = quaternion_to_rotation(quat)  # (N, 3, 3)
    S = torch.diag_embed(scale).to(scale.device)  # (N, 3, 3)

    sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)  # (N, 3, 3)
    return sigma

def project_gaussian(mu3, Sigma3, R_wc, t, K):
    # mu3: (N, 3) world means,  Sigma3: (N, 3, 3) world covariances
    mu_cam = mu3 @ R_wc.T + t                  # world -> camera
    # TODO: mu2   = perspective-project mu_cam with K            (N, 2)
    mu2 = (mu_cam @ K.T)[:, :2] / mu_cam[:, 2:3]  # (N, 2) # normalize by z
    # TODO: J     = Jacobian of the projection at mu_cam         (N, 2, 3)
    J = torch.zeros((mu_cam.shape[0], 2, 3), device=mu_cam.device)  # (N, 2, 3)
    J[:, 0, 0] = K[0, 0] / mu_cam[:, 2]
    J[:, 0, 2] = -K[0, 0] * mu_cam[:, 0] / (mu_cam[:, 2] ** 2)
    J[:, 1, 1] = K[1, 1] / mu_cam[:, 2]
    J[:, 1, 2] = -K[1, 1] * mu_cam[:, 1] / (mu_cam[:, 2] ** 2)


    # TODO: Scam  = R_wc @ Sigma3 @ R_wc.T                       (N, 3, 3)
    Scam = R_wc @ Sigma3 @ R_wc.transpose(-1, -2)  # (N, 3, 3)
    #       Sig2 = J @ Scam @ J.transpose(-1, -2)                (N, 2, 2)
    Sig2 = J @ Scam @ J.transpose(-1, -2)  # (N, 2, 2)
    depth = mu_cam[:, 2]
    return mu2, Sig2, depth


# P1
# def gaussian_weight(xy, mu, Sigma):
#     # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
#     # w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))
# 
#     xy_expanded = xy.unsqueeze(1)  # (P, 1, 2)
#     mu_centered = (xy_expanded - mu.unsqueeze(0)).unsqueeze(-2)  # (P, N, 1, 2)
#     sigma_inv = (torch.linalg.inv(Sigma)).unsqueeze(0)  # (1, N, 2, 2)
#     w_inner = -0.5 * (mu_centered @ sigma_inv @ mu_centered.transpose(-1, -2))  # (P, N, 1, 1)
# 
#     w = torch.exp(w_inner.squeeze(-1).squeeze(-1))  # (P, N)
# 
#     return w # (P, N)

def gaussian_weight(xy, mu, Sigma):
    a, b, d = Sigma[:, 0, 0], Sigma[:, 0, 1], Sigma[:, 1, 1]      # (N,)
    det = a * d - b * b
    ia, ib, id_ = d / det, -b / det, a / det                      # entries of Sigma^-1
    dx = xy[:, 0:1] - mu[None, :, 0]                              # (P, N)
    dy = xy[:, 1:2] - mu[None, :, 1]                              # (P, N)
    return torch.exp(-0.5 * (ia * dx * dx + 2 * ib * dx * dy + id_ * dy * dy))

# P2 (vectorized)
def render(mu, Sigma, color, opacity, order, H, W):
    # color: (N, 3),  opacity: (N,) in [0, 1],  order: indices sorted front -> back
    device, dtype = mu.device, mu.dtype
    xs = torch.arange(W, device=device, dtype=dtype)
    ys = torch.arange(H, device=device, dtype=dtype)
    xy = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1).reshape(-1, 2)  # (H*W, 2), pixel p = y*W + x holds (x, y)

    alpha = opacity[None, :] * gaussian_weight(xy, mu, Sigma)  # (P, N)
    alpha = alpha[:, order]                                      # front -> back
    col   = color[order]                                         # (N, 3)

    # transmittance T_n = prod_{k<n} (1 - alpha_k): exclusive cumprod along the gaussian axis
    T = torch.cumprod(1 - alpha, dim=1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=1)  # (P, N)

    C = (T * alpha) @ col                                        # (P, 3)
    return C.reshape(H, W, 3)



# P3
# parameters (leaf tensors, requires_grad=True); a spread-out init, e.g.:
N = 1024
# N = 1024
W = 160
H = 160

import json

device = get_device()

camera_data = json.load(open("images/spheres/cameras.json"))
W = camera_data["width"]
H = camera_data["height"]
K = torch.tensor(camera_data["K"], dtype=torch.float32, device=device)  # (3, 3)
train_cameras = camera_data["frames"]
val_cameras = camera_data["val_frames"]


mu3    = ((torch.rand(N, 3) * 2 - 1) * 1.5).to(device).requires_grad_(True)       # (N, 3)  cloud in ~[-1.5, 1.5]^3
log_s  = (torch.log(0.08 * torch.ones(N, 3))).to(device).requires_grad_(True) # (N, 3)  small 3D blobs log space

quat   = torch.zeros(N, 4, device=device)
quat[:, 0] = 1.0     # (N, 4)  identity rotation (w, x, y, z)
quat.requires_grad_(True)

color  = torch.zeros(N, 3, requires_grad=True, device=device)                                # (N, 3)  sigmoid -> 0.5 gray
op_raw = torch.full((N,), -2.0, requires_grad=True, device=device)                           # (N,)    sigmoid -> ~0.12 opacity

# parameters (leaf tensors, requires_grad=True); example init for this scene:
opt    = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)


# densification params
densify_every  = 200          # run a pass every 200 optimization steps
# grad_threshold = 5e-4         # densify Gaussian i if g_i > grad_threshold
# size_threshold = 0.1         # clone if max scale <= size_threshold TODO update this to world units
split_scale    = 1.6          # each split child gets (parent scale / split_scale)
split_jitter   = 0.05          # std-dev in world space of the random offset given to clones and split children
prune_opacity  = 0.01        # remove Gaussian i if its opacity < this
max_count      = 2048       # a target count N from P5 (e.g. 256, 1024, 4096)

# compute size threshold from the train cameras sphere.
cam_centers = torch.stack([ -torch.tensor(f['R_wc'], dtype=torch.float32).T @ torch.tensor(f['t'], dtype=torch.float32) for f in train_cameras ])                                                                # (F, 3)
scene_extent = (cam_centers - cam_centers.mean(0)).norm(dim=1).max().item()
percent_dense = 0.02
size_threshold = scene_extent * percent_dense

run_densify = True
# run_densify = False

# run inside the P3 training loop, every densify_every steps
def densify(gaussians, grad_mag, budget):
    mu, log_s, quat, color, op_raw = gaussians

    # loop in order of grad magnitude
    order = torch.argsort(grad_mag, descending=True)
    grad_mag = grad_mag[order]

    grad_threshold = torch.quantile(grad_mag, 0.7)  
    # print(grad_mag.max(), grad_mag.mean(), grad_mag.min(), grad_mag.quantile(0.5), grad_mag.quantile(0.9))
    mu = mu[order]
    log_s = log_s[order]
    quat = quat[order]
    color = color[order]
    op_raw = op_raw[order]

    op = op_raw.sigmoid()
    prune = op < prune_opacity  # remove any guassian that has low opacity

    # also prune based on max size
    max_world_size = scene_extent * 0.2
    too_big = log_s.exp().max(dim=1).values > max_world_size
    # max scale ratio can't exceed 10:1
    too_thin = log_s.exp().max(dim=1).values / log_s.exp().min(dim=1).values > 5.0
    prune = prune | too_big | too_thin


    mu, log_s, quat, color, op_raw, grad_mag = (x[~prune] for x in (mu, log_s, quat, color, op_raw, grad_mag))
    print(f"Densify: {(op < prune_opacity).sum().item()} pruned (opacity), {too_big.sum().item()} pruned (size), {too_thin.sum().item()} pruned (thin), {mu.shape[0]} remaining")

    dense = grad_mag > grad_threshold

    # calculate how many more gaussians we can add
    room_left = max(budget - mu.shape[0], 0)
    dense[room_left:] = False

    # clone and split
    max_scale = log_s.exp().max(dim=1).values  # (N,) max scale of each Gaussian



    clone = dense & (max_scale <= size_threshold)  # duplicate in place

    def parent_offset(log_s, quat, frac=1.0):
        R   = quaternion_to_rotation(quat)                       # (M, 3, 3)
        eps = torch.randn_like(log_s) * log_s.exp() * frac   # per-axis std = parent scale
        return (R @ eps.unsqueeze(-1)).squeeze(-1)
    
    mu = torch.cat([mu, mu[clone] + parent_offset(log_s[clone], quat[clone])], dim=0) # cloen with a jitter so that they can drift apart
    log_s = torch.cat([log_s, log_s[clone]], dim=0)
    quat = torch.cat([quat, quat[clone]], dim=0)
    color = torch.cat([color, color[clone]], dim=0)
    op_raw = torch.cat([op_raw, op_raw[clone]], dim=0)
    dense = torch.cat([dense, dense[clone]], dim=0) 
    max_scale = torch.cat([max_scale, max_scale[clone]], dim=0)

    print(f"Densify: {clone.sum().item()} cloned, {mu.shape[0]} remaining")


    split = dense & (max_scale > size_threshold)  # 2 children, scale / split_scale
    # replace the parent with the 2 children + jitter
    mu = torch.cat([mu[~split], mu[split] + parent_offset(log_s[split], quat[split]), mu[split] + parent_offset(log_s[split], quat[split])], dim=0)
    log_s = torch.cat([log_s[~split], log_s[split] - torch.log(torch.tensor(split_scale, device=device)), log_s[split] - torch.log(torch.tensor(split_scale, device=device))], dim=0)
    quat = torch.cat([quat[~split], quat[split], quat[split]], dim=0)
    color = torch.cat([color[~split], color[split], color[split]], dim=0)
    op_raw = torch.cat([op_raw[~split], op_raw[split], op_raw[split]], dim=0)

    print(f"Densify: {split.sum().item()} split, {mu.shape[0]} remaining")
    print(f"Densify: {len(mu)} total after densification")

    changed = bool(prune.any() or clone.any() or split.any())

    return mu, log_s, quat, color, op_raw, changed

grad_accum = torch.zeros(N, device=device)

iters = 1500
densify_until = iters // 2

for step in range(iters):                        # e.g. N = 4000 Gaussians, iters = 1500
    cam   = random.choice(train_cameras)

    img_path = f"images/spheres/{cam['file']}"
    R_wc = torch.tensor(cam['R_wc'], dtype=torch.float32, device=device)  # (3, 3)
    t = torch.tensor(cam['t'], dtype=torch.float32, device=device)        # (3,)

    target = Image.open(img_path).convert("RGB")
    target = torch.tensor(np.asarray(target), dtype=torch.float32) / 255.0  # (H, W, 3)
    target = target.to(device)

    Sig3  = covariance_3d(log_s.exp(), quat)            # 3D scale + rotation
    mu2, Sig2, depth = project_gaussian(mu3, Sig3, R_wc, t, K)
    order = torch.argsort(depth, descending=False)      # front-to-back: nearest (smallest z_c) first
    img   = render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, H, W)
    loss  = ((img - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()

    if run_densify:
        with torch.no_grad():
            grad_accum += mu3.grad.norm(dim=1)

    opt.step()
    with torch.no_grad():  
        log_s.clamp_(np.log(0.02), np.log(1.0)) # clamp scales in world space to avoid nans and thin gaussians

    if run_densify and ((step + 1) % densify_every == 0) and step < densify_until:
        g = grad_accum / densify_every

        with torch.no_grad():
            r_mu3, r_log_s, r_quat, r_color, r_op_raw, changed = densify(gaussians=(mu3, log_s, quat, color, op_raw), grad_mag=g, budget=max_count)

            # if ((step+1) % opacity_reset_every == 0) and step <0.8*iters:
            #     op_raw.clamp_(max=math.log(0.01/0.99))  # reset opacity might help with floaters

        
        if changed:
            print("resetting optimizer due to densification")
            mu3, log_s, quat, color, op_raw = [x.detach().requires_grad_(True) for x in (r_mu3, r_log_s, r_quat, r_color, r_op_raw)]
            depth_order = torch.arange(len(mu3), device=device)  # reset depth order

            # reset adam
            opt = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)
        grad_accum = torch.zeros(len(mu3), device=device)


    psnr = -10 * torch.log10(loss)
    if step % 100 == 0:
        print(f"step {step:04d} loss {loss.item():.6f} psnr {psnr.item():.2f}")

# render and save
N_final = len(mu3)                              # count after densification (== N if run_densify is False)
if run_densify:
    out_root = Path(f"images/densify_spheres_renders_N{N}_final{N_final}")
else:
    out_root = Path(f"images/spheres_renders_N{N_final}")

summary = {"N": N_final, "N_init": N, "max_count": max_count if run_densify else N, "steps": step + 1}

# save the gaussians
to_save = {
    "mu3":    mu3.detach().cpu(),        
    "log_s":  log_s.detach().cpu(),      
    "quat":   quat.detach().cpu(),       
    "color":  color.detach().cpu(),      
    "op_raw": op_raw.detach().cpu(),     
    "meta": {"N_init": N, "N_final": len(mu3), "steps": step + 1, "max_count": max_count, "run_densify": run_densify},
}


out_root.mkdir(parents=True, exist_ok=True)
torch.save(to_save, f"{out_root}/gaussians.pt")

with torch.no_grad():
    for split, cams in [("train", train_cameras), ("val", val_cameras)]:
        save_dir = out_root /f"{split}"
        save_dir.mkdir(parents=True, exist_ok=True)

        psnrs = {}
        for cam in cams:
            img_path = f"images/spheres/{cam['file']}"
            R_wc = torch.tensor(cam['R_wc'], dtype=torch.float32, device=device)  # (3, 3)
            t = torch.tensor(cam['t'], dtype=torch.float32, device=device)        # (3,)

            target = Image.open(img_path).convert("RGB")
            target = torch.tensor(np.asarray(target), dtype=torch.float32) / 255.0  # (H, W, 3)
            target = target.to(device)

            Sig3  = covariance_3d(log_s.exp(), quat)            # 3D scale + rotation
            mu2, Sig2, depth = project_gaussian(mu3, Sig3, R_wc, t, K)
            Sig2 = Sig2 + 0.3 * torch.eye(2, device=Sig2.device)
            order = torch.argsort(depth, descending=False)      # front-to-back: nearest (smallest z_c) first
            img   = render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, H, W)

            mse    = ((img - target) ** 2).mean()
            psnr = (-10 * torch.log10(mse)).item() 
            psnrs[cam['file']] = psnr

            stem = os.path.splitext(os.path.basename(cam['file']))[0]
            pil_image = Image.fromarray((img.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8"))
            pil_image.save(save_dir / f"{stem}_psnr{psnr:.2f}.png")

        mean_psnr = float(np.mean(list(psnrs.values())))
        summary[split] = {"mean_psnr": mean_psnr, "per_view_psnr": psnrs}
        print(f"[{split}] mean psnr over {len(psnrs)} views: {mean_psnr:.2f} dB")

print(f"train mean psnr {summary['train']['mean_psnr']:.2f} dB | val mean psnr {summary['val']['mean_psnr']:.2f} dB")
with open(f"{out_root}/psnr_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"renders + psnr_summary.json written to {out_root}/")
