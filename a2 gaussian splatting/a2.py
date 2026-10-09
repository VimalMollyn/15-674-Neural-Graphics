import os
import torch
from PIL import Image
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# P1
def covariance_2d(scale, theta):
    # scale: (N, 2) positive,  theta: (N,) radians
    # TODO: build R(theta) and S = diag(scale), return Sigma = R S S^T R^T  -> (N, 2, 2)
    S = torch.diag_embed(scale).to(scale.device)  # (N, 2, 2)
    R = torch.zeros((scale.shape[0], 2, 2), device=scale.device)  # (N, 2, 2)
    R[:, 0, 0] = torch.cos(theta)
    R[:, 0, 1] = -torch.sin(theta)
    R[:, 1, 0] = torch.sin(theta)
    R[:, 1, 1] = torch.cos(theta)

    sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)  # (N, 2, 2)
    return sigma

# ---- original loop render (kept for reference) ----
# # P2
# # TODO: vectorize to make faster
# def render(mu, Sigma, color, opacity, order, H, W):
#     # color: (N, 3),  opacity: (N,) in [0, 1],  order: indices sorted front -> back
#     # xy = pixel_grid(H, W)                     # (H*W, 2)
#
#     # move to the right device
#     device = Sigma.device
#     opacity = opacity.to(device)
#     color = color.to(device)
#
#     heights = torch.arange(H)
#     widths = torch.arange(W)
#     xy = torch.stack(torch.meshgrid(widths, heights, indexing='xy'), dim=-1).reshape(-1, 2).to(device)  # (H*W, 2)
#     w  = gaussian_weight(xy, mu, Sigma)       # (P, N)  from P1
#     alpha = opacity[None, :] * w              # (P, N)
#     C = torch.zeros(H * W, 3).to(device)  # (P, 3)
#     T = torch.ones(H * W).to(device)  # (P,)
#     for i in order:                           # front to back
#         a = alpha[:, i]
#         # TODO: C and T compositing here
#         C = C + (T * a).unsqueeze(-1) * color[i] # H*W, 3
#         T = T * (1 - a) # H*W
#     return C.reshape(H, W, 3)
# ---- end original ----

# P1
def gaussian_weight(xy, mu, Sigma):
    # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
    # w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))

    xy_expanded = xy.unsqueeze(1)  # (P, 1, 2)
    mu_centered = (xy_expanded - mu.unsqueeze(0)).unsqueeze(-2)  # (P, N, 1, 2)
    sigma_inv = (torch.linalg.inv(Sigma)).unsqueeze(0)  # (1, N, 2, 2)
    w_inner = -0.5 * (mu_centered @ sigma_inv @ mu_centered.transpose(-1, -2))  # (P, N, 1, 1)

    w = torch.exp(w_inner.squeeze(-1).squeeze(-1))  # (P, N)

    return w # (P, N)

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
N = 128
# N = 1024
W = 128
H = 128

# load an image
images = ["coffee.png", "cat.png", "astronaut.png"]
# for N in [128, 1024, 4096]:
for N in [256]:
    for image_name in images:
        img_path = f"images/{image_name}"
        stem = os.path.splitext(os.path.basename(img_path))[0]   # "coffee"
        image = Image.open(img_path).convert("RGB")

        # scale image to W x H
        image = image.resize((W, H))

        target = torch.tensor(np.asarray(image), dtype=torch.float32) / 255.0  # (H, W, 3)

        device = get_device()
        mu     = (torch.rand(N, 2) * torch.tensor([W, H])).to(device).requires_grad_(True)         # (N, 2)  spread across the image
        log_s  = (torch.log(0.02 * max(H, W) * torch.ones(N, 2))).to(device).requires_grad_(True)   # (N, 2)  small blobs, log space
        theta  = torch.zeros(N, requires_grad=True, device=device)                                   # (N,)    rotation
        color  = torch.zeros(N, 3, requires_grad=True, device=device)                                # (N, 3)  sigmoid -> 0.5 gray
        op_raw = torch.full((N,), -2.0, requires_grad=True, device=device)                           # (N,)    sigmoid -> ~0.12 opacity

        depth_order = torch.arange(N, device=device)  # (N,)    fixed ordering since this is just 2D
        target = target.to(device)

        opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)


        # densification params
        densify_every  = 200          # run a pass every 200 optimization steps
        grad_threshold = 5e-6         # densify Gaussian i if g_i > grad_threshold
        size_threshold = 0.02         # clone if max scale <= 2% of image width, else split
        split_scale    = 1.6          # each split child gets (parent scale / split_scale)
        split_jitter   = 0.5          # std-dev in pixels of the random offset given to clones and split children
        prune_opacity  = 0.005        # remove Gaussian i if its opacity < this
        max_count      = 1024       # a target count N from P5 (e.g. 256, 1024, 4096)

        # run_densify = True
        run_densify = False

        # run inside the P3 training loop, every densify_every steps
        def densify(gaussians, grad_mag, budget):
            mu, log_s, theta, color, op_raw = gaussians

            # loop in order of grad magnitude
            order = torch.argsort(grad_mag, descending=True)
            grad_mag = grad_mag[order]

            # grad_threshold = torch.quantile(grad_mag, 0.9)  
            # print(grad_mag.max(), grad_mag.mean(), grad_mag.min(), grad_mag.quantile(0.5), grad_mag.quantile(0.9))
            mu = mu[order]
            log_s = log_s[order]
            theta = theta[order]
            color = color[order]
            op_raw = op_raw[order]

            op = op_raw.sigmoid()
            prune = op < prune_opacity  # remove any guassian that has low opacity

            mu, log_s, theta, color, op_raw, grad_mag = (x[~prune] for x in (mu, log_s, theta, color, op_raw, grad_mag))
            print(f"Densify: {prune.sum().item()} pruned, {mu.shape[0]} remaining")

            dense = grad_mag > grad_threshold

            # calculate how many more gaussians we can add
            room_left = max(budget - mu.shape[0], 0)
            dense[room_left:] = False

            # clone and split
            max_scale_pixels = log_s.exp().max(dim=1).values  # (N,) max scale of each Gaussian
            max_scale = max_scale_pixels / max(H, W)  # (N,) max scale in normalized image coordinates

            clone = dense & (max_scale <= size_threshold)  # duplicate in place

            
            mu = torch.cat([mu, mu[clone] + split_jitter * torch.randn_like(mu[clone])], dim=0) # cloen with a jitter so that they can drift apart
            log_s = torch.cat([log_s, log_s[clone]], dim=0)
            theta = torch.cat([theta, theta[clone]], dim=0)
            color = torch.cat([color, color[clone]], dim=0)
            op_raw = torch.cat([op_raw, op_raw[clone]], dim=0)
            dense = torch.cat([dense, dense[clone]], dim=0) 
            max_scale = torch.cat([max_scale, max_scale[clone]], dim=0)

            print(f"Densify: {clone.sum().item()} cloned, {mu.shape[0]} remaining")


            split = dense & (max_scale > size_threshold)  # 2 children, scale / split_scale
            # replace the parent with the 2 children + jitter
            mu = torch.cat([mu[~split], mu[split] + split_jitter * torch.randn_like(mu[split]), mu[split] + split_jitter * torch.randn_like(mu[split])], dim=0)
            log_s = torch.cat([log_s[~split], log_s[split] - torch.log(torch.tensor(split_scale, device=device)), log_s[split] - torch.log(torch.tensor(split_scale, device=device))], dim=0)
            theta = torch.cat([theta[~split], theta[split], theta[split]], dim=0)
            color = torch.cat([color[~split], color[split], color[split]], dim=0)
            op_raw = torch.cat([op_raw[~split], op_raw[split], op_raw[split]], dim=0)

            print(f"Densify: {split.sum().item()} split, {mu.shape[0]} remaining")
            print(f"Densify: {len(mu)} total after densification")

            return mu, log_s, theta, color, op_raw


        grad_accum = torch.zeros(N, device=device)

        for step in range(2000):
            Sigma = covariance_2d(log_s.exp(), theta)
            img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order, H, W)
            loss  = ((img - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()

            if run_densify:
                with torch.no_grad():
                    grad_accum += mu.grad.norm(dim=1)

            opt.step()
            with torch.no_grad():  
                log_s.clamp_(np.log(0.3), np.log(max(H, W))) # clamp scales to avoid nans
            psnr = -10 * torch.log10(loss)

            if run_densify and ((step + 1) % densify_every == 0):
                g = grad_accum / densify_every

                with torch.no_grad():
                    mu, log_s, theta, color, op_raw = densify(gaussians=(mu, log_s, theta, color, op_raw), grad_mag=g, budget=max_count)

                mu, log_s, theta, color, op_raw = [x.detach().requires_grad_(True) for x in (mu, log_s, theta, color, op_raw)]
                depth_order = torch.arange(len(mu), device=device)  # reset depth order

                # reset adam
                opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
                grad_accum = torch.zeros(len(mu), device=device)

            if step % 100 == 0:
                print(f"step {step:04d} loss {loss.item():.6f} psnr {psnr.item():.2f}")




        # save the optimized image, the resized target
        final_psnr = psnr.item()
        out_path = f"images/{stem}_opt_N{len(mu)}_psnr{final_psnr:.2f}.png"
        print(f"done: N={len(mu)} gaussians, final loss {loss.item():.6f}, psnr {final_psnr:.2f} dB -> {out_path}")

        opt_image = (img.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
        opt_image = Image.fromarray(opt_image)
        opt_image.save(out_path)

        resized_target = (target.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
        resized_target = Image.fromarray(resized_target)
        resized_target.save(f"images/{stem}_resized.png")


