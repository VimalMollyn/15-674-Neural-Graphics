# import torch
import torch, torch.nn as nn, torch.nn.functional as F

from PIL import Image
import numpy as np
import math
from pathlib import Path
import random

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_rgb565(c):
    levels = np.array([31, 63, 31])                # 5, 6, 5 bits for R, G, B
    q      = np.round(c * levels)           # integer 565 code
    c_hat  = q / levels                  # back to [0, 1]
    return c_hat

def get_psnr(original, compressed):
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def train_model(texture_path, variant):
    # P3
    class FeatureGrid(nn.Module):
        def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=4):
            super().__init__()
            # one learnable grid per resolution, each shaped (1, feat_dim, R, R)
            self.grids = nn.ParameterList([nn.Parameter(torch.randn(1, feat_dim, R, R)*0.01) for R in resolutions])
            self.out_dim = feat_dim * len(resolutions)
        
        # uv: (N, 2) in [0, 1], N = batch size
        def forward(self, uv):
            # map uv to [-1, 1] for grid_sample
            grid = (2*uv - 1.0).view(1, 1, uv.shape[0], 2)
            bilinear_samples = [F.grid_sample(g, grid, mode="bilinear", padding_mode="border", align_corners=False) for g in self.grids]

            # concatenate the features across resolutions
            features = torch.cat(bilinear_samples, dim=1)  # (N, out_dim, 1, 1) -> (N, out_dim)
            return features.view(self.out_dim, -1).T

    # P4
    class ColorMLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            # 2 hidden layers of width 64 (Linear + ReLU), then Linear -> 3 and a Sigmoid (RGB in [0, 1])

            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    if variant == "large":
        resolutions = [16, 32, 64, 128]
        feat_dim = 4
    elif variant == "medium":
        resolutions = [16, 32, 64]
        feat_dim = 2
    elif variant == "small":
        resolutions = [64]
        feat_dim = 2

    # P5 training loop
    class NeuralTexture(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid = FeatureGrid(resolutions=resolutions, feat_dim=feat_dim)
            self.mlp  = ColorMLP(self.grid.out_dim)
        def forward(self, uv):
            return self.mlp(self.grid(uv))

    # load texture

    device = get_device()

    # set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    texture = np.asarray(Image.open(texture_path))

    texture_height, texture_width, _ = texture.shape

    # build coords (N, 2) of texel centers in [0, 1] and target colors (N, 3)
    coords = np.stack(np.meshgrid(np.arange(texture_width), np.arange(texture_height)), axis=-1).reshape(-1, 2) + 0.5
    target = texture[coords[:, 1].astype(int), coords[:, 0].astype(int)] / 255.0


    model = NeuralTexture().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_dir = Path(f"./training/{variant}/{texture_path.stem}")
    training_dir.mkdir(parents=True, exist_ok=True)
    training_csv = open(training_dir / "training.csv", "w")

    for step in range(2000):
        # minibatch sampling
        idx = np.random.choice(len(coords), size=16384, replace=False)
        batch_coords = torch.tensor(coords[idx], dtype=torch.float32) / torch.tensor([texture_width, texture_height], dtype=torch.float32)
        batch_target = torch.tensor(target[idx], dtype=torch.float32)

        batch_coords = batch_coords.to(device)
        batch_target = batch_target.to(device)

        # forward pass
        pred_colors = model(batch_coords)

        loss = F.mse_loss(pred_colors, batch_target)
        opt.zero_grad(); loss.backward(); opt.step()

        psnr = -10 * torch.log10(loss).item()
        # log to csv
        training_csv.write(f"{step},{psnr}\n")

        # PSNR from the loss:  psnr = -10 * torch.log10(loss)
        print(f"step {step}: loss = {loss.item():.6f}, psnr = {psnr:.2f} dB", end="\r")

    training_csv.close()

    # evaluate PSNR on the full texture
    # save the model
    with torch.no_grad():
        full_coords = torch.tensor(coords, dtype=torch.float32) / torch.tensor([texture_width, texture_height], dtype=torch.float32)
        full_coords = full_coords.to(device)
        full_pred_colors = model(full_coords).cpu().numpy()
        full_pred_colors = np.round(full_pred_colors * 255.0).astype(np.uint8)
        compressed_image = full_pred_colors.reshape(texture_height, texture_width, 3)
        psnr = get_psnr(texture, compressed_image)

    save_dir = texture_path.parent.parent / f"mlp_{variant}_compressed"
    save_dir.mkdir(exist_ok=True)
    Image.fromarray(compressed_image).save(save_dir / texture_path.name)

    model_dir = Path(f"./models/{variant}")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (texture_path.stem + ".pt")
    torch.save(model.state_dict(), model_path)

    return psnr

texture_names = ["gradient.png", "bricks.png", "clouds.png"]

# for P8
# texture_names = ["vimal.png", "body.png", "many_freqs.png"]

for variant in ["large", "medium", "small"]:
    for name in texture_names:
        texture_path = Path("./textures") / name
        psnr = train_model(texture_path, variant)
        print(variant, texture_path, psnr)
