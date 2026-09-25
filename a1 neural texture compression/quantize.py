from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F

path_to_models = Path("models")

variants = ["large", "medium", "small"]

device = torch.device("mps")

def get_psnr(original, compressed):
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)

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


# P5 training loop
class NeuralTexture(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = FeatureGrid(resolutions=resolutions, feat_dim=feat_dim)
        self.mlp  = ColorMLP(self.grid.out_dim)
    def forward(self, uv):
        return self.mlp(self.grid(uv))

for variant in variants:
    if variant == "large":
        resolutions = [16, 32, 64, 128]
        feat_dim = 4
    elif variant == "medium":
        resolutions = [16, 32, 64]
        feat_dim = 2
    elif variant == "small":
        resolutions = [64]
        feat_dim = 2

    for name in ["gradient.png", "bricks.png", "clouds.png"]:
        texture_path = Path("./textures") / name
        texture = np.asarray(Image.open(texture_path))

        texture_height, texture_width, _ = texture.shape
        # load the model
        model_path = path_to_models / variant / (texture_path.stem + ".pt")
        model = NeuralTexture().to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        # n_bytes = sum(t.numel() * t.element_size() for t in model.values())
        # print(n_bytes)

        # quantize model
        def quantize_uint8(x):
            lo, hi = torch.min(x.flatten()), torch.max(x.flatten())              # x = one array of float32 values
            scale  = (hi - lo) / 255             # 256 levels (8 bits)
            q      = torch.round((x - lo) / scale)     # integer index in [0, 255]
            x_hat  = lo + q * scale              # dequantized value used at decode
            return q, lo, scale, x_hat

        def quantize_model(model, quantize_mlp=False):
            # each feature-grid level: its own lo/scale, stored as 8-bit q
            for grid in model.grid.grids:
                q, lo, scale, x_hat = quantize_uint8(grid.data)
                grid.data.copy_(x_hat)

            # the MLP is tiny -- quantizing it is optional
            if quantize_mlp:
                for p in model.mlp.parameters():
                    q, lo, scale, x_hat = quantize_uint8(p.data)
                    p.data.copy_(x_hat)
            return model

        model = quantize_model(model, quantize_mlp=False)

        coords = np.stack(np.meshgrid(np.arange(texture_width), np.arange(texture_height)), axis=-1).reshape(-1, 2) + 0.5

        # evaluate PSNR on the full texture
        # save the model
        with torch.no_grad():
            full_coords = torch.tensor(coords, dtype=torch.float32) / torch.tensor([texture_width, texture_height], dtype=torch.float32)
            full_coords = full_coords.to(device)
            full_pred_colors = model(full_coords).cpu().numpy()
            full_pred_colors = np.round(full_pred_colors * 255.0).astype(np.uint8)
            compressed_image = full_pred_colors.reshape(texture_height, texture_width, 3)
            psnr = get_psnr(texture, compressed_image)

        save_dir = texture_path.parent.parent / f"mlp_{variant}_compressed_quantized"
        save_dir.mkdir(exist_ok=True)
        Image.fromarray(compressed_image).save(save_dir / texture_path.name)

        # model_dir = Path(f"./models/{variant}")
        # model_dir.mkdir(parents=True, exist_ok=True)
        # model_path = model_dir / (texture_path.stem + ".pt")
        # torch.save(model.state_dict(), model_path)

        def get_quantized_model_size(model, quantize_mlp=False):
            # each feature-grid level: its own lo/scale, stored as 8-bit q
            total_bytes = 0
            for grid in model.grid.grids:
                q, lo, scale, x_hat = quantize_uint8(grid.data)
                # grid.data.copy_(x_hat)
                total_bytes += q.numel() + lo.element_size() + scale.element_size()

            # the MLP is tiny -- quantizing it is optional
            for p in model.mlp.parameters():
                q, lo, scale, x_hat = quantize_uint8(p.data)
                # p.data.copy_(x_hat)
                if quantize_mlp:
                    total_bytes += q.numel() + lo.element_size() + scale.element_size()
                else:
                    total_bytes += q.numel()*p.element_size()  # add the size of the MLP parameters without quantization
            return total_bytes


        new_model_size = get_quantized_model_size(model, quantize_mlp=False)

        compression_ratio = new_model_size / (512 * 512 * 3)
        print(f"Compression ratio: {compression_ratio:.3f}")
        print(variant, texture_path, psnr, new_model_size)
        print()

