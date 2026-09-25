from pathlib import Path
import torch

path_to_models = Path("models")

variants = ["large", "medium", "small"]

for variant in variants:
    # load the model
    model_path = path_to_models / variant / "bricks.pt"
    model = torch.load(model_path, map_location=torch.device("cpu"))

    print(variant)
    n_bytes = sum(t.numel() * t.element_size() for t in model.values())
    print(n_bytes)

    compression_ratio = n_bytes / (512 * 512 * 3)
    print(f"Compression ratio: {compression_ratio:.3f}")

