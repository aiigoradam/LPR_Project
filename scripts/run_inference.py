# scripts/run_inference.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import mlflow
import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms
from models.diffusion_sr3 import Diffusion
from tqdm import tqdm
import argparse

# ===== argparse for override =====
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", default="A")
parser.add_argument("--models", type=str)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

# =====================================
# Configuration
# =====================================
DATA_DIR = "data/full_grid"
EXPERIMENT_NAME = args.experiment_name
RESULTS_ROOT = f"results/{EXPERIMENT_NAME}"

MODELS = [
    #"unet_base",
    #"unet_conditional",
    "restormer",
    #"pix2pix",
    #"diffusion_sr3",
]

# override if passed
if args.models:
    MODELS = [m.strip() for m in args.models.split(",") if m.strip()]

# Which of those MODELS should use diffusion-style inference
DIFFUSION_MODELS = {"diffusion_sr3"}


# =====================================
# MLflow model loader
# =====================================
def load_mlflow_model(experiment_name, run_name, device):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found")
    filter_str = f"tags.mlflow.runName = '{run_name}'"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No run '{run_name}' in experiment '{experiment_name}'")
    run_id = runs[0].info.run_id
    uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(uri).to(device).eval()
    print(f"Loaded '{run_name}' (run_id={run_id}) on {device}")
    return model


# =====================================
# Generic image-to-image inference
# =====================================
def run_general_inference(model, records, data_dir, out_dir, device, run_name, batch_size):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    os.makedirs(out_dir, exist_ok=True)

    # start timing
    t0 = time.time()

    # Process in mini-batches
    for start in tqdm(range(0, len(records), batch_size), desc=f"[{run_name}]", unit="batch"):
        batch = records[start : start + batch_size]
        imgs, out_paths = [], []

        # Load + stack
        for rec in batch:
            idx = rec["index"]
            inp = os.path.join(data_dir, f"distorted_{idx}.png")
            out_paths.append(os.path.join(out_dir, f"reconstructed_{idx}.png"))
            imgs.append(to_tensor(Image.open(inp).convert("RGB")))

        x = torch.stack(imgs, 0).to(device)
        with torch.no_grad():
            y = model(x)

        # Clamp & save each
        for img_t, save_path in zip(y.clamp(0, 1), out_paths):
            to_pil(img_t.cpu()).save(save_path)

    print(f" Done [{run_name}]: outputs in {out_dir}")
    total = time.time() - t0
    return total / len(records) * 1000


# =====================================
# Diffusion inference
# =====================================
def run_diffusion_inference(model, records, data_dir, out_dir, device, batch_size, steps):
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    to_pil = transforms.ToPILImage()
    diffusion = Diffusion(T=steps, device=device)

    os.makedirs(out_dir, exist_ok=True)

    # start timing
    t0 = time.time()

    for start in tqdm(range(0, len(records), batch_size), desc=f"[diffusion:{steps}]", unit="batch"):
        batch = records[start : start + batch_size]
        # Load & normalize the distorted plates
        conds = []
        for rec in batch:
            img = Image.open(os.path.join(data_dir, f"distorted_{rec['index']}.png")).convert("RGB")
            conds.append(to_tensor(img))

        cond_batch = torch.stack(conds, dim=0).to(device)
        # DDIM sampling in v-prediction mode
        with torch.no_grad():
            x_clean = diffusion.sample(model, cond_batch, steps=steps)

        # Un-normalize from [-1,1] -> [0,1]
        x_clean = (x_clean * 0.5 + 0.5).clamp(0.0, 1.0)

        # Save outputs
        for rec, img_t in zip(batch, x_clean):
            out_path = os.path.join(out_dir, f"reconstructed_{rec['index']}.png")
            to_pil(img_t.cpu()).save(out_path)

    print(f" Done [diffusion:{steps}]: outputs in {out_dir}")
    total = time.time() - t0
    avg_ms = total / len(records) * 1000
    print(f" Done [diffusion:{steps}]: avg {avg_ms:.1f} ms/image, outputs in {out_dir}")
    return avg_ms


# =====================================
# Main loop
# =====================================
def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    stats_path = os.path.join(RESULTS_ROOT, "inference_times.csv")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            next(f)  # skip header
            for line in f:
                model, val = line.strip().split(",")
                stats[model] = float(val)

    with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
        records = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_name in MODELS:
        print(f"\n=== Inference for {run_name} ===")
        model = load_mlflow_model(EXPERIMENT_NAME, run_name, device)
        out_dir = os.path.join(RESULTS_ROOT, run_name)

        if run_name in DIFFUSION_MODELS:
            avg_ms = run_diffusion_inference(model, records, DATA_DIR, out_dir, device, args.batch_size, args.steps)
        else:
            avg_ms = run_general_inference(model, records, DATA_DIR, out_dir, device, run_name, args.batch_size)

        stats[run_name] = avg_ms

    with open(stats_path, "w") as f:
        f.write("model,avg_time_ms\n")
        for model, val in stats.items():
            f.write(f"{model},{val:.3f}\n")

    print(f"\nWrote inference times to {stats_path}")


if __name__ == "__main__":
    main()
