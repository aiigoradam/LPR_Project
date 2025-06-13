# scripts/run_inference.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True, help="folder name under results/ for this model")
parser.add_argument("--data-dir", default="data/full_grid")
parser.add_argument("--results-root", default="results")
args = parser.parse_args()

run_name = args.run_name
data_dir = args.data_dir
results_root = args.results_root


# =========================
# CONFIG
# =========================
experiment_name = "Test_Extreme_samples"  #  MLflow experiment
run_name = "unet_residual"  # the run_name used when training
data_dir = "data/full_grid"
results_root = "results"  # root folder for all outputs
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fetch the right run from MLflow
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name(experiment_name)
if exp is None:
    raise RuntimeError(f"Experiment '{experiment_name}' not found in MLflow")

filter_str = f"tags.mlflow.runName = '{run_name}'" if run_name else None
runs = client.search_runs(
    experiment_ids=[exp.experiment_id], filter_string=filter_str, order_by=["attributes.start_time DESC"], max_results=1
)
if not runs:
    raise RuntimeError(f"No runs found in '{experiment_name}' with run_name='{run_name}'")
run_id = runs[0].info.run_id

model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri).to(device).eval()
print(f"Loaded model from {experiment_name}/{run_name or run_id} (run_id={run_id}) on {device}")

# 2) Read metadata.json
meta_path = os.path.join(data_dir, "metadata.json")
with open(meta_path, "r") as f:
    records = json.load(f)

# Prepare transforms matching the training
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Create output folder
out_dir = os.path.join(results_root, experiment_name, run_name or run_id)
os.makedirs(out_dir, exist_ok=True)

# Inference loop
inference_time = 0.0
for rec in tqdm(records, desc=f"Inference {experiment_name}/{run_name}", total=len(records)):
    idx = rec["index"]
    inp_path = os.path.join(data_dir, f"distorted_{idx}.png")
    out_path = os.path.join(out_dir, f"reconstructed_{idx}.png")

    # load and preprocess
    img = Image.open(inp_path).convert("RGB")
    x = to_tensor(img).unsqueeze(0).to(device)

    # inference
    inference_start_time = time.time()
    with torch.no_grad():
        y = model(x)
    inference_end_time = time.time()
    inference_time += inference_end_time - inference_start_time

    # postprocess & save
    y_img = to_pil(y.squeeze(0).clamp(0, 1).cpu())

    if run_name == "unet_residual":
        clean = torch.clamp(x - y, 0.0, 1.0)
        y_img = to_pil(clean.squeeze(0).cpu())

    y_img.save(out_path)

avg_inference_time = inference_time / len(records)
avg_inference_time_ms = avg_inference_time * 1000


print(f"\nDone! Reconstructed images are in:\n  {out_dir}")
print(f"Average inference time per image: {avg_inference_time_ms:.2f} milliseconds")
