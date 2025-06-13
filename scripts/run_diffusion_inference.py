# scripts/run_diffusion_inference.py

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

# === CONFIG ===
experiment_name = "Test_2"
run_name        = "diffusion"
data_dir        = "data/full_grid"
results_root    = "results/diffusion"
T               = 100               
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============

# 1) Load MLflow run
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name(experiment_name)
if exp is None:
    raise RuntimeError(f"Experiment '{experiment_name}' not found")
runs = client.search_runs(
    [exp.experiment_id],
    f"tags.mlflow.runName = '{run_name}'",
    order_by=["attributes.start_time DESC"],
    max_results=1
)
if not runs:
    raise RuntimeError(f"No run '{run_name}' in experiment '{experiment_name}'")
run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri).to(device).eval()

# 2) Import your Diffusion helper
from utils.utils_diffusion import Diffusion
diffusion = Diffusion(T=T, device=device)

# 3) Prepare transforms
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
to_pil = transforms.ToPILImage()

# 4) Read metadata and make output dir
meta = json.load(open(os.path.join(data_dir, "metadata.json")))
os.makedirs(results_root, exist_ok=True)

# 5) Inference loop
# process in batches of 64
batch_size = 64
for i in tqdm(range(0, len(meta), batch_size), desc="Diffusion Inference"):
    batch = meta[i : i + batch_size]

    # 1) load & preprocess entire batch
    x_list = []
    for rec in batch:
        idx = rec["index"]
        img = Image.open(os.path.join(data_dir, f"distorted_{idx}.png")).convert("RGB")
        x = to_tensor(img)
        x_list.append(x)
    x_noisy = torch.stack(x_list, dim=0).to(device)

    # 2) sample reverse diffusion on the batch
    with torch.no_grad():
        x_clean = diffusion.sample_ddpm(model, x_noisy, num_inference_steps=T)
    # de-normalize and clamp
    x_clean = x_clean * 0.5 + 0.5
    x_clean = torch.clamp(x_clean, 0.0, 1.0)

    # 3) save each image in the batch
    for rec, img_tensor in zip(batch, x_clean):
        idx = rec["index"]
        out_img = to_pil(img_tensor.cpu())
        out_img.save(os.path.join(results_root, f"reconstructed_{idx}.png"))
