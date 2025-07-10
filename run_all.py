# run_all.py

import os
import subprocess

# ============ USER CONFIGURATION ============

CONDA_ENV = "lpr2"
DATASET_NAME = "B"
DATA_ROOT = os.path.join("data", DATASET_NAME)

TRAIN_SCRIPTS = [
    #"src/train_unet_base.py",
    # "src/train_unet_conditional.py",
    # "src/train_restormer.py",
    # "src/train_pix2pix.py",
    # "src/train_diffusion_sr3.py",
]

INFERENCE_MODELS = [
    "unet_base",
    "unet_conditional",
    "restormer",
    "pix2pix",
    "diffusion_sr3",
]

METRICS_MODELS = [
    "unet_base",
    "unet_conditional",
    "restormer",
    "pix2pix",
    "diffusion_sr3",
]

RUN_INFERENCE = True
RUN_METRICS = True

INFERENCE_SCRIPT = "scripts/run_inference.py"
METRICS_SCRIPT = "scripts/compute_metrics.py"

# ============ HELPERS ============


def conda_call(args):
    inner = " ".join(["python"] + args)
    return ["cmd", "/c", f"call conda activate {CONDA_ENV} && {inner}"]


def run_training():
    print(f"\n=== TRAINING on dataset '{DATASET_NAME}' ===")
    for script in TRAIN_SCRIPTS:
        cmd = conda_call([script, "--experiment-name", DATASET_NAME, "--data-dir", DATA_ROOT])
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


def run_inference():
    print(f"\n=== INFERENCE (models: {INFERENCE_MODELS}) on full_grid ===")
    cmd = conda_call(
        [
            INFERENCE_SCRIPT,
            "--experiment-name",
            DATASET_NAME,
            "--models",
            ",".join(INFERENCE_MODELS),
            "--steps",
            "10",
            "--batch-size",
            "64",
        ]
    )
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_metrics():
    print(f"\n=== METRICS (models: {METRICS_MODELS}) on full_grid ===")
    cmd = conda_call([METRICS_SCRIPT, "--experiment-name", DATASET_NAME, "--models", ",".join(METRICS_MODELS)])
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if TRAIN_SCRIPTS:
        run_training()
    else:
        print("Skipping training")

    if RUN_INFERENCE:
        run_inference()
    else:
        print("Skipping inference")

    if RUN_METRICS:
        run_metrics()
    else:
        print("Skipping metrics")

    print("\n=== ALL TASKS COMPLETE ===")
