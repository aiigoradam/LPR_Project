## Project Framework Overview

All code is run in a dedicated Anaconda environment (`environment.yml` file), which includes PyTorch, OpenCV, Tesseract OCR, MLflow, and other required libraries. Installing and activating this environment ensures all dependencies are available for every script. The project is organized as follows:

* **scripts/** – `lp_processing.py` for data generation, splitting, and storing; `run_inference.py` for inference on the full grid and saving the resulting images; `compute_metrics.py` for evaluation.
* **models/** – Contains model code such as official PyTorch models from GitHub (e.g., Restormer), and adapted or custom models.
* **src/** – Contains training scripts for each model in the form `train_{model_name}.py`, shared utility functions in `utils.py`, and a custom `lp_dataset.py` class used with data loaders to feed data into models during training.
* **data/** – Data folder containing dataset folders A, B, C, and a `full_grid` folder. Each dataset contains subfolders for train, validation, and test splits, with images and a metadata file. `full_grid` directly includes all images and metadata.
* **Jupyter Notebooks** – Notebooks at the project root for experimentation and analysis. `ocr_test.ipynb` used for testing OCR, `sampling.ipynb` for experimenting and visualizing PDF and sampling, `report_artifacts.ipynb` for report figures, and `results.ipynb` for organizing, analyzing, and making figures of raw results data from CSV files.

---

### Dataset Generation

`lp_processing.py` – This script runs the full dataset generation pipeline, from creating clean plates to saving them in the data folder. In the main, we can set all the related parameters of the clean plates’ dimensions, font, the sampling PDF, the number of samples, etc. By default, three datasets are created and saved in `data/{dataset}/split/` folders as paired `original_{index}.png` and `distorted_{index}.png` images. Each split includes a `metadata.json` file that records the plate text, distortion angles, and bounding boxes for every digit, linking each distorted image with its clean original. A fixed random seed is used for reproducibility.

---

### Models and Training Workflow

The `models/` folder stores all our models: U-Net, Conditional U-Net, Restormer, Pix2Pix GAN, diffusion SR3. Each model takes a distorted plate image and returns a restored image of the same size.

Training scripts (`src/`) all follow the same steps:

* **Configuration**: The script reads command-line arguments and builds a config with epochs, batch size, learning rate, weight decay, and any model-specific settings.
* **Data loading**: The custom `LicensePlateDataset` class reads all image pairs into memory once and applies any necessary transforms (tensor conversion, normalization). PyTorch DataLoaders turn the dataset class into an iterable over batches, shuffle batches (only for training set), and deliver them to the model in the training loop.
* **Model setup**: The selected model is moved to the GPU. The script defines a loss (e.g., MSE) and an optimizer (e.g., AdamW). It defines a learning rate scheduler (e.g., CosineAnnealingLR). A fixed seed fixes data order and weight initialization.
* **MLflow tracking**: The script starts an MLflow run. It logs hyperparameters and copies the training script and model file as artifacts. Metrics and sample images are logged during training. Those artifacts are stored locally and can be accessed through the MLflow UI.
* **Training loop**: For U-Net and GANs, each epoch shuffles batches, runs forward pass, computes loss, back-propagates, updates weights, then runs a no-gradient validation pass. Diffusion follows the same cycle, but first adds noise to each input and predicts that noise across scheduled timesteps. After validation, the script logs loss, MSE, SSIM, and PSNR to MLflow and keeps the model state with the best validation SSIM.
* **Post-training**: After the final epoch or step, the best model weights are loaded. Evaluated on the unseen test set. Test MSE, SSIM, and PSNR are logged. The trained model is logged with an environment file for dependencies.

---

### Inference and Evaluation Scripts

* **run\_inference.py** – This script loads trained models from MLflow and applies them to the `data/full_grid/` set. It loads each model under the chosen dataset name, moves the model to the GPU, and enables evaluation mode. For U-Net, U-Net Conditional, Restormer, and Pix2Pix, each distorted image is fed through the network, and the restored output is saved in `results/{dataset}/{model}/`. For the diffusion model, it performs the scheduled denoising loop before saving the image. The script measures the average inference time per image and writes it to `inference_times.csv`. Command line flags control parameters such as model names, batch size, and diffusion sampling steps.
* **compute\_metrics.py** – Script compares each reconstructed image with its clean original. It looks up the plate text, angle pair, and digit boxes in `metadata.json`. For each image, it reports plate-level and digit-level metrics. It runs Tesseract in digit mode, extracts the digit patch, preprocesses it, and applies recognition. If recognition fails or is incomplete, the script tries alternative preprocessing or attempts to identify the digit from a full plate. All values (raw data) are saved to a CSV at `results/{dataset}/{model_name}.csv`.
* **run\_all.py** – This script automates the workflow for any dataset A, B, or C. It activates the project’s environment. It runs the training scripts for each model in sequence. After training, it calls `run_inference.py`. It then calls `compute_metrics.py`. The script can be configured to enable or skip specific models and steps. It uses subprocess to launch each script in the correct order. This allows running the entire pipeline with a single command once the data is available.

---

### Jupyter Notebooks

Several notebooks are included at the project root. These are not part of the main pipeline, but are used for checking, validation, and making figures.

* **ocr\_test.ipynb** – In this notebook, we test OCRs. We check that the OCR and preprocessing steps can read digits from distorted plates. Different preprocessing options are tried. The final approach is used in the main pipeline. This notebook confirms that OCR will work before it is used in `compute_metrics.py`.
* **report\_artifacts.ipynb** – This notebook makes the figures and sample images used in the project report. Here, we break down our code and plot intermediate results and other relevant illustrations.
* **sampling.ipynb** – In this notebook, we test the sampling distribution for rotation angles. We select parameters and plot PDFs and how rotation angles are sampled from the PDF for each dataset. We compare sampling methods. This helps us visually verify correctness of our distribution and sampling. We use those figures in the report. It is an extension to `report_artifacts.ipynb`.
* **results.ipynb** – In this notebook, we load the CSV files (raw data) from `compute_metrics.py` for each model and dataset as pandas dataframes. We compute means over all samples for each angle pair to get a general overview of each model. We transform the dataframes to draw statistics and make different plots, such as heatmaps, bar charts, scatter plots, and tables. This notebook is used to prepare the main summary figures and tables for the final report.


## Contributors

- **Igor Adamenko**
- **Orpaz Ben Aharon**
- **Mentor**: Dr. Sasha Apartsin
