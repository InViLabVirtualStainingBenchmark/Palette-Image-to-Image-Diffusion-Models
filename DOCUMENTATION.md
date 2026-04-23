# Palette Diffusion Model: Virtual Staining Benchmark Documentation

This document details the environment setup, data routing, and verification steps for the **Palette-Image-to-Image-Diffusion-Models** repository, adapted for virtual staining on BCI and MIST datasets.

---

## 1. Environment Specification
The environment is specified to match High-Performance Computing (HPC) cluster requirements within the constraints of the staging hardware (GTX 1080 Ti).

*   **Operating System:** Linux (WSL Ubuntu 24.04)
*   **Hardware:** 128GB RAM | 1x NVIDIA GTX 1080 Ti (11GB VRAM)
*   **Python Version:** 3.9.25
*   **PyTorch Lock:** Version 2.1.2 (CUDA 12.1)

### Setup Commands:
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_frozen.txt
```

---

## 2. Data Handling & Adaptation Rationale

### Repository Expectations (Original Design)
The Palette repository utilizes a `ColorizationDataset` loader class (found in `data/dataset.py`) designed with specific expectations:
1.  **Folder Structure:** It expects a root directory containing `gray/` (Condition/Input) and `color/` (Ground Truth/Target) sub-folders.
2.  **Naming Convention:** It dynamically generates filenames using a 5-digit zero-padded integer format: `str(index).zfill(5) + '.png'` (e.g., `00001.png`).
3.  **Index Mapping:** It requires a `.flist` file (a simple text list of integers) to define which indices to load.

### Adaptation Strategy
To preserve the repository's original implementation without code modification, the raw BCI and MIST datasets are transformed to fit the repository's native infrastructure:
*   **Routing:** H&E/Unstained images are mapped to the `gray/` folder; IHC/stained images to the `color/` folder.
*   **Renaming:** Original filenames (e.g., `10M2102916_10_15.jpg`) are renamed to the mandatory `00001.png` format.
*   **Format Normalization:** `.jpg` source files are converted to true PNG format using `cv2.imwrite()`, satisfying the loader's hard-coded `.png` extension requirement.
*   **Indexing:** Custom `train.flist` files are generated for each modality containing the sequence `1` to `10`.

**Data preparation:** The `data/` directories are already populated — no preparation is needed to run the smoke test. To add new data or recreate the data directories from a raw dataset, use `prepare_data.py`: it handles renaming, format conversion, and `train.flist` generation. Update `DATASETS_ROOT` at the top of the script to point to your dataset location before running.

---

## 3. Infrastructure Changes & Standardization

The naming convention is standardized across all datasets for clarity and scalability.

1.  **BCI (H&E → IHC):** Local data directory is `data/bci`.
2.  **MIST (Unstained → stained):** Separate folders for all four markers: `data/mist_er`, `data/mist_pr`, `data/mist_ki67`, and `data/mist_her2`.
3.  **Configs:** Dedicated JSON configs for every modality (e.g., `config/mist_er.json`), each pointing to its corresponding `data/mist_*` directory.

### 🚨 SMOKE TEST CONSTRAINTS (MUST REVERT ON HPC)
The following constraints are applied in the `config/*.json` files to fit the **11GB 1080 Ti** VRAM limit:
*   **Batch Size:** Set to `1`.
*   **Resolution:** Resized to `256x256` (Original BCI is 1024x1024).
*   **Inference Steps:** Set to `10` (Original is 2000) for rapid smoke-test verification.
*   **Epochs/Iterations:** Limited to `1 epoch` and `10 iterations` for initial verification.

---

## 4. Execution Guide (Verification Commands)

Run the following commands to verify the training and inference pipeline for each modality:

### BCI (H&E to IHC)
```bash
python run.py -c config/bci.json -p train -gpu 0
python run.py -c config/bci.json -p test -gpu 0
```
> **Note:** The test config reuses `train.flist` — no separate test split was prepared for the smoke test.

### MIST (All Modalities)
```bash
# Estrogen Receptor (ER)
python run.py -c config/mist_er.json -p train -gpu 0

# Progesterone Receptor (PR)
python run.py -c config/mist_pr.json -p train -gpu 0

# Ki67
python run.py -c config/mist_ki67.json -p train -gpu 0

# HER2
python run.py -c config/mist_her2.json -p train -gpu 0
```

---

## 5. Infrastructure Fixes

Permitted fixes applied to make the repository run correctly. These are not architectural changes — they do not alter the model, training logic, or diffusion process.

| File | Fix | Reason |
| :--- | :--- | :--- |
| `core/logger.py` | Replaced chained DataFrame assignment with `.loc[key, col]` in `LogTracker.update()` | Chained assignment (`df[col][key]`) raises `FutureWarning` in pandas and will break silently in pandas 3.0 |
| `requirements_frozen.txt` | Pinned `numpy==1.26.4`, `opencv-python==4.8.1.78`, removed `torchaudio` | PyTorch 2.1.2 is compiled against the NumPy 1.x C ABI — NumPy 2.x breaks tensor `.numpy()` calls at runtime. `torchaudio` is unused. |