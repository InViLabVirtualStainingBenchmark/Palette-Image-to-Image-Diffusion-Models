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

### The `VirtualStainingDataset` Loader
The original repository relied on a rigid `ColorizationDataset` class that forced specific folder names (`gray`/`color`), hard-coded `.png` extensions, required numbered filenames (e.g., `00001.png`), and relied on `.flist` text indices. This necessitated duplicating and renaming the source dataset.

To eliminate data duplication and align with the other benchmark projects, a custom `VirtualStainingDataset` class was added to `data/dataset.py`.

### Advantages of the New Loader:
*   **No Duplication:** It reads directly from the absolute paths of the source datasets.
*   **No Renaming:** It aligns input (Condition) and target (Ground Truth) files purely by alphabetical sorting, allowing original source filenames (like `10M2102916.jpg`) to remain intact.
*   **Extension Agnostic:** It loads standard image formats natively without forced `.png` conversion.
*   **No `.flist` Dependencies:** It automatically calculates the dataset length based on the actual files present in the directories.

**Data preparation:** No preparation is needed. All `.json` configuration files have been updated to point directly to `/home/vs_user/Virtual Staining/Datasets` using the `VirtualStainingDataset` class. The legacy `prepare_data.py` script has been removed.

---

## 3. Infrastructure Changes & Standardization

All configuration JSON files (`config/bci.json`, `config/mist_er.json`, etc.) now point directly to the source.

1.  **BCI (H&E → IHC):** 
    * Train: `/Datasets/BCI/HE/train` → `/Datasets/BCI/IHC/train`
    * Test: `/Datasets/BCI/HE/test` → `/Datasets/BCI/IHC/test`
2.  **MIST (Unstained → stained):**
    * Train: `/Datasets/MIST/[Marker]/TrainValAB/trainA` → `.../trainB`
    * Val/Test: `/Datasets/MIST/[Marker]/TrainValAB/valA` → `.../valB`

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
| `data/dataset.py` | Added `VirtualStainingDataset` class | Allowed direct reading from source datasets without data duplication or renaming. |
| `core/logger.py` | Replaced chained DataFrame assignment with `.loc[key, col]` in `LogTracker.update()` | Chained assignment (`df[col][key]`) raises `FutureWarning` in pandas and will break silently in pandas 3.0 |
| `requirements_frozen.txt` | Pinned `numpy==1.26.4`, `opencv-python==4.8.1.78`, removed `torchaudio` | PyTorch 2.1.2 is compiled against the NumPy 1.x C ABI — NumPy 2.x breaks tensor `.numpy()` calls at runtime. `torchaudio` is unused. |
