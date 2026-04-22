# Palette Diffusion Model: Virtual Staining Benchmark Documentation

This document details the standardization, environment setup, and data routing performed for the **Palette-Image-to-Image-Diffusion-Models** repository. This staging phase confirms that the repository reproduces correctly on a Linux environment (WSL Ubuntu 24.04) mimicking an HPC cluster.

---

## 1. Environment Specification
The environment was built to match the target High-Performance Computing (HPC) cluster requirements while respecting the constraints of the local staging hardware (GTX 1080 Ti).

*   **Operating System:** Linux (WSL Ubuntu 24.04)
*   **Hardware:** 128GB RAM | 1x NVIDIA GTX 1080 Ti (11GB VRAM)
*   **Python Version:** 3.9.25
*   **PyTorch Lock:** Version 2.1.2 (CUDA 12.1)
*   **Critical Dependencies & Compatibility Fixes:** 
    *   `numpy<2.0`: Required because PyTorch 2.1.2 is incompatible with NumPy 2.x.
    *   `opencv-python<4.9`: Required to maintain compatibility with the NumPy 1.x requirement.

### Setup Commands:
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "numpy<2" "opencv-python<4.9"
```

---

## 2. Data Handling & Adaptation Rationale

### Repository Expectations (Original Design)
The Palette repository utilizes a `ColorizationDataset` loader class (found in `data/dataset.py`) designed with rigid expectations:
1.  **Folder Structure:** It expects a root directory containing `gray/` (Condition/Input) and `color/` (Ground Truth/Target) sub-folders.
2.  **Naming Convention:** It dynamically generates filenames using a 5-digit zero-padded integer format: `str(index).zfill(5) + '.png'` (e.g., `00001.png`).
3.  **Index Mapping:** It requires a `.flist` file (a simple text list of integers) to define which indices to load.

### Our Adaptation Strategy
To adhere to the **Golden Rule** (no architectural or code changes), we transformed the raw BCI and MIST datasets to fit the repository's native infrastructure:
*   **Routing:** Mapped H&E/Unstained images to the `gray/` folder and IHC images to the `color/` folder.
*   **Renaming:** Renamed complex original filenames (e.g., `10M2102916_10_15.jpg`) to the mandatory `00001.png` format.
*   **Format Normalization:** Renamed `.jpg` files to `.png` to satisfy the code's string formatting, relying on the `PIL` library to handle the actual JPEG format during loading.
*   **Indexing:** Generated custom `train.flist` files for each modality containing the sequence `1` to `10`.

### Why We Renamed Instead of Coding
*   **Code Integrity:** Modifying the data instead of the code ensures we are testing the authors' implementation as published.
*   **Zero Architectural Change:** Changing the loader to accept original filenames would constitute an infrastructure modification, violating the benchmark's "Golden Rule."
*   **Pairing Accuracy:** Renaming source and target to the *same* 5-digit ID is the most reliable way to ensure the model always sees correctly paired H&E and IHC images.

---

## 3. Infrastructure Changes & Standardization

We standardized the naming convention across all datasets to ensure the benchmark is clean and scalable for the HPC cluster.

1.  **BCI (H&E → IHC):** Standardized from `bci_smoke` to `data/bci`.
2.  **MIST (Unstained → IHC):** Created separate folders for all four markers: `data/mist_er`, `data/mist_pr`, `data/mist_ki67`, and `data/mist_her2`.
3.  **Configs:** Generated dedicated JSON configs for every modality (e.g., `config/mist_er.json`) by cloning the smoke test template and updating the paths.

### 🚨 PHASE 1 HACKS (MUST REVERT ON HPC)
The following constraints were applied in the `config/*.json` files to fit the **11GB 1080 Ti** VRAM limit:
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

## 5. Summary of Actions & Reasons

| Action | Reason |
| :--- | :--- |
| **Used `ColorizationDataset`** | The repo treats Virtual Staining as a "colorization" task; this class is the native infrastructure for paired loading. |
| **Renamed Images to `00001.png`** | To satisfy the hard-coded `zfill(5)` requirement in the authors' data loader without editing their code. |
| **Downgraded NumPy/OpenCV** | To resolve compatibility conflicts between PyTorch 2.1.2 and the modern NumPy 2.x environment. |
| **Separated Folders (`gray`/`color`)** | The repo does not use stitched (side-by-side) images; it expects parallel folders for source and target. |
| **Created Modality-Specific Configs** | To ensure every MIST stain (ER, PR, Ki67, HER2) is individually verified for data-loading stability. |
