# Palette Virtual Staining — HPC Deployment Guide

Complete step-by-step guide for deploying the Palette virtual staining pipeline on the VSC CalcUA cluster. Covers first-time setup, data preparation, container builds, training, resuming, inference, and evaluation.

---

## 0. Cluster Reference

| Item | Value |
| :--- | :--- |
| Login node | `login.hpc.uantwerpen.be` |
| Username | `vsc21213` *(replace with your VSC username)* |
| Account | `ap_invilab_td_thesis` |
| NVIDIA partition | `ampere_gpu` (A100) |
| AMD partition | `arcturus_gpu` (MI100) |
| `$VSC_DATA` | `/data/antwerpen/212/vsc21213` |
| `$VSC_SCRATCH` | `/scratch/antwerpen/212/vsc21213` |
| Group scratch | `/scratch/antwerpen/grp/ap_invilab_td_thesis` |

### Key paths on the cluster

| What | Path |
| :--- | :--- |
| Palette code | `$VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models` |
| Evaluate repo | `$VSC_DATA/evaluate` |
| NVIDIA training container | `$VSC_SCRATCH/containers/palette_hpc.sif` |
| AMD training container | `$VSC_SCRATCH/containers/palette_amd.sif` |
| Evaluate container (AMD) | `$VSC_SCRATCH/containers/evaluate_amd.sif` |
| BCI dataset archive | `$VSC_SCRATCH/datasets/BCI_dataset.sqsh` |
| MIST dataset archive | `$VSC_SCRATCH/datasets/MIST_dataset.sqsh` |
| Training outputs | `$VSC_DATA/projects/palette/outputs/experiments` |
| Logs | `$VSC_DATA/projects/palette/logs` |
| BCI predictions | `/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/BCI` |
| MIST predictions | `/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_<STAIN>` |
| Eval results CSV | `$VSC_DATA/projects/palette/outputs/eval_results.csv` |

---

## 1. One-Time Setup (New Developer)

### 1.1 SSH into the cluster

```bash
ssh vsc21213@login.hpc.uantwerpen.be
```

### 1.2 Create the directory structure

Run once on the login node. These directories are expected by all SLURM scripts.

```bash
mkdir -p $VSC_DATA/projects/palette/code
mkdir -p $VSC_DATA/projects/palette/logs
mkdir -p $VSC_DATA/projects/palette/outputs/experiments
mkdir -p $VSC_DATA/projects/palette/outputs/plots
mkdir -p $VSC_SCRATCH/containers
mkdir -p $VSC_SCRATCH/datasets
mkdir -p /scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/BCI
mkdir -p /scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_ER
mkdir -p /scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_PR
mkdir -p /scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_HER2
mkdir -p /scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_KI67
```

### 1.3 Clone the repositories

**Palette model repo:**
```bash
cd $VSC_DATA/projects/palette/code
git clone <palette-repo-url> Palette-Image-to-Image-Diffusion-Models
```

**Evaluate repo** (shared benchmark infrastructure):
```bash
git clone <evaluate-repo-url> $VSC_DATA/evaluate
```

### 1.4 Adapt paths for your username

The SLURM `#SBATCH -o` and `#SBATCH -e` log directives cannot use environment variables — they are static strings evaluated by the scheduler before the job runs. Every SLURM script in `slurm/` contains hardcoded paths with `vsc21213`. A new developer must replace these with their own username.

Run this once from inside the repo to replace all occurrences in one go:

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models

# Replace username in all SLURM scripts
grep -rl "vsc21213" slurm/ | xargs sed -i 's|vsc21213|YOUR_VSC_USERNAME|g'

# Replace username in all config JSON files
grep -rl "vsc21213" config/ | xargs sed -i 's|vsc21213|YOUR_VSC_USERNAME|g'
```

> **Note:** The number `212` in `/data/antwerpen/212/` and `/scratch/antwerpen/212/` is the user group prefix derived from your VSC number. For `vsc21213` it is `212`. For a different user it may differ. Check your actual `$VSC_DATA` path after login and adjust accordingly if the prefix differs.

---

## 2. Data Preparation

### 2.1 Required folder structure

The datasets must be structured exactly as follows before squashing. The scripts mount the squashfs archive and access files at these relative paths.

**BCI dataset:**
```
BCI/
├── HE/
│   ├── train/     ← H&E training images
│   └── test/      ← H&E test images
└── IHC/
    ├── train/     ← IHC training images (ground truth)
    └── test/      ← IHC test images (ground truth)
```

**MIST dataset:**
```
MIST/
├── ER/
│   └── TrainValAB/
│       ├── trainA/    ← H&E training images
│       ├── trainB/    ← IHC training images
│       ├── valA/      ← H&E test images  (official test set)
│       └── valB/      ← IHC test images  (official test set)
├── HER2/
│   └── TrainValAB/ ...
├── Ki67/              ← NOTE: capital K, lowercase i — this exact casing is required
│   └── TrainValAB/ ...
└── PR/
    └── TrainValAB/ ...
```

> **Important:** The MIST `valA`/`valB` folders are the **official test set** despite being named "val". All inference and evaluation uses these folders, not a separate test folder.

> **Important:** The Ki67 folder must be named `Ki67` exactly (not `KI67` or `ki67`). The SLURM scripts handle this case mapping automatically when you pass `STAIN=KI67`.

### 2.2 Create SquashFS archives

Run on your local Linux machine (WSL2 or native Linux). `mksquashfs` must be installed (`sudo apt install squashfs-tools`).

```bash
# Create archives — adjust source paths to where your datasets live
mksquashfs /path/to/BCI   BCI_dataset.sqsh  -comp lz4 -noI -noX
mksquashfs /path/to/MIST  MIST_dataset.sqsh -comp lz4 -noI -noX
```

> `-comp lz4 -noI -noX` uses fast compression. The resulting files will be large (several GB each) but mount very quickly on the cluster, which matters for I/O-heavy training.

### 2.3 Upload to cluster scratch

Use `rsync` rather than `scp` — it supports resuming if the upload is interrupted.

```bash
rsync -avz --progress BCI_dataset.sqsh \
    vsc21213@login.hpc.uantwerpen.be:$VSC_SCRATCH/datasets/BCI_dataset.sqsh

rsync -avz --progress MIST_dataset.sqsh \
    vsc21213@login.hpc.uantwerpen.be:$VSC_SCRATCH/datasets/MIST_dataset.sqsh
```

### 2.4 Verify the archives on the cluster

```bash
# Check archive metadata (fast — does not decompress)
unsquashfs -s $VSC_SCRATCH/datasets/BCI_dataset.sqsh
unsquashfs -s $VSC_SCRATCH/datasets/MIST_dataset.sqsh

# List top-level contents to confirm folder structure
unsquashfs -l $VSC_SCRATCH/datasets/BCI_dataset.sqsh  | head -20
unsquashfs -l $VSC_SCRATCH/datasets/MIST_dataset.sqsh | head -20
```

---

## 3. Container Build and Upload

All containers are built locally on a Linux machine with Apptainer installed (`sudo apt install apptainer` or via the official PPA). The `.sif` files are never committed to git — they are built locally and uploaded to `$VSC_SCRATCH/containers/`.

> **Note:** Container builds pull large base images and can take 15–30 minutes each.

### 3.1 NVIDIA training container (`palette_hpc.sif`)

Built from `palette_nvidia.def` in the project root. Used for training and inference on the `ampere_gpu` (A100) partition.

```bash
cd /path/to/Palette-Image-to-Image-Diffusion-Models
apptainer build --fakeroot palette_hpc.sif palette_nvidia.def
```

Upload to the cluster:
```bash
rsync -avz --progress palette_hpc.sif \
    vsc21213@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/palette_hpc.sif
```

Verify on the cluster (login node):
```bash
module purge && module load calcua/2025a
apptainer exec $VSC_SCRATCH/containers/palette_hpc.sif python -c "
import torch; print('torch:', torch.__version__)
import torchvision; print('torchvision:', torchvision.__version__)
print('All imports OK')
"
```

### 3.2 AMD training container (`palette_amd.sif`)

Built from `palette_amd.def` in the project root. Used for training and inference on the `arcturus_gpu` (MI100) partition.

```bash
cd /path/to/Palette-Image-to-Image-Diffusion-Models
apptainer build --fakeroot palette_amd.sif palette_amd.def
```

Upload:
```bash
rsync -avz --progress palette_amd.sif \
    vsc21213@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/palette_amd.sif
```

> **Note:** `bci_amd.def` in the project root is an abandoned earlier attempt that does not work correctly on BCI (MIOpen kernel tuning runs out of VRAM). Do not use it. The correct AMD container definition is `palette_amd.def`.

### 3.3 Evaluate container (`evaluate_amd.sif`)

The evaluation scripts use a **separate shared container** that lives in the `evaluate` repository (not in the Palette repo). This container holds all evaluation dependencies: `torchmetrics`, `lpips`, `torch-fidelity`, `cellpose`.

The container definition is at `evaluate/hpc_jobs/evaluate_nvidia.def`. Build it from the evaluate repo directory:

```bash
cd /path/to/evaluate/hpc_jobs
apptainer build --fakeroot evaluate_amd.sif evaluate_nvidia.def
```

Upload:
```bash
rsync -avz --progress evaluate_amd.sif \
    vsc21213@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/evaluate_amd.sif
```

#### Pre-download LPIPS and Cellpose weights on the login node

**This step is mandatory.** Compute nodes have no internet access. LPIPS and Cellpose both download model weights on first use. If the weights are not cached before the eval job runs, it will fail silently or produce wrong metrics.

SSH into the cluster and run the following **on the login node** (not via sbatch):

```bash
module purge && module load calcua/2025a

# Pre-download LPIPS weights (AlexNet and VGG backbones)
apptainer exec --nv $VSC_SCRATCH/containers/evaluate_amd.sif python -c "
import lpips
lpips.LPIPS(net='alex')
lpips.LPIPS(net='vgg')
print('LPIPS weights cached.')
"

# Pre-download Cellpose weights
apptainer exec --nv $VSC_SCRATCH/containers/evaluate_amd.sif python -c "
from cellpose import models
models.CellposeModel(pretrained_model='cyto2')
print('Cellpose weights cached.')
"
```

Verify the caches exist before submitting any eval job:
```bash
ls ~/.cache/torch/hub/checkpoints/   # should contain alexnet-*.pth and vgg16-*.pth
ls ~/.cellpose/models/               # should contain cyto2
```

---

## 4. Training

All training jobs use `n_iter: 100000` as the stopping condition (set in the config JSON). Training outputs land in `$VSC_DATA/projects/palette/outputs/experiments/` in a folder named `train_<config_name>_<SLURM_JOB_ID>`.

**Before submitting:** make sure `resume_state` is `null` in the config JSON for a fresh run:
```json
"resume_state": null
```

### 4.1 NVIDIA — BCI

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models
sbatch slurm/train/nvidia/train_bci_randomcrop.slurm
```

### 4.2 NVIDIA — MIST (one job per stain)

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models

sbatch --job-name=palette_mist_randomcrop_er   --export=ALL,STAIN=ER   slurm/train/nvidia/train_mist_randomcrop.slurm
sbatch --job-name=palette_mist_randomcrop_pr   --export=ALL,STAIN=PR   slurm/train/nvidia/train_mist_randomcrop.slurm
sbatch --job-name=palette_mist_randomcrop_her2 --export=ALL,STAIN=HER2 slurm/train/nvidia/train_mist_randomcrop.slurm
sbatch --job-name=palette_mist_randomcrop_ki67 --export=ALL,STAIN=KI67 slurm/train/nvidia/train_mist_randomcrop.slurm
```

### 4.3 AMD — BCI

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models
sbatch slurm/train/amd/train_bci_randomcrop.slurm
```

### 4.4 AMD — MIST (one job per stain)

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models

sbatch --job-name=palette_mist_randomcrop_amd_er   --export=ALL,STAIN=ER   slurm/train/amd/train_mist_randomcrop.slurm
sbatch --job-name=palette_mist_randomcrop_amd_pr   --export=ALL,STAIN=PR   slurm/train/amd/train_mist_randomcrop.slurm
sbatch --job-name=palette_mist_randomcrop_amd_her2 --export=ALL,STAIN=HER2 slurm/train/amd/train_mist_randomcrop.slurm
sbatch --job-name=palette_mist_randomcrop_amd_ki67 --export=ALL,STAIN=KI67 slurm/train/amd/train_mist_randomcrop.slurm
```

### 4.5 Monitoring training

```bash
# Check running jobs
squeue -u vsc21213

# Follow live log output (replace JOBID)
tail -f $VSC_DATA/projects/palette/logs/train_bci_randomcrop.<JOBID>.out

# Check GPU utilization CSV logged by the script
cat $VSC_DATA/projects/palette/logs/gpu_bci_randomcrop_<JOBID>.csv
```

The log will print iteration progress periodically (every `log_iter: 10000` iterations):
```
INFO: iters: 10000   mse_loss: 0.0423
INFO: iters: 20000   mse_loss: 0.0381
...
```

---

## 5. Resuming Training After a Timeout

If a job hits the SLURM time limit before reaching 100k iterations, the `last` checkpoint is saved automatically at the point of interruption and training can be resumed.

### 5.1 Find the experiment folder name

```bash
ls -lt $VSC_DATA/projects/palette/outputs/experiments/ | head -5
```

The folder is named `train_bci_randomcrop_<JOBID>` for BCI, or `train_mist_er_randomcrop_<JOBID>` for MIST ER, etc. Note the full name.

### 5.2 Confirm the last checkpoint exists

```bash
ls $VSC_DATA/projects/palette/outputs/experiments/train_bci_randomcrop_<JOBID>/checkpoint/
# Expected: best_Network.pth  best_Network_ema.pth  best.state  last_Network.pth  last_Network_ema.pth  last.state
```

### 5.3 Set `resume_state` in the config JSON

Open the relevant config file (e.g., `config/bci_randomcrop.json`) and set the `resume_state` field to the **prefix path** of the last checkpoint — without any file extension. The training code appends `_Network.pth` and `.state` automatically.

```json
"resume_state": "/data/antwerpen/212/vsc21213/projects/palette/outputs/experiments/train_bci_randomcrop_<JOBID>/checkpoint/last"
```

For MIST, the config files are `config/mist_er_randomcrop.json`, `config/mist_pr_randomcrop.json`, etc.

> **Do not** set `test_checkpoint` here — that field is only for inference. `resume_state` and `test_checkpoint` are completely separate.

### 5.4 Re-submit the training job

```bash
sbatch slurm/train/nvidia/train_bci_randomcrop.slurm
# or AMD:
sbatch slurm/train/amd/train_bci_randomcrop.slurm
```

Training picks up from the last saved iteration. The new job creates a new experiment folder (with the new SLURM_JOB_ID), but loads weights and optimizer state from the previous run.

### 5.5 Reset `resume_state` after training completes

Once training is finished, set `resume_state` back to `null` in the config so a future accidental re-run does not load stale weights:

```json
"resume_state": null
```

---

## 6. Inference (Tiling)

The randomcrop models require tiling inference (`tile_inference.py`) at test time. The script splits each 1024×1024 H&E image into 16 non-overlapping 256×256 tiles, runs the diffusion sampling chain on each tile, and stitches the predictions back into a 1024×1024 IHC image.

Predictions are saved directly into the group scratch prediction folders.

### 6.1 Pre-requisite: set `test_checkpoint` in the config JSON

This must be done **before submitting any inference job**. Open the config for the model you want to run inference with and fill in the `test_checkpoint` field with the full path to the best EMA checkpoint.

**How to find the path:**
```bash
# Find the experiment folder from training
ls -lt $VSC_DATA/projects/palette/outputs/experiments/ | grep train_bci_randomcrop | head -3

# Confirm the best checkpoint exists
ls $VSC_DATA/projects/palette/outputs/experiments/train_bci_randomcrop_<JOBID>/checkpoint/best_Network_ema.pth
```

**Set it in `config/bci_randomcrop.json`:**
```json
"test_checkpoint": "/data/antwerpen/212/vsc21213/projects/palette/outputs/experiments/train_bci_randomcrop_<JOBID>/checkpoint/best_Network_ema.pth"
```

**For MIST**, set the same field in each stain's config. The experiment folder names follow the pattern `train_mist_<stain>_randomcrop_<JOBID>`:

| Config file | `test_checkpoint` experiment folder pattern |
| :--- | :--- |
| `config/mist_er_randomcrop.json` | `train_mist_er_randomcrop_<JOBID>` |
| `config/mist_pr_randomcrop.json` | `train_mist_pr_randomcrop_<JOBID>` |
| `config/mist_her2_randomcrop.json` | `train_mist_her2_randomcrop_<JOBID>` |
| `config/mist_ki67_randomcrop.json` | `train_mist_ki67_randomcrop_<JOBID>` |

Always use `best_Network_ema.pth` — not `best_Network.pth` or `last_Network_ema.pth`.

### 6.2 AMD — BCI inference

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models
sbatch slurm/test/amd/test_bci_randomcrop.slurm
```

The checkpoint path is read from `config/bci_randomcrop.json → path.test_checkpoint`. No additional arguments needed at submit time.

Predictions are saved to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/BCI/
```

### 6.3 AMD — MIST inference (one job per stain)

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models

sbatch --job-name=palette_mist_infer_amd_er   --export=ALL,STAIN=ER   slurm/test/amd/test_mist_randomcrop.slurm
sbatch --job-name=palette_mist_infer_amd_pr   --export=ALL,STAIN=PR   slurm/test/amd/test_mist_randomcrop.slurm
sbatch --job-name=palette_mist_infer_amd_her2 --export=ALL,STAIN=HER2 slurm/test/amd/test_mist_randomcrop.slurm
sbatch --job-name=palette_mist_infer_amd_ki67 --export=ALL,STAIN=KI67 slurm/test/amd/test_mist_randomcrop.slurm
```

Each script reads its checkpoint from the corresponding `config/mist_<stain>_randomcrop.json → path.test_checkpoint`.

Predictions are saved to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_ER/
/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_PR/
/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_HER2/
/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_KI67/
```

### 6.4 NVIDIA — BCI inference

The NVIDIA inference scripts require the checkpoint to be passed as the `BEST_CKPT` environment variable at submit time (instead of reading from the config).

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models

sbatch --export=ALL,BEST_CKPT="$VSC_DATA/projects/palette/outputs/experiments/train_bci_randomcrop_<JOBID>/checkpoint/best_Network_ema.pth" \
    slurm/test/nvidia/test_bci_randomcrop.slurm
```

### 6.5 NVIDIA — MIST inference (one job per stain)

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models

sbatch --job-name=palette_mist_infer_er \
       --export=ALL,STAIN=ER,BEST_CKPT="$VSC_DATA/projects/palette/outputs/experiments/train_mist_er_randomcrop_<JOBID>/checkpoint/best_Network_ema.pth" \
       slurm/test/nvidia/test_mist_randomcrop.slurm

sbatch --job-name=palette_mist_infer_pr \
       --export=ALL,STAIN=PR,BEST_CKPT="$VSC_DATA/projects/palette/outputs/experiments/train_mist_pr_randomcrop_<JOBID>/checkpoint/best_Network_ema.pth" \
       slurm/test/nvidia/test_mist_randomcrop.slurm

sbatch --job-name=palette_mist_infer_her2 \
       --export=ALL,STAIN=HER2,BEST_CKPT="$VSC_DATA/projects/palette/outputs/experiments/train_mist_her2_randomcrop_<JOBID>/checkpoint/best_Network_ema.pth" \
       slurm/test/nvidia/test_mist_randomcrop.slurm

sbatch --job-name=palette_mist_infer_ki67 \
       --export=ALL,STAIN=KI67,BEST_CKPT="$VSC_DATA/projects/palette/outputs/experiments/train_mist_ki67_randomcrop_<JOBID>/checkpoint/best_Network_ema.pth" \
       slurm/test/nvidia/test_mist_randomcrop.slurm
```

---

## 7. Evaluation

Evaluation runs `evaluate.py` from the shared `evaluate` repo against the prediction folders and ground truth images from the squashfs archive. Results are appended to the shared CSV file.

**Before submitting any eval job**, confirm the LPIPS and Cellpose weights are pre-downloaded (Section 3.3). Missing weights will cause silent metric errors on the compute node.

### 7.1 AMD — BCI evaluation

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models
sbatch slurm/eval/amd/eval_bci_randomcrop.slurm
```

Results appended to: `$VSC_DATA/projects/palette/outputs/eval_results.csv`

### 7.2 AMD — MIST evaluation (one job per stain)

```bash
sbatch --job-name=palette_mist_eval_amd_er   --export=ALL,STAIN=ER   slurm/eval/amd/eval_mist_randomcrop.slurm
sbatch --job-name=palette_mist_eval_amd_pr   --export=ALL,STAIN=PR   slurm/eval/amd/eval_mist_randomcrop.slurm
sbatch --job-name=palette_mist_eval_amd_her2 --export=ALL,STAIN=HER2 slurm/eval/amd/eval_mist_randomcrop.slurm
sbatch --job-name=palette_mist_eval_amd_ki67 --export=ALL,STAIN=KI67 slurm/eval/amd/eval_mist_randomcrop.slurm
```

### 7.3 NVIDIA — BCI evaluation

```bash
sbatch slurm/eval/nvidia/eval_bci_randomcrop.slurm
```

### 7.4 NVIDIA — MIST evaluation (one job per stain)

```bash
sbatch --job-name=palette_mist_eval_er   --export=ALL,STAIN=ER   slurm/eval/nvidia/eval_mist_randomcrop.slurm
sbatch --job-name=palette_mist_eval_pr   --export=ALL,STAIN=PR   slurm/eval/nvidia/eval_mist_randomcrop.slurm
sbatch --job-name=palette_mist_eval_her2 --export=ALL,STAIN=HER2 slurm/eval/nvidia/eval_mist_randomcrop.slurm
sbatch --job-name=palette_mist_eval_ki67 --export=ALL,STAIN=KI67 slurm/eval/nvidia/eval_mist_randomcrop.slurm
```

---

## 8. Plotting Training Curves

Training curves are generated locally using `plot_training_curves_diffusion_models.py` at the root of the repo. It auto-detects the Palette log format and produces a PNG plot and a summary CSV.

**Prerequisites:** `numpy`, `matplotlib` (install locally with `pip install numpy matplotlib`).

### 8.1 Download the SLURM log

After training completes, copy the `.out` log from the cluster:

```bash
scp vsc21213@login.hpc.uantwerpen.be:/data/antwerpen/212/vsc21213/projects/palette/logs/<job>.out ./logs/
```

### 8.2 Plot a single model

```bash
python plot_training_curves_diffusion_models.py \
    --logs  logs/train_bci_randomcrop_12345.out \
    --labels  Palette-BCI \
    --name  bci \
    --out-dir  results/training_curves/palette
```

Outputs written to `results/training_curves/palette/`:
- `bci_training_curves.png` — all panels (loss, val MAE, LR, epoch time)
- `bci_training_summary.csv` — final averages and best checkpoint positions

### 8.3 Plot multiple runs on the same axes (e.g. all MIST stains)

```bash
python plot_training_curves_diffusion_models.py \
    --logs  logs/mist_er.out logs/mist_pr.out logs/mist_her2.out logs/mist_ki67.out \
    --labels  Palette-ER Palette-PR Palette-HER2 Palette-Ki67 \
    --name  mist_all \
    --out-dir  results/training_curves/palette
```

### 8.4 Key options

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--smooth N` | 5 | Moving-average window for training loss curves |
| `--dpi N` | 150 | Output image resolution |
| `--last-n N` | 10 | Number of final epochs averaged in the summary CSV |

---

## 9. What to Change If You Are a New Developer

### 8.1 Username replacement

The username `vsc21213` and the path prefix `/data/antwerpen/212/` appear in multiple places. The bulk replacement command in Section 1.4 handles all of them at once. Below is a reference of every file affected, for manual verification.

**SLURM log headers (hardcoded — cannot use `$VSC_DATA`):**

| Script | Lines to change |
| :--- | :--- |
| `slurm/train/nvidia/train_bci_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/train/nvidia/train_mist_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/train/amd/train_bci_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/train/amd/train_mist_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/test/nvidia/test_bci_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/test/nvidia/test_mist_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/test/amd/test_bci_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/test/amd/test_mist_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/eval/nvidia/eval_bci_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/eval/nvidia/eval_mist_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/eval/amd/eval_bci_randomcrop.slurm` | `#SBATCH -o` and `-e` |
| `slurm/eval/amd/eval_mist_randomcrop.slurm` | `#SBATCH -o` and `-e` |

**Config JSON files (hardcoded data and output paths):**

| File | Fields to update |
| :--- | :--- |
| `config/bci_randomcrop.json` | `path.base_dir`, `datasets.train.cond_dir`, `datasets.train.target_dir`, `datasets.test.cond_dir`, `datasets.test.target_dir`, `path.test_checkpoint` |
| `config/bci_resize.json` | Same fields (except `test_checkpoint`) |
| `config/mist_er_randomcrop.json` | Same as above |
| `config/mist_pr_randomcrop.json` | Same |
| `config/mist_her2_randomcrop.json` | Same |
| `config/mist_ki67_randomcrop.json` | Same |
| `config/mist_er_resize.json` | Same (except `test_checkpoint`) |
| `config/mist_pr_resize.json` | Same |
| `config/mist_her2_resize.json` | Same |
| `config/mist_ki67_resize.json` | Same |

### 8.2 Paths that resolve automatically (no change needed)

The following paths use `$VSC_DATA`, `$VSC_SCRATCH`, or `$VSC_REPO` environment variables that are set inside the SLURM script body. These resolve correctly for any user as long as the directory structure from Section 1.2 exists:

| Variable | Resolves to |
| :--- | :--- |
| `$VSC_DATA` | Set by the cluster for your account automatically |
| `$VSC_SCRATCH` | Set by the cluster for your account automatically |
| `$VSC_REPO` | Derived from `$VSC_DATA` inside each script |
| `$CONTAINER` | Derived from `$VSC_SCRATCH` inside each script |
| `$DATA_SQSH` | Derived from `$VSC_SCRATCH` inside each script |
| `$PRED_DIR` | Points to group scratch — same for all users |

### 8.3 Paths that are safe to change without breaking scripts

| What | Where | Notes |
| :--- | :--- | :--- |
| `SLURM_JOB_ID` in experiment folder name | `core/praser.py` | Auto-generated, never hardcoded |
| Container names (`.sif` filenames) | All SLURM scripts via `$CONTAINER` | Change the variable at the top of the script |
| Prediction output folders | `$PRED_DIR` in test SLURM scripts | Change the variable at the top; the folder must exist first |
| Output CSV path | `$OUTPUT_CSV` in eval SLURM scripts | Change the variable at the top |
| `n_iter` training duration | Config JSON `train.n_iter` | Change to any value; script does not depend on it |

---

## 10. Updating Code and Rebuilding

### Pull latest code on the cluster

```bash
cd $VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models
git pull
```

Code changes (configs, Python scripts, SLURM scripts) do **not** require rebuilding any container.

### Rebuild NVIDIA training container

Only needed when `palette_nvidia.def` changes (new Python dependency, updated base image):

```bash
cd /path/to/Palette-Image-to-Image-Diffusion-Models
apptainer build --fakeroot palette_hpc.sif palette_nvidia.def

rsync -avz --progress palette_hpc.sif \
    vsc21213@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/palette_hpc.sif
```

### Rebuild AMD training container

```bash
cd /path/to/Palette-Image-to-Image-Diffusion-Models
apptainer build --fakeroot palette_amd.sif palette_amd.def

rsync -avz --progress palette_amd.sif \
    vsc21213@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/palette_amd.sif
```

### Rebuild evaluate container

```bash
cd /path/to/evaluate/hpc_jobs
apptainer build --fakeroot evaluate_amd.sif evaluate_nvidia.def

rsync -avz --progress evaluate_amd.sif \
    vsc21213@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/evaluate_amd.sif
```

After rebuilding the evaluate container, re-run the LPIPS and Cellpose weight pre-download step (Section 3.3) because the cache inside the container is reset.
