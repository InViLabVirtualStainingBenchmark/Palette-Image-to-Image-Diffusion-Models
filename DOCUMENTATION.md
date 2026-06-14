# Palette Virtual Staining: The Comprehensive Master Guide

This document is the definitive resource for the Palette Virtual Staining adaptation. It combines the **technical rationale** (the "why") with **code-level evidence** (the "what changed") to give the next developer a complete picture.

---

This repository is a specialized adaptation of the **Palette** conditional diffusion model, originally designed for general-purpose image restoration tasks (colorization, inpainting, uncropping). It has been re-engineered for **Virtual Staining** — translating H&E pathology slides into digital IHC stains — and optimized for deployment on the **VSC CalcUA HPC cluster**.

**Datasets:**
- **BCI** — H&E → IHC (breast cancer, 1024×1024, ~3504 train / 977 test images)
- **MIST** — H&E → IHC for four stain markers: ER, PR, HER2, Ki67 (1024×1024, valA/valB are the official test set despite the naming)

**The benchmark constraint:** The Palette diffusion architecture (math, UNet, loss, noise schedule) is treated as a **black box and must not be changed**. All adaptations are at the data loading, infrastructure, and deployment layers only.

---

## 1. Architectural Deep Dive: From General CV to Pathology Virtual Staining

### A. Data Alignment — The Alphabetical Pairing Problem

**The Original State (`ColorizationDataset`):**
The original dataset class required numbered filenames (`00001.png`, `00002.png`), forced `.png` extensions, demanded a specific `gray/` and `color/` subfolder structure, and depended on manually-maintained `.flist` index files.

```python
# Original — brittle, requires renaming thousands of clinical files
file_name = str(self.flist[index]).zfill(5) + '.png'
img        = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray',  file_name)))
```

**The Problem:** Pathology datasets contain thousands of files with clinical naming conventions (e.g., `10M2102916.jpg`). Renaming them or generating `.flist` files manually is error-prone and unsustainable.

**The Solution (`VirtualStainingDataset` in `data/dataset.py`):**

```python
# New — reads any filename, any extension, paired by alphabetical sort
exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
self.img_names = sorted([f for f in os.listdir(cond_dir)   if os.path.splitext(f)[1].lower() in exts])
self.lbl_names = sorted([f for f in os.listdir(target_dir) if os.path.splitext(f)[1].lower() in exts])

# Safety assertion — catches folder mismatches immediately at startup
cond_stems   = [os.path.splitext(f)[0] for f in self.img_names]
target_stems = [os.path.splitext(f)[0] for f in self.lbl_names]
assert cond_stems == target_stems, f"Filename mismatch between cond_dir and target_dir. ..."
```

**Why this works:** Both the H&E (`cond_dir`) and IHC (`target_dir`) folders contain the same filenames. Sorting both lists alphabetically guarantees perfect pair alignment without any index files or renaming.

---

### B. Scale Preservation — The Cell Size Problem

**The Original State:** All datasets were resized to a fixed resolution (224×224 or 256×256) using `transforms.Resize`.

```python
# Original — distorts biological scale
self.tfs = transforms.Compose([
    transforms.Resize((image_size[0], image_size[1])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

**The Problem:** BCI and MIST images are 1024×1024. Resizing to 256×256 shrinks every biological structure by 4×. A cell nucleus that spans 20px at 40× magnification now spans 5px — the model sees a completely different morphology. Training on this distorted data means the model learns the wrong spatial relationships between H&E and IHC features.

**The Solution (Random Crop path in `VirtualStainingDataset`):**

```python
# New — crops a 256x256 window at the original magnification
if self.use_random_crop:
    cond_pil   = self.loader(cond_path)
    target_pil = self.loader(target_path)
    # Same crop coordinates applied to BOTH images — preserves paired alignment
    i, j, h, w = transforms.RandomCrop.get_params(cond_pil, output_size=(self.image_size[0], self.image_size[1]))
    cond_image = self.post_tfs(TF.crop(cond_pil,   i, j, h, w))
    img        = self.post_tfs(TF.crop(target_pil, i, j, h, w))
```

**Why this is correct:** In pathology, magnification (e.g., 40×) is a physical constant. A nucleus always occupies the same number of pixels at the same magnification. By extracting a random 256×256 window from the full 1024×1024 slide, the model trains on tissue at its **true biological scale**, which is what makes the virtual staining clinically meaningful.

**The trade-off:** This introduces two training modes with different inference requirements:

| Mode | Config flag | Training | Inference |
| :--- | :--- | :--- | :--- |
| **resize** | `use_random_crop: false` | Full image resized to 256×256 | `run.py -p test` (single forward pass) |
| **randomcrop** | `use_random_crop: true` | Random 256×256 crop from 1024×1024 | `tile_inference.py` (tiling required) |

---

### C. Task Hijacking — Repurposing Colorization for Virtual Staining

The original Palette supports four tasks: `colorization`, `inpainting`, `uncropping`, and `inpainting_with_mask`. Virtual staining is a **1:1 image-to-image** translation task with no masking or spatial manipulation — which maps perfectly onto the `colorization` task.

**How the task parameter controls the model (`models/model.py`):**

```python
# In val_step / test — colorization task skips mask logic entirely
if self.task in ['inpainting', 'uncropping']:
    self.output, self.visuals = self.netG.restoration(
        self.cond_image, y_t=self.cond_image, y_0=self.gt_image, mask=self.mask, ...
    )
else:
    # colorization path — H&E is passed as the sole conditioning signal
    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
```

**The 6-channel input:** The UNet receives a 6-channel input: 3 channels of the H&E conditioning image concatenated with 3 channels of the noisy target IHC image. The network learns to denoise the IHC given the H&E as a guide. This is set via `task: "colorization"` in all config JSONs — no code change needed.

---

### D. Best-Metric Checkpointing — The "Best vs. Last" Problem

**The Original State:** Checkpoints were saved every N epochs unconditionally, with no awareness of model quality.

```python
# Original — saves blindly on a schedule
if self.epoch % save_epoch == 0:
    self.save_network(epoch_number)
```

**The Problem:** Diffusion models can train for hundreds of thousands of iterations. The model at epoch 450 may have lower validation error than at the final epoch 500 if any overfitting occurs. A blind "save last" strategy risks benchmarking a degraded checkpoint.

**The Solution (`core/base_model.py`):**

```python
# New — saves only when validation MAE improves
val_metric = val_log.get('mae', list(val_log.values())[0])
if val_metric < self.best_metric:
    self.best_metric = val_metric
    self.logger.info('New best metric {:.6f} at epoch {:d}, saving best checkpoint.'.format(val_metric, self.epoch))
    self.save_everything(tag='best')
```

Validation runs every `val_epoch: 5` epochs. The `best` checkpoint is **overwritten** each time MAE improves, so there is always exactly one best checkpoint. A separate `last` checkpoint is saved when training ends, regardless of performance.

---

### E. HPC Disk Optimization — The Code Bloat Problem

**The Original State:** The parser (`core/praser.py`) automatically copied the entire source tree into every experiment folder.

**The Problem:** On a multi-user HPC cluster, running multiple training jobs (e.g., 5 stains × 2 configs = 10 runs) would create 10 identical copies of `core/`, `models/`, `data/`, etc. This quickly hits disk quotas and degrades NFS metadata performance.

**The Solution (`core/praser.py`):**
The code backup loop was removed entirely. Experiments are now uniquely identified by embedding the `SLURM_JOB_ID` in the folder name, which also makes it trivial to correlate a checkpoint with the SLURM log that produced it.

```python
# New — unique experiment folder via SLURM job ID, no source copy
def get_timestamp():
    return os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%y%m%d_%H%M%S')

experiments_root = os.path.join(opt['path']['base_dir'], '{}_{}'.format(opt['name'], get_timestamp()))
os.makedirs(experiments_root, exist_ok=True)
```

A single copy of the config JSON is still saved per experiment for reproducibility.

---

### F. Modernization & Robustness Fixes

These are not architectural changes. They are fixes required to run on current hardware and software versions.

| File | Fix | Reason |
| :--- | :--- | :--- |
| `core/logger.py` | Replaced chained DataFrame assignment with `.loc[key, col]` | Pandas 2.0+ raises `FutureWarning` on chained assignment; Pandas 3.0 will break it silently |
| `models/model.py` | Explicitly passes `device` object to the noise schedule | Prevents "tensor on wrong device" errors during single-GPU initialization on the cluster |
| `requirements_frozen.txt` | Pinned `numpy==1.26.4`, removed `torchaudio` | PyTorch 2.1.2 is compiled against the NumPy 1.x C ABI — NumPy 2.x breaks `.numpy()` calls at runtime. `torchaudio` is unused. |

---

## 3. Whole Slide Inference: Tiling (`tile_inference.py`)

### The Resolution Bridge Problem

The model is trained on 256×256 patches (randomcrop mode) but pathology slides are 1024×1024. Running a 1024×1024 image through the model in a single pass is impossible because:
1. The model architecture was built and trained for 256×256 inputs.
2. The attention resolution (`attn_res: [16]`) assumes a specific spatial scale.

### How Tiling Works

`tile_inference.py` implements a non-overlapping grid inference strategy:

```
1024×1024 input
    │
    ▼
┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │   ← 16 non-overlapping 256×256 tiles
├───┼───┼───┼───┤      (4 rows × 4 cols)
│ 4 │ 5 │ 6 │ 7 │
├───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │
├───┼───┼───┼───┤
│12 │13 │14 │15 │
└───┴───┴───┴───┘
    │
    ▼ (each tile → diffusion sampling → prediction tile)
    │
    ▼
1024×1024 stitched output (no overlap, no blending needed)
```

**Core tiling logic (`tile_inference.py`):**

```python
TILE_SIZE = 256

def tile_and_infer(net, cond_tensor, device):
    _, _, H, W = cond_tensor.shape
    assert H % TILE_SIZE == 0 and W % TILE_SIZE == 0

    output = torch.zeros_like(cond_tensor)
    rows, cols = H // TILE_SIZE, W // TILE_SIZE

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            x0, x1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
            tile = cond_tensor[:, :, y0:y1, x0:x1]
            pred, _ = net.restoration(tile, sample_num=1)
            output[:, :, y0:y1, x0:x1] = pred

    return output
```

### Checkpoint Resolution in `tile_inference.py`

The script resolves the checkpoint in order of priority:
1. `--checkpoint` CLI argument (full path to `.pth` file)
2. `path.test_checkpoint` field in the config JSON

```python
checkpoint = args.checkpoint or opt['path'].get('test_checkpoint')
if not checkpoint:
    raise ValueError("No checkpoint provided. Pass --checkpoint or set path.test_checkpoint in the config.")
```

This keeps the SLURM scripts clean — the developer sets `test_checkpoint` once in the config JSON and the script reads it automatically.

---

## 4. Training & Checkpoint System

### What Gets Saved and When

`save_everything(tag)` is called with tag `'best'` or `'last'`, saving three files each time:

| File | Saved when | Purpose |
| :--- | :--- | :--- |
| `best_Network.pth` | Validation MAE improves (every 5 epochs) — **overwrites** previous best | Raw network weights at best validation point |
| `best_Network_ema.pth` | Same trigger | EMA-averaged weights at best validation point — **use this for inference** |
| `best.state` | Same trigger | Optimizer + scheduler + epoch + iter state — only needed to resume from best |
| `last_Network.pth` | Training ends (`n_iter` reached or job completes) | Raw weights at final iteration |
| `last_Network_ema.pth` | Same trigger | EMA weights at final iteration |
| `last.state` | Same trigger | Optimizer + scheduler state — **use this to resume a timed-out job** |

### EMA Weights: Why They Matter

EMA (Exponential Moving Average) maintains a shadow copy of the network weights that updates as a slow-moving average of all past weights:

```python
# models/model.py
class EMA():
    def update_average(self, old, new):
        return old * self.beta + (1 - self.beta) * new  # beta = 0.9999
```

Config settings: `ema_start: 1`, `ema_iter: 1`, `ema_decay: 0.9999`. EMA starts from iteration 1 and updates every iteration. The high decay (0.9999) means the shadow weights change very slowly — they represent a heavily smoothed history of training, which produces more stable and visually consistent predictions than the raw weights.

**Rule:** Always use `best_Network_ema.pth` for inference. Always use `last.state` to resume a timed-out job.

---

## 5. Architecture Preservation Audit

The following components are mathematically identical to the original Palette paper and were not modified.

| Component | File | Status |
| :--- | :--- | :--- |
| Diffusion forward/reverse process | `models/network.py` | **Untouched** |
| UNet backbone | `models/guided_diffusion_modules/unet.py` | **Untouched** |
| SR3 UNet variant | `models/sr3_modules/unet.py` | **Untouched** |
| Loss functions | `models/loss.py` | **Untouched** |
| Linear & cosine noise schedules | `models/network.py` | **Untouched** |
| EMA update rule | `models/model.py` | **Untouched** |

---

## 6. File Map (Functional View)

| File / Folder | Purpose | Change Category |
| :--- | :--- | :--- |
| `data/dataset.py` | Added `VirtualStainingDataset` — alphabetical pairing, random crop | **Modified** |
| `core/base_model.py` | Best-metric checkpoint saving (MAE-driven) | **Modified** |
| `core/praser.py` | Removed source backup loop; SLURM_JOB_ID-based folder naming | **Modified** |
| `core/logger.py` | pandas `.loc` fix for Pandas 2.0+ | **Modified** |
| `models/model.py` | Device argument fix for noise schedule | **Modified** |
| `models/network.py` | Diffusion math | **Untouched** |
| `models/guided_diffusion_modules/unet.py` | UNet backbone | **Untouched** |
| `models/loss.py` | Loss functions | **Untouched** |
| `tile_inference.py` | Tiling inference for 1024×1024 slides using randomcrop models | **Added** |
| `config/bci_resize.json` | BCI training config — resize mode (256×256, single forward pass) | **Added** |
| `config/bci_randomcrop.json` | BCI training config — randomcrop mode (requires tiling at inference) | **Added** |
| `config/mist_*_resize.json` | MIST configs per stain — resize mode | **Added** |
| `config/mist_*_randomcrop.json` | MIST configs per stain — randomcrop mode | **Added** |
| `slurm/train/nvidia/` | SLURM training scripts for NVIDIA A100 (ampere_gpu) | **Added** |
| `slurm/train/amd/` | SLURM training scripts for AMD MI100 (arcturus_gpu) | **Added** |
| `slurm/test/nvidia/` | SLURM inference scripts for NVIDIA (run.py for resize, tile_inference.py for randomcrop) | **Added** |
| `slurm/test/amd/` | SLURM inference scripts for AMD (tile_inference.py, palette_amd.sif container) | **Added** |
| `slurm/eval/nvidia/` | SLURM evaluation scripts for NVIDIA (evaluate.py) | **Added** |
| `slurm/eval/amd/` | SLURM evaluation scripts for AMD (evaluate_amd.sif container) | **Added** |
| `Benchmark_evaluation/evaluate.py` | Full metric suite: PSNR, SSIM, MS-SSIM, LPIPS, MAE, FID | **Provided** (not modified) |
| `requirements_frozen.txt` | Pinned dependencies for reproducibility | **Modified** |

---

## 7. Current Status (June 2026)

### Training

All five models (BCI and MIST stains: ER, PR, HER2, Ki67) have been fully trained for **100k iterations** on the NVIDIA A100 partition (`ampere_gpu`) in randomcrop mode. Training is complete.

### Inference & Evaluation

Inference and evaluation have **not yet been run**. The tiling inference step (`tile_inference.py`) is slow — processing 977 images at 16 tiles each through the full diffusion sampling chain takes many hours — and the cluster time limit was reached before inference could be completed. This is the immediate next step for the project.

### Containers on the Cluster

| Container | Location | Used for |
| :--- | :--- | :--- |
| `palette_hpc.sif` | `$VSC_SCRATCH/containers/` | Training and inference on NVIDIA |
| `palette_amd.sif` | `$VSC_SCRATCH/containers/` | Training and inference on AMD |
| `evaluate_amd.sif` | `$VSC_SCRATCH/containers/` | Evaluation only (AMD partition) |

### Next Steps

1. Replace `<JOBID>` placeholders in `path.test_checkpoint` in all five randomcrop configs with the actual SLURM job IDs from the completed training runs.
2. Submit inference jobs via `slurm/test/` scripts for all five datasets.
3. Once predictions are complete, submit evaluation jobs via `slurm/eval/` scripts to produce the final benchmark CSV.
