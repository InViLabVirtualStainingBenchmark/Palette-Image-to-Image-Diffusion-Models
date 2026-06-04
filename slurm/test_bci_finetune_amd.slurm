#!/bin/bash
#SBATCH --job-name=palette_bci_finetune_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=7:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/test_bci_finetune_amd.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/test_bci_finetune_amd.%j.err

set -euo pipefail

# Checkpoint is read from path.resume_state in config/bci_finetune.json.
# Set that field to the best checkpoint path before submitting, e.g.:
#   .../experiments/train_bci_finetune_<JOBID>/checkpoint/best

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_amd.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/BCI"
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/BCI"

# Per-job MIOpen cache to avoid conflicts between concurrent jobs
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH

mkdir -p "$DATA_DIR"
mkdir -p "$PRED_DIR"

module purge
module load calcua/2025a

echo "Starting Palette BCI Finetune Inference on AMD (arcturus_gpu)..."
echo "Job ID     : $SLURM_JOB_ID"
echo "Output dir : $PRED_DIR"

unset SLURM_CPUS_PER_TASK

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python tile_inference.py \
        --config config/bci_finetune.json \
        --cond_dir $DATA_DIR/HE/test \
        --output_dir $PRED_DIR"

rm -rf $MIOPEN_USER_DB_PATH

echo "BCI finetune inference complete. Predictions at: $PRED_DIR"
