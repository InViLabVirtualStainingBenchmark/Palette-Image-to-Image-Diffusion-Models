#!/bin/bash
#SBATCH --job-name=palette_mist_finetune_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=7:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/test_mist_finetune_amd_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/test_mist_finetune_amd_%x.%j.err

set -euo pipefail

# Checkpoint is read from path.resume_state in config/mist_<stain>_finetune.json.
# Set that field to the best checkpoint path before submitting, e.g.:
#   .../experiments/train_mist_er_finetune_<JOBID>/checkpoint/best
#
# Submit with STAIN, e.g.:
#   sbatch --job-name=palette_mist_finetune_infer_er \
#          --export=ALL,STAIN=ER \
#          slurm/test_mist_finetune_amd.slurm

if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set."
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_amd.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/MIST"
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_${STAIN}"

# Per-job MIOpen cache to avoid conflicts between concurrent jobs
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH

mkdir -p "$DATA_DIR"
mkdir -p "$PRED_DIR"

module purge
module load calcua/2025a

echo "Starting Palette MIST Finetune Inference on AMD (arcturus_gpu) — Stain: $STAIN"
echo "Job ID     : $SLURM_JOB_ID"
echo "Output dir : $PRED_DIR"

unset SLURM_CPUS_PER_TASK

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python tile_inference.py \
        --config config/mist_${STAIN_LOWER}_finetune.json \
        --cond_dir $DATA_DIR/$STAIN/TrainValAB/valA \
        --output_dir $PRED_DIR"

rm -rf $MIOPEN_USER_DB_PATH

echo "MIST $STAIN finetune inference complete. Predictions at: $PRED_DIR"
