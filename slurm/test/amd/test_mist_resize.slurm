#!/bin/bash
#SBATCH --job-name=palette_mist_resize_infer_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/test_mist_resize_amd_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/test_mist_resize_amd_%x.%j.err

set -euo pipefail

# Set resume_state in config/Virtual staining/original resize/mist_<stain>_resize.json before submitting.
# e.g. "resume_state": ".../experiments/train_mist_er_resize_<JOBID>/checkpoint/last"
#
# Submit with:
#   sbatch --job-name=palette_mist_resize_infer_amd_er --export=ALL,STAIN=ER slurm/test/amd/test_mist_resize.slurm

if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set. Submit with: sbatch --export=ALL,STAIN=ER slurm/test/amd/test_mist_resize.slurm"
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_amd.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/MIST"

# Per-job MIOpen cache to avoid conflicts between concurrent jobs
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH

mkdir -p "$DATA_DIR"

module purge
module load calcua/2025a

echo "Starting Palette MIST Resize Inference on AMD (arcturus_gpu) — Stain: $STAIN"
echo "Job ID : $SLURM_JOB_ID"

unset SLURM_CPUS_PER_TASK

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python3 run.py -c \"config/Virtual staining/original resize/mist_${STAIN_LOWER}_resize.json\" -p test -gpu 0"

rm -rf $MIOPEN_USER_DB_PATH

echo ""
echo "MIST $STAIN resize inference complete."
echo "Results : $VSC_DATA/projects/palette/outputs/experiments"
