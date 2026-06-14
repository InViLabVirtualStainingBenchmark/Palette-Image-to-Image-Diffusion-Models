#!/bin/bash
#SBATCH --job-name=palette_bci_resize_infer_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/test_bci_resize_amd.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/test_bci_resize_amd.%j.err

set -euo pipefail

# Set resume_state in config/Virtual staining/original resize/bci_resize.json before submitting.
# e.g. "resume_state": ".../experiments/train_bci_resize_<JOBID>/checkpoint/last"

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_amd.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/BCI"

# Per-job MIOpen cache to avoid conflicts between concurrent jobs
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH

mkdir -p "$DATA_DIR"

module purge
module load calcua/2025a

echo "Starting Palette BCI Resize Inference on AMD (arcturus_gpu)..."
echo "Job ID : $SLURM_JOB_ID"

unset SLURM_CPUS_PER_TASK

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python3 run.py -c \"config/Virtual staining/original resize/bci_resize.json\" -p test -gpu 0"

rm -rf $MIOPEN_USER_DB_PATH

echo ""
echo "BCI resize inference complete."
echo "Results : $VSC_DATA/projects/palette/outputs/experiments"
