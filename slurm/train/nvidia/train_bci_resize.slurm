#!/bin/bash
#SBATCH --job-name=palette_bci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/train_bci.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/train_bci.%j.err

set -euo pipefail

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_hpc.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/BCI"

module purge
module load calcua/2025a

echo "Starting Palette BCI Training on HPC..."
echo "Job ID: $SLURM_JOB_ID"

mkdir -p "$DATA_DIR"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$VSC_DATA/projects/palette/logs/gpu_bci_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python run.py -c \"config/Virtual staining/original resize/bci_resize.json\" -p train -gpu 0"

kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "BCI training complete."
echo "GPU log : $VSC_DATA/projects/palette/logs/gpu_bci_${SLURM_JOB_ID}.csv"
echo "Results : $VSC_DATA/projects/palette/outputs/experiments"
