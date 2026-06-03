#!/bin/bash
#SBATCH --job-name=palette_mist_finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/train_mist_finetune_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/train_mist_finetune_%x.%j.err

set -euo pipefail

# Pass stain at submit time:
#   sbatch --job-name=palette_mist_finetune_er --export=ALL,STAIN=ER slurm/train_mist_finetune_hpc.slurm
if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set. Submit with: sbatch --export=ALL,STAIN=ER slurm/train_mist_finetune_hpc.slurm"
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_hpc.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/MIST"

module purge
module load calcua/2025a

echo "Starting Palette MIST Fine-tune Training — Stain: $STAIN"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p "$DATA_DIR"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$VSC_DATA/projects/palette/logs/gpu_mist_finetune_${STAIN_LOWER}_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python run.py -c config/mist_${STAIN_LOWER}_finetune.json -p train -gpu 0"

kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "$STAIN fine-tune training complete."
echo "GPU log : $VSC_DATA/projects/palette/logs/gpu_mist_finetune_${STAIN_LOWER}_${SLURM_JOB_ID}.csv"
echo "Results : $VSC_DATA/projects/palette/outputs/experiments"
