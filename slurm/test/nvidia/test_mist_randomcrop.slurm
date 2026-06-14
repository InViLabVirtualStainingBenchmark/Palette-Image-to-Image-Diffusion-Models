#!/bin/bash
#SBATCH --job-name=palette_mist_finetune_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/test_mist_finetune_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/test_mist_finetune_%x.%j.err

set -euo pipefail

# Set BEST_CKPT and STAIN before submitting, e.g.:
#   sbatch --job-name=palette_mist_finetune_test_er \
#          --export=ALL,STAIN=ER,BEST_CKPT=.../checkpoint/best \
#          slurm/test_mist_finetune_hpc.slurm

if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set."
    exit 1
fi
if [ -z "${BEST_CKPT:-}" ]; then
    echo "ERROR: BEST_CKPT not set."
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_hpc.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/MIST"
export EVAL_DIR="$VSC_DATA/projects/palette/outputs/experiments"
export PLOTS_DIR="$VSC_DATA/projects/palette/outputs/plots"
export OUTPUT_DIR="$VSC_DATA/projects/palette/outputs/predictions/mist_${STAIN_LOWER}_randomcrop_${SLURM_JOB_ID}"

module purge
module load calcua/2025a

echo "Starting Palette MIST Fine-tune Test â€” Stain: $STAIN"
echo "Job ID     : $SLURM_JOB_ID"
echo "Checkpoint : $BEST_CKPT"
echo "Output dir : $OUTPUT_DIR"

mkdir -p "$DATA_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p "$OUTPUT_DIR"

# Capture the training experiment folder (latest finetune run for this stain)
TRAIN_EXP_NAME=$(ls -t "$EVAL_DIR" | grep "^train_mist_${STAIN_LOWER}_randomcrop" | head -1)
TRAIN_EXP_PATH="$EVAL_DIR/$TRAIN_EXP_NAME"
echo "Training experiment: $TRAIN_EXP_NAME"

# 1. Tiling inference â€” 1024x1024 predictions
echo ""
echo "Step 1: Tiling inference..."
srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python tile_inference.py \
        --config "config/Virtual staining/new randomcrop/mist_${STAIN_LOWER}_randomcrop.json" \
        --checkpoint $BEST_CKPT \
        --cond_dir $DATA_DIR/$STAIN/TrainValAB/valA \
        --output_dir $OUTPUT_DIR"

# 2. Evaluate predictions vs original 1024x1024 GT â€” no resizing needed
echo ""
echo "Step 2: Benchmark evaluation..."
srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python Benchmark_evaluation/evaluate.py \
        --pred $OUTPUT_DIR \
        --gt $DATA_DIR/$STAIN/TrainValAB/valB \
        --model_name palette_mist_${STAIN_LOWER}_randomcrop \
        --dataset_name mist \
        --split_name test \
        --output $VSC_DATA/projects/palette/outputs/eval_results.csv"

# 3. Plot learning curves from training log
echo ""
echo "Step 3: Plotting learning curves..."
srun apptainer exec \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python plot_learning_curves.py \
        --exp $TRAIN_EXP_PATH \
        --out $PLOTS_DIR \
        --name palette_mist_${STAIN_LOWER}_randomcrop"

echo ""
echo "All done."
echo "Predictions  : $OUTPUT_DIR"
echo "Eval results : $VSC_DATA/projects/palette/outputs/eval_results.csv"
echo "Learning curves : $PLOTS_DIR/palette_mist_${STAIN_LOWER}_randomcrop_learning_curves.png"

