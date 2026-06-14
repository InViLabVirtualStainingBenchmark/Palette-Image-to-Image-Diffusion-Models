#!/bin/bash
#SBATCH --job-name=palette_mist_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/test_mist_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/test_mist_%x.%j.err

set -euo pipefail

# Set resume_state in config/Virtual staining/original resize/mist_<stain>_resize.json before submitting.
# e.g. "resume_state": ".../checkpoint/best"
#
# Submit with:
#   sbatch --job-name=palette_mist_test_er --export=ALL,STAIN=ER slurm/test_mist_hpc.slurm

if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set. Submit with: sbatch --export=ALL,STAIN=ER slurm/test_mist_hpc.slurm"
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

module purge
module load calcua/2025a

echo "Starting Palette MIST Test — Stain: $STAIN"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p "$DATA_DIR"
mkdir -p "$PLOTS_DIR"

# Capture the training experiment folder (latest for this stain, before test runs)
TRAIN_EXP_NAME=$(ls -t "$EVAL_DIR" | grep "^mist_${STAIN_LOWER}" | head -1)
TRAIN_EXP_PATH="$EVAL_DIR/$TRAIN_EXP_NAME"
echo "Training experiment: $TRAIN_EXP_NAME"

# 1. Run test phase — creates a new mist_<stain>_* folder
srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python run.py -c \"config/Virtual staining/original resize/mist_${STAIN_LOWER}_resize.json\" -p test -gpu 0"

# Capture the test experiment folder (latest for this stain after test runs)
TEST_EXP_NAME=$(ls -t "$EVAL_DIR" | grep "^mist_${STAIN_LOWER}" | head -1)
TEST_EXP_PATH="$EVAL_DIR/$TEST_EXP_NAME"
echo "Test experiment: $TEST_EXP_NAME"

echo ""
echo "Test complete. Preparing eval outputs..."

# 2. Strip Out_ prefix → benchmark_outputs/
srun apptainer exec \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python prepare_eval_outputs.py --exp $TEST_EXP_PATH"

echo ""
echo "Running benchmark evaluation..."

# 3. Evaluate predictions vs ground truth (valB is the MIST official test set)
srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python Benchmark_evaluation/evaluate.py \
        --pred $VSC_REPO/benchmark_outputs/$TEST_EXP_NAME \
        --gt $DATA_DIR/$STAIN/TrainValAB/valB \
        --model_name palette_mist_${STAIN_LOWER} \
        --dataset_name mist \
        --split_name test \
        --output $VSC_DATA/projects/palette/outputs/eval_results.csv"

echo ""
echo "Plotting learning curves..."

# 4. Plot training + validation learning curves from train.log
srun apptainer exec \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python plot_learning_curves.py \
        --exp $TRAIN_EXP_PATH \
        --out $PLOTS_DIR \
        --name palette_mist_${STAIN_LOWER}"

echo ""
echo "All done."
echo "Eval results : $VSC_DATA/projects/palette/outputs/eval_results.csv"
echo "Learning curves : $PLOTS_DIR/palette_mist_${STAIN_LOWER}_learning_curves.png"
