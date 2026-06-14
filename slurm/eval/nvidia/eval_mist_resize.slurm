#!/bin/bash
#SBATCH --job-name=palette_mist_resize_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/eval_mist_resize_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/eval_mist_resize_%x.%j.err

set -euo pipefail

# Submit with:
#   sbatch --job-name=palette_mist_resize_eval_er --export=ALL,STAIN=ER slurm/eval/nvidia/eval_mist_resize.slurm

if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set. Submit with: sbatch --export=ALL,STAIN=ER slurm/eval/nvidia/eval_mist_resize.slurm"
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

# Handle case-sensitivity for the Ki67 dataset folder inside the squashfs
if [ "$STAIN" == "KI67" ]; then
    STAIN_DIR="Ki67"
else
    STAIN_DIR="$STAIN"
fi

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_hpc.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/MIST"
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/predictions/Palette/MIST_${STAIN}"
export GT_DIR="$DATA_DIR/$STAIN_DIR/TrainValAB/valB"
export OUTPUT_CSV="/data/antwerpen/212/vsc21213/projects/palette/outputs/results/eval_results.csv"

module purge
module load calcua/2025a

unset SLURM_CPUS_PER_TASK

echo "Starting Palette MIST Resize Evaluation — Stain: $STAIN"
echo "Job ID : $SLURM_JOB_ID"
echo "Pred   : $PRED_DIR"
echo "GT     : $GT_DIR"

mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$OUTPUT_CSV")"

srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "python $VSC_REPO/Benchmark_evaluation/evaluate.py \
        --pred $PRED_DIR \
        --gt $GT_DIR \
        --model_name palette_mist_${STAIN_LOWER} \
        --dataset_name mist \
        --split_name test \
        --output $OUTPUT_CSV"

echo ""
echo "Done. Results saved to: $OUTPUT_CSV"
