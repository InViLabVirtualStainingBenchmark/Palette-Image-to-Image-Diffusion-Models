#!/bin/bash
#SBATCH --job-name=palette_mist_resize_eval_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/eval_mist_resize_amd_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/eval_mist_resize_amd_%x.%j.err

set -euo pipefail

# Submit with:
#   sbatch --job-name=palette_mist_resize_eval_amd_er --export=ALL,STAIN=ER slurm/eval/amd/eval_mist_resize.slurm

if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set. Submit with: sbatch --export=ALL,STAIN=ER slurm/eval/amd/eval_mist_resize.slurm"
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

# Handle case-sensitivity for the Ki67 dataset folder inside the squashfs
if [ "$STAIN" == "KI67" ]; then
    STAIN_DIR="Ki67"
else
    STAIN_DIR="$STAIN"
fi

export CONTAINER="$VSC_SCRATCH/containers/evaluate_amd.sif"
export EVAL_REPO="$VSC_DATA/evaluate"
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/predictions/Palette/MIST_${STAIN}"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="/tmp/MIST_eval_resize"
export OUTPUT_CSV="$VSC_DATA/projects/palette/outputs/eval_results.csv"

module purge
module load calcua/2025a

echo "Starting Palette MIST Resize Evaluation on AMD — Stain: $STAIN"
echo "Job ID : $SLURM_JOB_ID"
echo "Pred   : $PRED_DIR"
echo "GT     : $DATA_DIR/$STAIN_DIR/TrainValAB/valB"

mkdir -p "$DATA_DIR"

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $EVAL_REPO && python evaluate.py \
        --pred $PRED_DIR \
        --gt $DATA_DIR/$STAIN_DIR/TrainValAB/valB \
        --model_name palette_mist_${STAIN_LOWER} \
        --dataset_name mist \
        --split_name test \
        --output $OUTPUT_CSV"

echo ""
echo "MIST $STAIN resize evaluation complete. Results appended to: $OUTPUT_CSV"
