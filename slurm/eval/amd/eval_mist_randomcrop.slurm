#!/bin/bash
#SBATCH --job-name=palette_mist_finetune_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/eval_mist_finetune_amd_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/eval_mist_finetune_amd_%x.%j.err

set -euo pipefail

# Submit with STAIN, e.g.:
#   sbatch --job-name=palette_mist_finetune_eval_er \
#          --export=ALL,STAIN=ER \
#          slurm/eval_mist_finetune_amd.slurm

if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set."
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
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/Palette/MIST_${STAIN}"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="/tmp/MIST_eval_finetune"
export OUTPUT_CSV="$VSC_DATA/projects/palette/outputs/eval_results.csv"

module purge
module load calcua/2025a

echo "Starting Palette MIST Finetune Evaluation on AMD â€” Stain: $STAIN"
mkdir -p "$DATA_DIR"

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $EVAL_REPO && python evaluate.py \
        --pred $PRED_DIR \
        --gt $DATA_DIR/$STAIN_DIR/TrainValAB/valB \
        --model_name palette_mist_${STAIN_LOWER}_randomcrop \
        --dataset_name mist \
        --split_name test \
        --output $OUTPUT_CSV"

echo "MIST $STAIN finetune evaluation complete. Results appended to: $OUTPUT_CSV"

