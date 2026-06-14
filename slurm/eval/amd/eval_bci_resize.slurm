#!/bin/bash
#SBATCH --job-name=palette_bci_resize_eval_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/eval_bci_resize_amd.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/eval_bci_resize_amd.%j.err

set -euo pipefail

export CONTAINER="$VSC_SCRATCH/containers/evaluate_amd.sif"
export EVAL_REPO="$VSC_DATA/evaluate"
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/predictions/Palette/BCI"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="/tmp/BCI_eval_resize"
export OUTPUT_CSV="$VSC_DATA/projects/palette/outputs/eval_results.csv"

module purge
module load calcua/2025a

echo "Starting Palette BCI Resize Evaluation on AMD..."
echo "Job ID : $SLURM_JOB_ID"
echo "Pred   : $PRED_DIR"
echo "GT     : $DATA_DIR/IHC/test"

mkdir -p "$DATA_DIR"

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $EVAL_REPO && python evaluate.py \
        --pred $PRED_DIR \
        --gt $DATA_DIR/IHC/test \
        --model_name palette_bci \
        --dataset_name bci \
        --split_name test \
        --output $OUTPUT_CSV"

echo ""
echo "BCI resize evaluation complete. Results appended to: $OUTPUT_CSV"
