#!/bin/bash
#SBATCH --job-name=palette_mist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1

set -euo pipefail

export PROJECT_NAME="Palette-Image-to-Image-Diffusion-Models"
export VSC_REPO="$VSC_DATA/projects/palette/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/palette_hpc.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST.sqsh"

module purge
module load calcua/2025a

echo "Starting Palette MIST Training on HPC..."
echo "Job ID: $SLURM_JOB_ID"

# MIST.sqsh contains ER/ HER2/ Ki67/ PR/ at its root.
for STAIN in ER HER2 Ki67 PR; do
    STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

    echo ""
    echo "===== Stain: $STAIN ====="

    srun apptainer exec --nv \
        -B "$DATA_SQSH:/data/MIST:image-src=/" \
        "$CONTAINER" \
        bash -c "cd $VSC_REPO && python run.py -c config/mist_${STAIN_LOWER}.json -p train -gpu 0"

    echo "  $STAIN done."
done

echo ""
echo "All stains complete. Results saved to $VSC_DATA/projects/palette/outputs/experiments"
