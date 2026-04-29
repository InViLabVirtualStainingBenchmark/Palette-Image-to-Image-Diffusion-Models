#!/bin/bash
#SBATCH --job-name=palette_mist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/train_mist.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/train_mist.%j.err

set -euo pipefail

# =========================================================
# USER SETTINGS
# =========================================================

REPO_DIR="$VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models"

# =========================================================
# ENVIRONMENT
# =========================================================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load OpenCV/4.8.1-foss-2023a-contrib

source "$VSC_DATA/venvs/venv_palette/bin/activate"

# =========================================================
# PRE-FLIGHT CHECKS
# =========================================================

echo "Python:"
which python
python -V

echo "Checking repository path..."
if [ ! -f "$REPO_DIR/run.py" ]; then
    echo "ERROR: run.py not found in $REPO_DIR"
    deactivate; exit 1
fi

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# =========================================================
# TRAINING — ALL MIST STAINS
# =========================================================

cd "$REPO_DIR"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$VSC_DATA/projects/palette/logs/gpu_mist_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

for stain in ER HER2 Ki67 PR; do

    stain_lower=$(echo "$stain" | tr '[:upper:]' '[:lower:]')
    CONFIG="$REPO_DIR/config/mist_${stain_lower}.json"
    DATA_ROOT="$VSC_SCRATCH/datasets/MIST/$stain/TrainValAB"

    echo ""
    echo "========================================="
    echo "  Stain: $stain"
    echo "========================================="

    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config not found at $CONFIG"
        kill $GPU_LOG_PID 2>/dev/null || true
        deactivate; exit 1
    fi

    for dir in "$DATA_ROOT/trainA" "$DATA_ROOT/trainB" "$DATA_ROOT/valA" "$DATA_ROOT/valB"; do
        if [ ! -d "$dir" ]; then
            echo "ERROR: Missing dataset folder: $dir"
            kill $GPU_LOG_PID 2>/dev/null || true
            deactivate; exit 1
        fi
    done

    echo "Training images : $(find "$DATA_ROOT/trainA" -maxdepth 1 -type f | wc -l)"
    echo "Validation images: $(find "$DATA_ROOT/valA" -maxdepth 1 -type f | wc -l)"
    echo "Config  : $CONFIG"

    CUDA_VISIBLE_DEVICES=0 python run.py -p train -c "$CONFIG"

    echo "  $stain done."

done

kill $GPU_LOG_PID 2>/dev/null || true

deactivate

echo ""
echo "All MIST stains complete."
echo "GPU log : $VSC_DATA/projects/palette/logs/gpu_mist_${SLURM_JOB_ID}.csv"
