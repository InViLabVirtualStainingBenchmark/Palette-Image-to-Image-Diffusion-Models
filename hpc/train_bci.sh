#!/bin/bash
#SBATCH --job-name=palette_bci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/train_bci.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/train_bci.%j.err

set -euo pipefail

# =========================================================
# USER SETTINGS
# =========================================================

REPO_DIR="$VSC_DATA/projects/palette/code/Palette-Image-to-Image-Diffusion-Models"
CONFIG="$REPO_DIR/config/bci.json"
DATA_ROOT="$VSC_SCRATCH/datasets/BCI"

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

echo "Checking config..."
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    deactivate; exit 1
fi

echo "Checking dataset..."
for dir in "$DATA_ROOT/HE/train" "$DATA_ROOT/HE/test" "$DATA_ROOT/IHC/train" "$DATA_ROOT/IHC/test"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Missing dataset folder: $dir"
        deactivate; exit 1
    fi
done

echo "Training images : $(find "$DATA_ROOT/HE/train" -maxdepth 1 -type f | wc -l)"
echo "Test images     : $(find "$DATA_ROOT/HE/test"  -maxdepth 1 -type f | wc -l)"

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# =========================================================
# TRAINING
# =========================================================

cd "$REPO_DIR"

echo ""
echo "Starting BCI training..."
echo "  config : $CONFIG"
echo "  data   : $DATA_ROOT"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$VSC_DATA/projects/palette/logs/gpu_bci_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

CUDA_VISIBLE_DEVICES=0 python run.py -p train -c "$CONFIG"

kill $GPU_LOG_PID 2>/dev/null || true

deactivate

echo ""
echo "BCI training complete."
echo "GPU log : $VSC_DATA/projects/palette/logs/gpu_bci_${SLURM_JOB_ID}.csv"
