#!/bin/bash
#SBATCH --job-name=palette_install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/palette/logs/install.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/palette/logs/install.%j.err

set -euo pipefail

# =========================================================
# CONFIG
# =========================================================

VENV_DIR="$VSC_DATA/venvs/venv_palette"

# =========================================================
# MODULES
# =========================================================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load OpenCV/4.8.1-foss-2023a-contrib

echo "Python used:"
which python
python -V

# =========================================================
# RECREATE VENV (CLEAN)
# =========================================================

rm -rf "$VENV_DIR"

python -m venv "$VENV_DIR" --system-site-packages

source "$VENV_DIR/bin/activate"

echo "Active python:"
which python
python -V

# =========================================================
# INSTALL EXTRA PACKAGES
# =========================================================

python -m pip install --upgrade pip

python -m pip install \
    tensorboardX \
    clean-fid \
    --no-cache-dir \
    --no-build-isolation

# =========================================================
# SANITY CHECKS
# =========================================================

python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import cv2; print('cv2:', cv2.__version__)"
python -c "import scipy; print('scipy OK')"
python -c "import pandas; print('pandas OK')"
python -c "import tqdm; print('tqdm OK')"
python -c "import tensorboardX; print('tensorboardX OK')"
python -c "import cleanfid; print('clean-fid OK')"

deactivate

echo "Done. Next: sbatch train_bci.sh  OR  sbatch train_mist.sh"
