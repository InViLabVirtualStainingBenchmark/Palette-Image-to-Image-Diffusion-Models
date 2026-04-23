import os
import cv2

DATASETS_ROOT = "/home/vs_user/Virtual Staining/Datasets"


def prepare_bci():
    src_he  = os.path.join(DATASETS_ROOT, "BCI_dataset/HE/test")
    src_ihc = os.path.join(DATASETS_ROOT, "BCI_dataset/IHC/test")
    dst_root = "data/bci"
    _setup_dirs(dst_root)

    files = [f for f in os.listdir(src_he) if f.endswith(".png")][:10]
    with open(os.path.join(dst_root, "flist", "train.flist"), "w") as flist:
        for i, f in enumerate(files):
            padded = f"{i+1:05d}"
            cv2.imwrite(os.path.join(dst_root, "gray",  f"{padded}.png"), cv2.imread(os.path.join(src_he,  f)))
            cv2.imwrite(os.path.join(dst_root, "color", f"{padded}.png"), cv2.imread(os.path.join(src_ihc, f)))
            flist.write(f"{i+1}\n")
    print(f"BCI: wrote {len(files)} pairs to {dst_root}")


def _prepare_mist_modality(modality, dst_root, ext=".jpg"):
    src_unstained = os.path.join(DATASETS_ROOT, f"MIST/{modality}/TrainValAB/trainA")
    src_stained   = os.path.join(DATASETS_ROOT, f"MIST/{modality}/TrainValAB/trainB")
    _setup_dirs(dst_root)

    files = [f for f in os.listdir(src_unstained) if f.endswith(ext)][:10]
    with open(os.path.join(dst_root, "flist", "train.flist"), "w") as flist:
        for i, f in enumerate(files):
            padded = f"{i+1:05d}"
            cv2.imwrite(os.path.join(dst_root, "gray",  f"{padded}.png"), cv2.imread(os.path.join(src_unstained, f)))
            cv2.imwrite(os.path.join(dst_root, "color", f"{padded}.png"), cv2.imread(os.path.join(src_stained,   f)))
            flist.write(f"{i+1}\n")
    print(f"MIST {modality}: wrote {len(files)} pairs to {dst_root}")


def _setup_dirs(root):
    for sub in ("color", "gray", "flist"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)


if __name__ == "__main__":
    prepare_bci()
    _prepare_mist_modality("Ki67", "data/mist_ki67")
    _prepare_mist_modality("PR",   "data/mist_pr")
    _prepare_mist_modality("HER2", "data/mist_her2")
    _prepare_mist_modality("ER",   "data/mist_er")