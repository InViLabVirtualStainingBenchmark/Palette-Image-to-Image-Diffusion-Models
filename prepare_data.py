import os
import cv2
import numpy as np

def prepare_bci():
    src_he = "/home/vs_user/Virtual Staining/Datasets/BCI_dataset/HE/test"
    src_ihc = "/home/vs_user/Virtual Staining/Datasets/BCI_dataset/IHC/test"
    dst_root = "data/bci_smoke"
    os.makedirs(os.path.join(dst_root, "color"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "gray"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "flist"), exist_ok=True)

    files = [f for f in os.listdir(src_he) if f.endswith(".png")][:10]
    with open(os.path.join(dst_root, "flist", "train.flist"), "w") as f_list:
        for i, f in enumerate(files):
            padded = f"{i+1:05d}"
            # HE -> gray
            img_he = cv2.imread(os.path.join(src_he, f))
            cv2.imwrite(os.path.join(dst_root, "gray", f"{padded}.png"), img_he)
            # IHC -> color
            img_ihc = cv2.imread(os.path.join(src_ihc, f))
            cv2.imwrite(os.path.join(dst_root, "color", f"{padded}.png"), img_ihc)
            f_list.write(f"{i+1}\n")

def prepare_mist():
    src_unstained = "/home/vs_user/Virtual Staining/Datasets/MIST/Ki67/TrainValAB/trainA"
    src_stained = "/home/vs_user/Virtual Staining/Datasets/MIST/Ki67/TrainValAB/trainB"
    dst_root = "data/mist_smoke"
    os.makedirs(os.path.join(dst_root, "color"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "gray"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "flist"), exist_ok=True)

    files = [f for f in os.listdir(src_unstained) if f.endswith(".jpg")][:10]
    with open(os.path.join(dst_root, "flist", "train.flist"), "w") as f_list:
        for i, f in enumerate(files):
            padded = f"{i+1:05d}"
            # Unstained -> gray
            img_u = cv2.imread(os.path.join(src_unstained, f))
            cv2.imwrite(os.path.join(dst_root, "gray", f"{padded}.png"), img_u)
            # Stained -> color
            img_s = cv2.imread(os.path.join(src_stained, f))
            cv2.imwrite(os.path.join(dst_root, "color", f"{padded}.png"), img_s)
            f_list.write(f"{i+1}\n")

if __name__ == "__main__":
    prepare_bci()
    prepare_mist()
