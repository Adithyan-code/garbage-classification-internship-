# week1_preprocessing.py

import os
import cv2
import numpy as np
from pathlib import Path

# Set dataset path (update to your local path)
BASE_DIR = Path("E:\AICTE\TrashType_Image_Dataset")
IMG_SIZE = 100
CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def load_data(subset):
    data = []
    for cat in CATEGORIES:
        path = BASE_DIR / subset / cat
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(str(path / img))
                resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append((resized, CATEGORIES.index(cat)))
            except Exception:
                continue
    print(f"{subset.upper()} set: {len(data)} images loaded.")
    return data


# Load train, validation, and test datasets
train_data = load_data("train")
val_data = load_data("val")
test_data = load_data("test")

# Save the processed data as .npy files (optional for later use)
np.save("train_data.npy", train_data)
np.save("val_data.npy", val_data)
np.save("test_data.npy", test_data)
