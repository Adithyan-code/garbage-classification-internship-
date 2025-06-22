# week1_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Load the processed train data (generated in the previous script)
train_data = np.load("train_data.npy", allow_pickle=True)

# Display the first 12 images with their class names
plt.figure(figsize=(12, 6))
for i in range(12):
    image, label = train_data[i]
    plt.subplot(3, 4, i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(CATEGORIES[label])
    plt.axis('off')

plt.tight_layout()
plt.show()
