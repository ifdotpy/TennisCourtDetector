import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import courtDataset

# Assuming courtDataset is defined and implemented correctly in dataset.py
train_dataset = courtDataset('train', augment=True)  # Enable augmentation


batch = train_dataset[random.randint(0, len(train_dataset) - 1)]
print('Got batch')
inp = batch[0]
hm_hp = batch[1]

inp = np.squeeze(inp)

print(inp.shape)
print(np.min(inp), np.max(inp))

def visualize_heatmaps_on_image(inp, hm_hp):
    # Combine all heatmaps
    # combined_heatmap = np.sum(hm_hp, axis=0)
    combined_heatmap = hm_hp[0]

    print(combined_heatmap.shape)

    # Create a figure
    plt.figure(figsize=(10, 8))

    # Overlay the combined heatmap on the input image
    # add point
    # plot array of points in batch[2]

    points = batch[2]
    # filter out None values using numpy
    points = points[~np.isnan(points).any(axis=1)]



    plt.scatter(points[:, 0], points[:, 1], c='red', s=10)
    plt.scatter(points[:, 0], points[:, 1], c='red', s=10)
    plt.imshow(inp, cmap='Grays')
    plt.imshow(combined_heatmap, cmap='Reds', alpha=0.5, interpolation='nearest')
    plt.title('Combined Heatmaps')
    plt.axis('off')

    plt.show()


visualize_heatmaps_on_image(inp, hm_hp)
