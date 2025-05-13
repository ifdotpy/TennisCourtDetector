import numpy as np
import cv2
from scipy.ndimage import label


def find_centroid_adaptive_threshold(matrix, base_threshold=0.1, adaptive_ratio=0.3):
    """
    Find centroid with adaptive threshold based on the maximum value in the heatmap
    """
    if np.max(matrix) <= 0:
        return [], (None, None)

    # Find max position and value in one go
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    max_val = matrix[max_idx]
    
    threshold = max(base_threshold, adaptive_ratio * max_val)
    mask = matrix >= threshold

    if not np.any(mask):
        return [], (None, None)

    labeled_matrix, num_features = label(mask)
    if num_features == 0:
        return [], (None, None)

    max_label = labeled_matrix[max_idx]
    if max_label == 0:
        return [], (None, None)

    # Get points directly where labeled_matrix equals max_label
    points = np.argwhere(labeled_matrix == max_label)
    if len(points) == 0:  # Safeguard, though shouldn't happen given above checks
        return [], (None, None)
        
    # Use numpy fancy indexing instead of list comprehension
    values = matrix[points[:,0], points[:,1]]
    total_weight = np.sum(values)
    
    centroid_y = np.sum(points[:, 0] * values) / total_weight
    centroid_x = np.sum(points[:, 1] * values) / total_weight

    return points, (centroid_x, centroid_y)

# Export the methods so they can be imported in evaluate_centroid_methods.py
__all__ = ["find_centroid_adaptive_threshold"] 