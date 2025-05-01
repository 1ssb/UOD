import numpy as np
import logging
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
from typing import Tuple, List, Dict, Callable, Any
from utilities_3D import *

# Define a global epsilon
EPS = 1e-10

def threshold_func(grid: np.ndarray) -> float:
    """
    Computes a threshold value based on the size of the input grid.
    Avoids division by zero for an empty grid.
    """
    if grid.size == 0:
        raise ValueError("Grid size is zero; cannot compute threshold.")
    return 1.4 / grid.size

def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Normalize the heatmap using the specified tau parameter.
    Applies the transformation y = softmax(ln(c / (1 - c)) / tau).
    Ensures all-zero heatmaps are adjusted by EPS to avoid division by zero.

    Parameters:
        heatmap (np.ndarray): The input heatmap array with values in (0, 1).
        tau (float): Temperature parameter to scale the normalization.

    Returns:
        np.ndarray: The normalized heatmap array.
    """
    # Ensure values are within (0, 1) by clipping
    clip_heatmap = np.clip(heatmap, EPS, 1 - EPS)

    # Compute the logit transformation ln(c / (1 - c))
    logits = np.log(clip_heatmap / (1 - clip_heatmap))

    # Apply softmax to obtain normalized probabilities
    exp_logits = np.exp(logits)
    normalized_heatmap = exp_logits / np.sum(exp_logits)

    return normalized_heatmap

def compute_rwa(gt_heatmap: np.ndarray, pred_heatmap: np.ndarray) -> float:
    """
    Compute region-wise classification accuracy for 640x360 heatmaps using region-wise labels.
    """
    if gt_heatmap.ndim != 2 or gt_heatmap.shape not in [(640, 360), (360, 640)]:
        raise ValueError("gt_heatmap must be a 2D array with dimensions 640x360 or 360x640.")
    if pred_heatmap.ndim != 2 or pred_heatmap.shape not in [(640, 360), (360, 640)]:
        raise ValueError("pred_heatmap must be a 2D array with dimensions 640x360 or 360x640.")
    
    if gt_heatmap.shape == (360, 640):
        gt_heatmap = gt_heatmap.T
    if pred_heatmap.shape == (360, 640):
        pred_heatmap = pred_heatmap.T

    gt_normalized = normalize_heatmap(gt_heatmap)
    pred_normalized = normalize_heatmap(pred_heatmap)
    
    global_threshold = threshold_func(gt_normalized)
    regions = [slice(0, 140), slice(140, 500), slice(500, 640)]

    gt_labels = [int(np.any(gt_normalized[:, region] > global_threshold)) for region in regions]
    pred_labels = [int(np.any(pred_normalized[:, region] > global_threshold)) for region in regions]

    return np.mean([gt == pred for gt, pred in zip(gt_labels, pred_labels)])

def compute_entropy(heatmap: np.ndarray) -> float:
    """
    Compute the Shannon entropy of a heatmap, handling edge cases to prevent NaN.
    """
    normalized_heatmap = normalize_heatmap(heatmap)

    # Compute entropy safely
    entropy = -np.sum(normalized_heatmap * np.log(normalized_heatmap))
    print("ENT:", entropy)
    norm_entropy = entropy / np.log(heatmap.size)
    return norm_entropy

def compute_cross_entropy(gt_heatmap: np.ndarray, pred_heatmap: np.ndarray, confidence_threshold: float = 1e-6) -> float:
    """
    Compute the cross-entropy between two heatmaps, focusing only on non-zero regions to handle sparsity.
    """

    # Normalize both heatmaps
    gt_normalized = normalize_heatmap(gt_heatmap)
    pred_normalized = normalize_heatmap(pred_heatmap)

    # Calculate cross-entropy
    cross_entropy = -np.sum(gt_normalized * np.log(pred_normalized))
    print("CE:", cross_entropy)
    # Normalize by the number of significant elements for an average cross-entropy
    norm_cross_ent = cross_entropy / np.log(gt_heatmap.size)

    return norm_cross_ent

def is_uniform_grid(grid: np.ndarray) -> bool:
    """
    Check if all values in the grid are close to uniform probability values.
    """
    uniform_value = 1.0 / grid.size
    return np.all(np.abs(grid - uniform_value) < EPS)

def are_grids_identical(grid1: np.ndarray, grid2: np.ndarray) -> bool:
    """
    Check if two grids are identical within a small tolerance.
    """
    return np.all(np.abs(grid1 - grid2) < EPS)

def compute_DNN(G_grid: np.ndarray, H_grid: np.ndarray) -> float:
    """
    Compute the peak-to-peak distance between G_grid and H_grid.
    """

    kernel_sizes = [3, 5, 7, 10]
    all_G_peaks = detect_peaks(G_grid, kernel_sizes)

    if all_G_peaks.size == 0:
        return float('inf')

    G_grid = normalize_heatmap(G_grid)
    H_grid = normalize_heatmap(H_grid)

    if is_uniform_grid(H_grid):
        return float('inf')

    if are_grids_identical(G_grid, H_grid):
        return 0.0

    all_H_peaks = detect_peaks(H_grid, kernel_sizes)

    if all_H_peaks.size == 0:
        return float('inf')

    H_tree = cKDTree(all_H_peaks)
    distances, _ = H_tree.query(all_G_peaks, k=1)

    return np.mean(distances) if distances.size > 0 else 0.0

def detect_peaks(grid: np.ndarray, kernel_sizes: list) -> np.ndarray:
    """
    Detect peaks in the grid based on a list of kernel sizes.
    """
    all_peaks = []
    threshold = threshold_func(grid)

    for kernel_size in kernel_sizes:
        kernel = (kernel_size,) * grid.ndim
        local_maxima = (maximum_filter(grid, size=kernel) == grid) & (grid > threshold)
        peaks = np.argwhere(local_maxima)
        if peaks.size > 0:
            all_peaks.append(peaks)

    return np.unique(np.vstack(all_peaks), axis=0) if all_peaks else np.array([])

def classify_grid(G_grid: np.ndarray, H_grid: np.ndarray) -> str:
    """
    Classify the grid comparison as "TP" (True Positive), "FN" (False Negative), or "E" (Everything Else).
    
    Args:
        G_grid (np.ndarray): Ground truth grid.
        H_grid (np.ndarray): Predicted grid.

    Returns:
        str: A string classification for the entire grid:
             - "TP" if both G_grid and H_grid exceed the threshold in significant parts.
             - "FN" if G_grid exceeds the threshold but H_grid does not.
             - "E" (Everything Else) otherwise.
    """
    # Determine the threshold using G_grid
    threshold = threshold_func(G_grid)

    # Check if G_grid has significant values above the threshold
    g_above_threshold = np.any(G_grid > threshold)

    # Check if H_grid has significant values above the threshold
    h_above_threshold = np.any(H_grid > threshold)

    # Classify based on the above conditions
    if g_above_threshold and h_above_threshold:
        return "TP"  # True Positive
    elif g_above_threshold and not h_above_threshold:
        return "FN"  # False Negative
    else:
        return "E"  # Everything Else

def get_results(GT_array: np.ndarray, H_array: np.ndarray) -> Dict[str, float]:
    """
    Get 2D array results including entropy, cross-entropy, DNN, RWA, and failure rate.
    """
    try:
        return {
            "Entropy_2D": compute_entropy(H_array),
            "Cross_Entropy_2D": compute_cross_entropy(GT_array, H_array),
            "DNN_2D": round(compute_DNN(GT_array, H_array) * 100 / 743.03, 4),
            "RWA": compute_rwa(GT_array, H_array),
            "Failure": classify_grid(GT_array, H_array)
        }
    except Exception as e:
        logging.error(f"Error in get_results: {e}")
        raise

def compare_values(GT_grid: np.ndarray, H_grid: np.ndarray) -> Dict[str, float]:
    """
    Get 3D array results including entropy, cross-entropy, DNN, and failure rate.
    """
    try:
        x, y, z = H_grid.shape
        diagonal = (x**2 + y**2 + z**2) ** 0.5

        return {
            "Entropy_3D": compute_entropy(H_grid),
            "Cross_Entropy_3D": compute_cross_entropy(GT_grid, H_grid),
            "DNN_3D": compute_DNN(GT_grid, H_grid) / diagonal,
            "Failure": classify_grid(GT_grid, H_grid)
        }
    except Exception as e:
        logging.error(f"Error in compare_values: {e}")
        raise
