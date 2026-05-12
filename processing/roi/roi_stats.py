# processing/roi/roi_stats.py

import numpy as np

   
" Compute the histogram of intensity values in the ROI."
def compute_roi_histogram(roi: np.ndarray, num_bins: int = 256):
    "convert 2D array to 1D list 3shan a3ml loop 3la kol pixel "
    flat = roi.flatten().astype(np.int32)

    "h3ml counter fady "
    hist = np.zeros(num_bins, dtype=np.int64)

    "h3dy 3la kol pixel w h3od a3d kol color mwgod f kam pixel mn sora"
    for pixel in flat:
        idx = max(0, min(int(pixel), num_bins - 1))
        hist[idx] += 1
    
    "b3ml defines the boundaries bt3t kol bucket"
    bin_edges = np.arange(num_bins + 1)
    return hist, bin_edges


def compute_roi_mean(roi: np.ndarray) -> float:
    """
    Formula: sum(pixels) / count
    """
    flat  = roi.flatten().astype(np.float64)
    
    total = 0.0
    for px in flat:
        total += px
    return total / len(flat)


def compute_roi_variance(roi: np.ndarray) -> float:
    """
    Formula: sum((xi - mean)^2) / N
    """
    flat     = roi.flatten().astype(np.float64)
    mean     = compute_roi_mean(roi)
    sq_sum   = 0.0
    for px in flat:
        sq_sum += (px - mean) ** 2
    return sq_sum / len(flat)