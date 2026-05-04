import numpy as np
def compute_histogram(image_array, bins=256):
    # Input validation
    if image_array is None:
        raise ValueError("Input image is None. Please provide a valid image array.")
    
    if image_array.size == 0:
        raise ValueError("Input image is empty. Cannot compute histogram.")
    
    # Initialize histogram with zeros (using numpy for efficiency)
    hist = np.zeros(bins, dtype=np.int64)
    
    # Flatten image to 1D for iteration
    flat_image = image_array.flatten()
    
    # Manual counting loop - this is the "from scratch" implementation
    # No np.histogram, np.bincount, or similar built-ins
    for pixel_value in flat_image:
        pixel_int = int(pixel_value)
        # Map pixel value to bin index based on number of bins
        if bins == 256:
            # Standard 8-bit: direct mapping
            if 0 <= pixel_int < 256:
                hist[pixel_int] += 1
        else:
            # Custom bins: scale pixel value to bin range
            bin_index = int(pixel_int * bins / 256)
            if 0 <= bin_index < bins:
                hist[bin_index] += 1
    
    # Create bin edges (0 to 255 for 8-bit images)
    bin_edges = np.arange(bins + 1, dtype=np.int64)
    
    return hist, bin_edges
def compute_cdf(hist):
    # Input validation
    if hist is None:
        raise ValueError("Histogram is None. Cannot compute CDF.")
    
    if len(hist) == 0:
        raise ValueError("Histogram is empty. Cannot compute CDF.")
    
    # Compute cumulative sum manually
    cdf = np.zeros_like(hist, dtype=np.float64)
    cumulative_sum = 0
    
    for i in range(len(hist)):
        cumulative_sum += hist[i]
        cdf[i] = cumulative_sum
    
    # Handle uniform image case (all same pixel value)
    if cumulative_sum == 0:
        return np.zeros_like(cdf, dtype=np.float64)
    
    # Normalize to [0, 255] using min-max normalization
    cdf_min = cdf.min()
    cdf_max = cdf.max()
    
    if cdf_max == cdf_min:
        # All counts are the same - return uniform mapping
        return np.linspace(0, 255, len(hist))
    
    cdf_normalized = (cdf - cdf_min) / (cdf_max - cdf_min) * 255.0
    
    return cdf_normalized


def global_histogram_equalization(image_array):
    # Input validation
    if image_array is None:
        raise ValueError("Input image is None. Please provide a valid image array.")
    
    if image_array.size == 0:
        raise ValueError("Input image is empty. Cannot perform equalization.")
    
    if len(image_array.shape) != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image_array.shape}"
        )
    
    # Step 1: Compute histogram
    hist, _ = compute_histogram(image_array)
    
    # Step 2: Compute normalized CDF (mapping function)
    cdf_normalized = compute_cdf(hist)
    
    # Step 3: Map each pixel through the CDF
    rows, cols = image_array.shape
    equalized = np.zeros_like(image_array, dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            pixel_value = int(image_array[i, j])
            equalized[i, j] = np.uint8(np.round(cdf_normalized[pixel_value]))
    
    return equalized


def local_histogram_equalization(image_array, block_size=8):
    # ===== Input Validation =====
    if image_array is None:
        raise ValueError("Input image is None. Cannot perform equalization.")
    
    if image_array.size == 0:
        raise ValueError("Input image is empty. Cannot perform equalization.")
    
    if len(image_array.shape) != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image_array.shape}"
        )
    
    if not isinstance(block_size, (int, np.integer)):
        raise ValueError(
            f"Block size must be an integer, got {type(block_size).__name__}"
        )
    
    block_size = int(block_size)
    
    if block_size < 2:
        raise ValueError(
            f"Block size must be at least 2 for meaningful local statistics. "
            f"Got {block_size}."
        )
    
    if block_size % 2 == 0:
        # Even block sizes work but create asymmetric neighborhoods
        # Warning is optional - algorithm still functions correctly
        pass
    
    rows, cols = image_array.shape
    
    if block_size > min(rows, cols):
        raise ValueError(
            f"Block size ({block_size}) exceeds image dimensions "
            f"({rows}×{cols}). Block size must be ≤ {min(rows, cols)}."
        )
    
    # ===== Main Algorithm =====
    result = np.zeros_like(image_array, dtype=np.uint8)
    
    # Pad image using reflect mode for edge handling
    # Pad by half block_size on each side
    pad_size = block_size // 2
    padded = np.pad(image_array, pad_size, mode='reflect')
    
    # Process each pixel
    for i in range(rows):
        for j in range(cols):
            # Extract local block centered at (i, j)
            # In padded coordinates: (i + pad_size, j + pad_size) is the center
            local_block = padded[i:i + block_size, j:j + block_size]
            
            # Compute histogram of local region
            local_hist, _ = compute_histogram(local_block)
            
            # Compute CDF of local region
            local_cdf = compute_cdf(local_hist)
            
            # Map center pixel through local CDF
            center_pixel = int(image_array[i, j])
            result[i, j] = np.uint8(np.round(local_cdf[center_pixel]))
    
    return result


def local_histogram_equalization_optimized(image_array, block_size=8):
    # Input validation (same as standard version)
    if image_array is None or image_array.size == 0:
        raise ValueError("Input image is None or empty.")
    
    if len(image_array.shape) != 2:
        raise ValueError("Expected 2D grayscale image.")
    
    if not isinstance(block_size, (int, np.integer)) or block_size < 2:
        raise ValueError(f"Block size must be an integer ≥ 2, got {block_size}")
    
    block_size = int(block_size)
    rows, cols = image_array.shape
    
    if block_size > min(rows, cols):
        raise ValueError(f"Block size exceeds image dimensions.")
    
    # Initialize output
    result = np.zeros_like(image_array, dtype=np.float64)
    
    # Pad image
    pad_size = block_size // 2
    padded = np.pad(image_array.astype(np.float64), pad_size, mode='reflect')
    
    total_pixels = block_size * block_size
    
    # Process using rank-based equalization
    for i in range(rows):
        for j in range(cols):
            # Extract local block
            local_block = padded[i:i + block_size, j:j + block_size]
            
            # Sort to find ranks
            sorted_vals = np.sort(local_block.flatten())
            
            # Get center pixel value
            center_val = padded[i + pad_size, j + pad_size]
            
            # Find rank using binary search
            rank = np.searchsorted(sorted_vals, center_val, side='right')
            
            # Equalize based on rank
            # rank ranges from 1 to total_pixels (center pixel is included)
            result[i, j] = min(255.0, (rank / total_pixels) * 256.0)
    
    return np.clip(np.round(result), 0, 255).astype(np.uint8)