import numpy as np
def _to_gray_float(arr: np.ndarray) -> np.ndarray:
    """
    Convert an image array to a 2-D float64 grayscale array.
    Handles both grayscale (H, W) and RGB (H, W, 3) inputs.
    Uses the standard luminance weights — implemented manually.
    """
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Luminance formula: Y = 0.2989·R + 0.5870·G + 0.1140·B
        return (0.2989 * arr[:, :, 0].astype(np.float64)
                + 0.5870 * arr[:, :, 1].astype(np.float64)
                + 0.1140 * arr[:, :, 2].astype(np.float64))
    raise ValueError(f"Unsupported image shape: {arr.shape}")

def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Return a (H, W, 3) uint8 array regardless of whether input is gray or RGB."""
    if arr.ndim == 2:
        channel = np.clip(arr, 0, 255).astype(np.uint8)
        return np.stack([channel, channel, channel], axis=-1)
    return np.clip(arr, 0, 255).astype(np.uint8)

def _draw_rect(image_rgb: np.ndarray,
               r1: int, c1: int, r2: int, c2: int,
               color=(255, 0, 0), thickness: int = 2) -> None:
    """
    Draw a filled-border rectangle on a (H, W, 3) uint8 array **in-place**.
   
    """
    ih, iw = image_rgb.shape[:2]

    r1 = max(r1, 0)
    c1 = max(c1, 0)
    r2 = min(r2, ih - 1)
    c2 = min(c2, iw - 1)

    for t in range(thickness):
        # Horizontal edges (top / bottom)
        if r1 + t < ih:
            image_rgb[r1 + t, c1:c2 + 1] = color
        if r2 - t >= 0:
            image_rgb[r2 - t, c1:c2 + 1] = color
        # Vertical edges (left / right)
        if c1 + t < iw:
            image_rgb[r1:r2 + 1, c1 + t] = color
        if c2 - t >= 0:
            image_rgb[r1:r2 + 1, c2 - t] = color



