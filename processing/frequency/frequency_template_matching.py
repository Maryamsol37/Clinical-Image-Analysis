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


def fourier_cross_correlate(image: np.ndarray, template: np.ndarray):
    """
    Locate `template` inside `image` using Fourier cross-correlation.

    Returns
    -------
    result_image  : (H, W, 3) uint8 — original image with red bounding box
    norm_corr_map : (H, W) float64  — normalised correlation map in [0, 1]
    peak          : (row, col)      — top-left corner of best match
    (th, tw)      : template height and width
    """

    # ── 1. Grayscale float ────────────────────────────────────────────────────
    gray_image    = _to_gray_float(image)
    gray_template = _to_gray_float(template)

    ih, iw = gray_image.shape
    th, tw = gray_template.shape

    if th > ih or tw > iw:
        raise ValueError(
            f"Template ({th}×{tw}) is larger than the image ({ih}×{iw})."
        )
    if th < 2 or tw < 2:
        raise ValueError("Template must be at least 2×2 pixels.")

    # ── 2. Padded size (linear cross-correlation — no wrap-around) ────────────
    pad_rows = ih + th - 1
    pad_cols = iw + tw - 1

    # ── 3. Zero-pad BOTH to the same padded size ──────────────────────────────
    image_padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    image_padded[:ih, :iw] = gray_image          # ← gray_image, not image

    template_padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    template_padded[:th, :tw] = gray_template    # ← same padded size as image

    # ── 4. Forward FFTs ───────────────────────────────────────────────────────
    F_image    = np.fft.fft2(image_padded)        # shape: (pad_rows, pad_cols)
    F_template = np.fft.fft2(template_padded)     # shape: (pad_rows, pad_cols) ✓

    # ── 5. Cross-correlation spectrum: C = F · conj(G) ───────────────────────
    corr_spectrum = F_image * np.conj(F_template)

    # ── 6. Inverse FFT → full correlation map ────────────────────────────────
    correlation_map = np.real(np.fft.ifft2(corr_spectrum))
    # correlation_map.shape = (pad_rows, pad_cols)

    # ── 7. Crop to valid region before searching for the peak ─────────────────
    # A valid match top-left must sit at row in [0, ih) and col in [0, iw)
    # so the template still fits inside the image.
    valid_map = correlation_map[:ih, :iw]

    # Normalise valid map to [0, 1] for display
    v_min, v_max = valid_map.min(), valid_map.max()
    if v_max - v_min > 1e-10:
        norm_corr_map = (valid_map - v_min) / (v_max - v_min)
    else:
        norm_corr_map = np.zeros_like(valid_map)

    # Peak on the valid region (raw values, not normalised)
    peak_flat = int(np.argmax(valid_map))
    peak_row, peak_col = np.unravel_index(peak_flat, valid_map.shape)

    # ── 8. Draw bounding box on a copy of the original image ─────────────────
    result_image = _to_rgb_uint8(image).copy()
    r1, c1 = int(peak_row), int(peak_col)
    r2, c2 = r1 + th - 1, c1 + tw - 1
    _draw_rect(result_image, r1, c1, r2, c2, color=(255, 0, 0), thickness=2)

    return result_image, norm_corr_map, (peak_row, peak_col), (th, tw)