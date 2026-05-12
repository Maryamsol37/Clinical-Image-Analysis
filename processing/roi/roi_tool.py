import numpy as np
"(x1, y1) top-left corner, (x2, y2) bottom-right corner"
def extract_roi(image, x1, y1, x2, y2):
    "as8r value lel row ely fo2"
    row_start = min(y1, y2) 
    "akbr value lelb row ely ta7t"
    row_end   = max(y1, y2) 
    "most left"
    col_start = min(x1, x2) 
    "most right"
    col_end   = max(x1, x2)

    "error handling: clamp to image bounderies"
    row_start = max(0, row_start)

    "image.shape[0] ->S the number of rows of the image ely howa el height"
    row_end   = min(image.shape[0], row_end)

    col_start = max(0, col_start)

    "image.shape[1] ->S the number of columns of the image ely howa el width"
    col_end   = min(image.shape[1], col_end)

    # Guard: if the selection is too tiny, return original
    if row_end - row_start < 2 or col_end - col_start < 2:
        raise ValueError("ROI too small. Draw a larger rectangle.")

    return image[row_start:row_end, col_start:col_end]


def draw_roi_on_image(image, x1, y1, x2, y2, color=255):
    """
    Draws a rectangle border on a copy of the image to show the ROI.
    """
    "h3ml copy 3shan pixels bt3t original image my7slhash destortion "
    result = image.copy()
    row_start = min(y1, y2)
    row_end   = max(y1, y2)
    col_start = min(x1, x2)
    col_end   = max(x1, x2)

    # Clamp to image boundaries
    row_start = max(0, row_start)
    row_end   = min(image.shape[0] - 1, row_end)
    col_start = max(0, col_start)
    col_end   = min(image.shape[1] - 1, col_end)

    "Draw 4 edges of the rectangle"
    result[row_start, col_start:col_end] = color   # top edge
    result[row_end,   col_start:col_end] = color   # bottom edge
    result[row_start:row_end, col_start] = color   # left edge
    result[row_start:row_end, col_end]   = color   # right edge

    return result