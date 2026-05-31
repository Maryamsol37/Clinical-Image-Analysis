# Clinical-Image-Analysis

Clinical Image Analysis Workbench is a desktop GUI for exploring medical and general grayscale images. It focuses on common image processing tasks (filtering, histogram equalization, morphology, frequency domain tools, and template matching) with a simple workflow and visual feedback.

## Features
- Dual image viewer (original vs processed) with fit/actual view and zoom
- Pipeline mode for stacking operations, with undo and reset
- Filtering: average, gaussian, median, sobel edge detection
- Histogram enhancement: local histogram equalization
- Geometry: rotation and shearing
- Morphology: thresholding, erosion, dilation, opening, closing, boundary extraction
- Noise modeling: gaussian and uniform
- ROI tools: draw ROI, isolate ROI, and view ROI statistics
- Template matching with Fourier cross-correlation and correlation map
- Frequency domain viewer with FFT magnitude and notch filtering
- Metadata viewer for DICOM tags and image properties

## Supported Formats
Load:
- DICOM (.dcm)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)

Save:
- JPEG (.jpg)
- BMP (.bmp)
- PNG (.png)

## Quick Start
1) Create and activate a virtual environment.
2) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3) Run the app:

```bash
python main.py
```

Sample DICOM files are available under assets/dicom.

## Usage Notes
- Pipeline mode ON applies each operation on the current processed image.
- Pipeline mode OFF applies operations on the original image.
- Morphology operations always use the current processed image (use binarize first).
- In the Frequency Domain tab, click on the spectrum to add notch points, then apply the filter.
- In Template Matching, draw a crop region to set the template, then run cross-correlation.

## Project Layout
- gui/ - CustomTkinter application and widgets
- image_io/ - DICOM and regular image loading + metadata formatting
- processing/ - Image processing algorithms (filtering, morphology, frequency, ROI, etc.)
- pipeline/ - Operation pipeline manager (stack, undo, reset)
- tests/ - Unit tests for processing components
- assets/ - Sample DICOM and image files

## Tests
If you have pytest installed:

```bash
pytest
```
