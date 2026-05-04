"""
Unit Tests for compute_histogram function
==========================================
Tests histogram computation from scratch.

Menna Hesham Ragab Allam - 1220321
Module: processing/histogram/local_equalization.py
Function: compute_histogram
"""

import numpy as np
import pytest
import sys
import os
    
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from processing.histogram.local_equalization import compute_histogram


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_image():
    """2x2 test image with known values"""
    return np.array([[100, 150], [200, 100]], dtype=np.uint8)


@pytest.fixture
def uniform_image():
    """Image where all pixels have the same value"""
    return np.full((10, 10), 128, dtype=np.uint8)


@pytest.fixture
def checkerboard_image():
    """Black and white checkerboard pattern"""
    img = np.zeros((8, 8), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    return img


@pytest.fixture
def random_image():
    """Random noise image for stress testing"""
    np.random.seed(42)
    return np.random.randint(0, 256, (50, 50), dtype=np.uint8)


@pytest.fixture
def edge_value_image():
    """Image with min and max intensity values"""
    return np.array([[0, 255], [255, 0]], dtype=np.uint8)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestComputeHistogramBasic:
    """Test basic histogram computation functionality"""
    
    def test_returns_tuple(self, simple_image):
        """Test function returns two arrays"""
        result = compute_histogram(simple_image)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_histogram_type(self, simple_image):
        """Test histogram is numpy array"""
        hist, _ = compute_histogram(simple_image)
        assert isinstance(hist, np.ndarray)
    
    def test_edges_type(self, simple_image):
        """Test bin edges is numpy array"""
        _, edges = compute_histogram(simple_image)
        assert isinstance(edges, np.ndarray)
    
    def test_histogram_shape_default_bins(self, simple_image):
        """Test histogram shape with default 256 bins"""
        hist, _ = compute_histogram(simple_image)
        assert hist.shape == (256,)
    
    def test_edges_shape_default_bins(self, simple_image):
        """Test edges shape with default 256 bins"""
        _, edges = compute_histogram(simple_image)
        assert edges.shape == (257,)


# ============================================================================
# ACCURACY TESTS
# ============================================================================

class TestComputeHistogramAccuracy:
    """Test histogram values are correct"""
    
    def test_known_values(self, simple_image):
        """Test histogram counts match known pixel values"""
        hist, _ = compute_histogram(simple_image)
        
        # Image is: [[100, 150], [200, 100]]
        assert hist[100] == 2  # Two occurrences
        assert hist[150] == 1  # One occurrence
        assert hist[200] == 1  # One occurrence
    
    def test_total_count(self, simple_image):
        """Test sum of all bins equals total pixels"""
        hist, _ = compute_histogram(simple_image)
        assert hist.sum() == 4  # 2x2 image = 4 pixels
    
    def test_uniform_image(self, uniform_image):
        """Test all pixels counted in single bin"""
        hist, _ = compute_histogram(uniform_image)
        assert hist[128] == 100  # 10x10 = 100 pixels
        assert hist.sum() == 100
    
    def test_other_bins_zero_for_uniform(self, uniform_image):
        """Test non-matching bins are zero"""
        hist, _ = compute_histogram(uniform_image)
        assert hist[0] == 0
        assert hist[127] == 0
        assert hist[129] == 0
        assert hist[255] == 0
    
    def test_checkerboard_black_count(self, checkerboard_image):
        """Test black pixel count in checkerboard"""
        hist, _ = compute_histogram(checkerboard_image)
        assert hist[0] == 32  # Half of 8x8
    
    def test_checkerboard_white_count(self, checkerboard_image):
        """Test white pixel count in checkerboard"""
        hist, _ = compute_histogram(checkerboard_image)
        assert hist[255] == 32  # Half of 8x8
    
    def test_edge_values(self, edge_value_image):
        """Test min and max intensity values"""
        hist, _ = compute_histogram(edge_value_image)
        assert hist[0] == 2
        assert hist[255] == 2
        assert hist.sum() == 4
    
    def test_total_matches_image_size(self, random_image):
        """Test histogram sum equals image.size"""
        hist, _ = compute_histogram(random_image)
        assert hist.sum() == random_image.size
    
    def test_non_negative_counts(self, random_image):
        """Test all bin counts are non-negative"""
        hist, _ = compute_histogram(random_image)
        assert np.all(hist >= 0)
    
    def test_no_negative_counts(self, random_image):
        """Test no count is below zero"""
        hist, _ = compute_histogram(random_image)
        assert hist.min() >= 0


# ============================================================================
# BIN EDGES TESTS
# ============================================================================

class TestComputeHistogramEdges:
    """Test bin edges properties"""
    
    def test_edges_start_at_zero(self, simple_image):
        """Test first edge is 0"""
        _, edges = compute_histogram(simple_image)
        assert edges[0] == 0
    
    def test_edges_end_at_bins(self, simple_image):
        """Test last edge equals number of bins"""
        _, edges = compute_histogram(simple_image)
        assert edges[-1] == 256
    
    def test_edges_are_monotonic(self, simple_image):
        """Test edges are strictly increasing"""
        _, edges = compute_histogram(simple_image)
        assert np.all(np.diff(edges) > 0)
    
    def test_edges_are_equally_spaced(self, simple_image):
        """Test edges have equal spacing"""
        _, edges = compute_histogram(simple_image)
        spacing = np.diff(edges)
        assert np.all(spacing == 1)
    
    def test_edges_length(self, simple_image):
        """Test edges length is bins + 1"""
        _, edges = compute_histogram(simple_image)
        assert len(edges) == 257


# ============================================================================
# CUSTOM BINS TESTS
# ============================================================================

class TestComputeHistogramCustomBins:
    """Test with different bin counts"""
    
    def test_128_bins_shape(self, simple_image):
        """Test histogram with 128 bins"""
        hist, edges = compute_histogram(simple_image, bins=128)
        assert hist.shape == (128,)
        assert edges.shape == (129,)
    
    def test_64_bins_shape(self, simple_image):
        """Test histogram with 64 bins"""
        hist, edges = compute_histogram(simple_image, bins=64)
        assert hist.shape == (64,)
        assert edges.shape == (65,)
    
    def test_custom_bins_total_count(self, random_image):
        """Test total count preserved with custom bins"""
        hist, _ = compute_histogram(random_image, bins=128)
        assert hist.sum() == random_image.size
    
    def test_2_bins(self, simple_image):
        """Test with minimum meaningful bins"""
        hist, edges = compute_histogram(simple_image, bins=2)
        assert hist.shape == (2,)
        assert edges.shape == (3,)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestComputeHistogramErrors:
    """Test error handling and edge cases"""
    
    def test_none_input(self):
        """Test None raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            compute_histogram(None)
        assert "None" in str(exc_info.value)
    
    def test_empty_array(self):
        """Test empty array raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            compute_histogram(np.array([], dtype=np.uint8))
        assert "empty" in str(exc_info.value)
    
    def test_zero_size_array(self):
        """Test 0x0 array raises ValueError"""
        with pytest.raises(ValueError):
            compute_histogram(np.zeros((0, 0), dtype=np.uint8))
    
    def test_empty_2d_array(self):
        """Test 2D array with no elements raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            compute_histogram(np.array([[]], dtype=np.uint8))
        assert "empty" in str(exc_info.value)


# ============================================================================
# DATA TYPE TESTS
# ============================================================================

class TestComputeHistogramDtype:
    """Test handling of different data types"""
    
    def test_uint8_input(self, simple_image):
        """Test standard uint8 input works"""
        hist, _ = compute_histogram(simple_image)
        assert hist.dtype == np.int64
    
    def test_histogram_dtype_is_int(self, random_image):
        """Test histogram values are integers"""
        hist, _ = compute_histogram(random_image)
        assert np.issubdtype(hist.dtype, np.integer)
    
    def test_edges_dtype_is_int(self, random_image):
        """Test edge values are integers"""
        _, edges = compute_histogram(random_image)
        assert np.issubdtype(edges.dtype, np.integer)


# ============================================================================
# SINGLE PIXEL TESTS
# ============================================================================

class TestComputeHistogramSinglePixel:
    """Test single pixel edge cases"""
    
    def test_single_pixel_image(self):
        """Test 1x1 image"""
        img = np.array([[128]], dtype=np.uint8)
        hist, _ = compute_histogram(img)
        assert hist[128] == 1
        assert hist.sum() == 1
    
    def test_single_row_image(self):
        """Test 1xN image"""
        img = np.array([[10, 20, 30, 40]], dtype=np.uint8)
        hist, _ = compute_histogram(img)
        assert hist.sum() == 4
        assert hist[10] == 1
        assert hist[20] == 1
        assert hist[30] == 1
        assert hist[40] == 1
    
    def test_single_column_image(self):
        """Test Nx1 image"""
        img = np.array([[10], [20], [30], [40]], dtype=np.uint8)
        hist, _ = compute_histogram(img)
        assert hist.sum() == 4
        assert hist[10] == 1
        assert hist[20] == 1
        assert hist[30] == 1
        assert hist[40] == 1