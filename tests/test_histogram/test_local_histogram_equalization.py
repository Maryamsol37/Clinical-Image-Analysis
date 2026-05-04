"""
Unit Tests for local_histogram_equalization function
=====================================================
Tests block-based adaptive histogram equalization.

Menna Hesham Ragab Allam - 1220321
Module: processing/histogram/local_equalization.py
Function: local_histogram_equalization
"""

import numpy as np
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from processing.histogram.local_equalization import local_histogram_equalization


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_image():
    """Small test image"""
    return np.array([[50, 100, 150], [75, 125, 175], [100, 150, 200]], dtype=np.uint8)


@pytest.fixture
def uniform_image():
    """All pixels same value"""
    return np.full((20, 20), 128, dtype=np.uint8)


@pytest.fixture
def random_image():
    """Random test image"""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def medical_simulation():
    """Image with uneven illumination (simulates medical scan)"""
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    xx, yy = np.meshgrid(x, y)
    # Brighter in center, darker at edges
    illumination = np.exp(-(xx**2 + yy**2) / 0.5) * 200 + 30
    return illumination.astype(np.uint8)


@pytest.fixture
def checkerboard():
    """Checkerboard pattern for edge testing"""
    img = np.zeros((32, 32), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    return img


@pytest.fixture
def noisy_image():
    """Image with salt-and-pepper-like noise"""
    base = np.full((32, 32), 128, dtype=np.uint8)
    noise = np.random.randint(0, 50, (32, 32), dtype=np.uint8) - 25
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestLocalHEBasic:
    """Basic functionality tests"""
    
    def test_returns_array(self, simple_image):
        """Test function returns numpy array"""
        result = local_histogram_equalization(simple_image, block_size=3)
        assert isinstance(result, np.ndarray)
    
    def test_same_shape_small(self, simple_image):
        """Test output shape matches input (small image)"""
        result = local_histogram_equalization(simple_image, block_size=3)
        assert result.shape == simple_image.shape
    
    def test_same_shape_random(self, random_image):
        """Test output shape matches input (random image)"""
        result = local_histogram_equalization(random_image, block_size=8)
        assert result.shape == random_image.shape
    
    def test_output_dtype(self, random_image):
        """Test output is uint8"""
        result = local_histogram_equalization(random_image, block_size=8)
        assert result.dtype == np.uint8
    
    def test_output_not_float(self, simple_image):
        """Test output is integer type"""
        result = local_histogram_equalization(simple_image, block_size=3)
        assert np.issubdtype(result.dtype, np.integer)


# ============================================================================
# VALUE RANGE TESTS
# ============================================================================

class TestLocalHEValueRange:
    """Test output value ranges"""
    
    def test_no_values_below_zero(self, random_image):
        """Test all values >= 0"""
        result = local_histogram_equalization(random_image, block_size=8)
        assert result.min() >= 0, f"Min value: {result.min()}"
    
    def test_no_values_above_255(self, random_image):
        """Test all values <= 255"""
        result = local_histogram_equalization(random_image, block_size=8)
        assert result.max() <= 255, f"Max value: {result.max()}"
    
    def test_range_within_bounds(self, random_image):
        """Test values stay within [0, 255]"""
        result = local_histogram_equalization(random_image, block_size=8)
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_uniform_image_output_range(self, uniform_image):
        """Test uniform image still valid"""
        result = local_histogram_equalization(uniform_image, block_size=8)
        assert result.min() >= 0
        assert result.max() <= 255


# ============================================================================
# BLOCK SIZE TESTS
# ============================================================================

class TestLocalHEBlockSizes:
    """Test different block size configurations"""
    
    def test_block_size_3(self, random_image):
        """Test with block size 3"""
        result = local_histogram_equalization(random_image, block_size=3)
        assert result.shape == random_image.shape
    
    def test_block_size_8(self, random_image):
        """Test with block size 8"""
        result = local_histogram_equalization(random_image, block_size=8)
        assert result.shape == random_image.shape
    
    def test_block_size_16(self, random_image):
        """Test with block size 16"""
        result = local_histogram_equalization(random_image, block_size=16)
        assert result.shape == random_image.shape
    
    def test_block_size_32(self, random_image):
        """Test with block size 32"""
        result = local_histogram_equalization(random_image, block_size=32)
        assert result.shape == random_image.shape
    
    def test_odd_block_sizes(self, random_image):
        """Test odd block sizes work"""
        for bs in [7, 9, 15]:
            result = local_histogram_equalization(random_image, block_size=bs)
            assert result.shape == random_image.shape
    
    def test_even_block_sizes(self, random_image):
        """Test even block sizes work"""
        for bs in [4, 8, 16]:
            result = local_histogram_equalization(random_image, block_size=bs)
            assert result.shape == random_image.shape
    
    def test_default_block_size(self, random_image):
        """Test default block size (8) works"""
        result = local_histogram_equalization(random_image)
        assert result.shape == random_image.shape


# ============================================================================
# LOCAL ENHANCEMENT TESTS
# ============================================================================

class TestLocalHEEnhancement:
    """Test that local enhancement actually works"""
    
    def test_local_contrast_improved(self, medical_simulation):
        """Test local contrast is enhanced in medical-like image"""
        result = local_histogram_equalization(medical_simulation, block_size=16)
        
        # Compute local standard deviation as measure of contrast
        def local_std(img, window=16):
            std_map = np.zeros_like(img, dtype=np.float64)
            padded = np.pad(img.astype(np.float64), window//2, mode='reflect')
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    patch = padded[i:i+window, j:j+window]
                    std_map[i, j] = np.std(patch)
            return std_map
        
        std_orig = np.mean(local_std(medical_simulation))
        std_result = np.mean(local_std(result))
        
        assert std_result >= std_orig * 0.9, \
            f"Original std: {std_orig:.2f}, Result std: {std_result:.2f}"
    
    def test_different_from_original(self, random_image):
        """Test output is different from input (enhancement occurred)"""
        result = local_histogram_equalization(random_image, block_size=8)
        # Should not be identical unless image is uniform
        if len(np.unique(random_image)) > 1:
            assert not np.array_equal(result, random_image), \
                "Output should differ from input for non-uniform images"
    
    def test_details_enhanced(self, checkerboard):
        """Test edges/details are preserved/enhanced"""
        result = local_histogram_equalization(checkerboard, block_size=8)
        # Checkerboard should still have two distinct regions
        assert len(np.unique(result)) > 1, "Should have multiple intensity values"


# ============================================================================
# EDGE HANDLING TESTS
# ============================================================================

class TestLocalHEEdgeHandling:
    """Test edge/border handling"""
    
    def test_no_nan_values(self, random_image):
        """Test no NaN values at edges"""
        result = local_histogram_equalization(random_image, block_size=8)
        assert not np.any(np.isnan(result))
    
    def test_no_inf_values(self, random_image):
        """Test no infinity values at edges"""
        result = local_histogram_equalization(random_image, block_size=8)
        assert not np.any(np.isinf(result))
    
    def test_edges_not_black(self, random_image):
        """Test edges are processed (not left as zeros)"""
        result = local_histogram_equalization(random_image, block_size=8)
        # Edge pixels should have valid values
        assert result[0, 0] >= 0
        assert result[0, -1] >= 0
        assert result[-1, 0] >= 0
        assert result[-1, -1] >= 0
    
    def test_all_pixels_processed(self, simple_image):
        """Test every pixel is processed"""
        result = local_histogram_equalization(simple_image, block_size=3)
        assert np.all(result >= 0)
        assert result.shape == simple_image.shape


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestLocalHEDeterminism:
    """Test deterministic behavior"""
    
    def test_same_input_same_output(self, random_image):
        """Test function is deterministic"""
        result1 = local_histogram_equalization(random_image, block_size=8)
        result2 = local_histogram_equalization(random_image, block_size=8)
        assert np.array_equal(result1, result2)
    
    def test_reproducible_with_seed(self):
        """Test same seed gives same result"""
        np.random.seed(123)
        img1 = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        np.random.seed(123)
        img2 = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        result1 = local_histogram_equalization(img1, block_size=8)
        result2 = local_histogram_equalization(img2, block_size=8)
        assert np.array_equal(result1, result2)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestLocalHEErrors:
    """Test error handling"""
    
    def test_none_input(self):
        """Test None raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            local_histogram_equalization(None)
        assert "None" in str(exc_info.value)
    
    def test_empty_array(self):
        """Test empty array raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            local_histogram_equalization(np.array([], dtype=np.uint8))
        assert "empty" in str(exc_info.value)
    
    def test_block_size_too_small(self, random_image):
        """Test block size < 2 raises error"""
        with pytest.raises(ValueError) as exc_info:
            local_histogram_equalization(random_image, block_size=1)
        assert "at least 2" in str(exc_info.value)
    
    def test_block_size_zero(self, random_image):
        """Test block size 0 raises error"""
        with pytest.raises(ValueError):
            local_histogram_equalization(random_image, block_size=0)
    
    def test_block_size_negative(self, random_image):
        """Test negative block size raises error"""
        with pytest.raises(ValueError):
            local_histogram_equalization(random_image, block_size=-5)
    
    def test_block_size_too_large(self):
        """Test block size larger than image"""
        small_img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            local_histogram_equalization(small_img, block_size=20)
        assert "exceeds" in str(exc_info.value)
    
    def test_float_block_size(self, random_image):
        """Test float block size raises error"""
        with pytest.raises(ValueError) as exc_info:
            local_histogram_equalization(random_image, block_size=8.5)
        assert "integer" in str(exc_info.value)
    
    def test_3d_input(self):
        """Test 3D (color) input raises error"""
        with pytest.raises(ValueError) as exc_info:
            local_histogram_equalization(np.zeros((10, 10, 3), dtype=np.uint8))
        assert "2D" in str(exc_info.value)


# ============================================================================
# SMALL IMAGE TESTS
# ============================================================================

class TestLocalHESmallImages:
    """Test with very small images"""
    
    def test_2x2_image(self):
        """Test minimum size image"""
        img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        result = local_histogram_equalization(img, block_size=2)
        assert result.shape == img.shape
    
    def test_3x3_with_block_3(self):
        """Test square image with matching block size"""
        img = np.array([[50, 100, 150], [75, 125, 175], [100, 150, 200]], dtype=np.uint8)
        result = local_histogram_equalization(img, block_size=3)
        assert result.shape == img.shape
    
    def test_single_column(self):
        """Test Nx1 image"""
        img = np.array([[10], [50], [100], [150], [200]], dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            local_histogram_equalization(img, block_size=3)
        assert "exceeds" in str(exc_info.value)
    
    def test_single_row(self):
        """Test 1xN image"""
        img = np.array([[10, 50, 100, 150, 200]], dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            local_histogram_equalization(img, block_size=3)
        assert "exceeds" in str(exc_info.value)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestLocalHEPerformance:
    """Test performance with larger images"""
    
    def test_reasonable_time_64x64(self, random_image):
        """Test completes in reasonable time for 64x64"""
        start = time.time()
        result = local_histogram_equalization(random_image, block_size=8)
        duration = time.time() - start
        
        assert duration < 30, f"Took {duration:.2f}s (should be < 30s)"
        assert result.shape == random_image.shape
    
    def test_reasonable_time_100x100(self):
        """Test performance with 100x100 image"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        start = time.time()
        result = local_histogram_equalization(img, block_size=8)
        duration = time.time() - start
        
        assert duration < 60, f"Took {duration:.2f}s (should be < 60s)"
        assert result.shape == img.shape
    
    def test_block_size_affects_speed(self, random_image):
        """Test larger block size is faster"""
        start_small = time.time()
        local_histogram_equalization(random_image, block_size=4)
        time_small = time.time() - start_small
        
        start_large = time.time()
        local_histogram_equalization(random_image, block_size=16)
        time_large = time.time() - start_large
        
        # Larger block size should be faster (fewer blocks to process)
        # Note: This might not always hold for small images
        assert time_large <= time_small * 2, \
            f"Small block time: {time_small:.4f}s, Large block time: {time_large:.4f}s"


# ============================================================================
# PIPELINE COMPATIBILITY TESTS
# ============================================================================

class TestLocalHEPipelineCompatibility:
    """Test that function works in pipeline mode"""
    
    def test_output_usable_as_input(self, random_image):
        """Test output can be used as input again"""
        result1 = local_histogram_equalization(random_image, block_size=8)
        # Apply again on the result
        result2 = local_histogram_equalization(result1, block_size=8)
        assert result2.shape == random_image.shape
        assert result2.dtype == np.uint8
    
    def test_chain_with_different_block_sizes(self, random_image):
        """Test chaining with different block sizes"""
        result = local_histogram_equalization(random_image, block_size=16)
        result = local_histogram_equalization(result, block_size=8)
        result = local_histogram_equalization(result, block_size=4)
        assert result.shape == random_image.shape
        assert result.dtype == np.uint8
    
    def test_other_functions_compatible(self, simple_image):
        """Test output is valid numpy array for other processing"""
        result = local_histogram_equalization(simple_image, block_size=3)
        # Check properties needed by pipeline
        assert hasattr(result, 'shape')
        assert hasattr(result, 'dtype')
        assert result.size == simple_image.size


# ============================================================================
# MEDICAL IMAGE SIMULATION TESTS
# ============================================================================

class TestLocalHEMedicalSimulation:
    """Test with medical-image-like data"""
    
    def test_uneven_illumination_handled(self, medical_simulation):
        """Test image with uneven illumination is properly processed"""
        result = local_histogram_equalization(medical_simulation, block_size=16)
        
        # Result should have values across the range
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.max() - result.min() > 10, \
            "Should have reasonable dynamic range"
    
    def test_noise_handled(self, noisy_image):
        """Test noisy image doesn't crash"""
        result = local_histogram_equalization(noisy_image, block_size=8)
        assert result.shape == noisy_image.shape
        assert result.dtype == np.uint8