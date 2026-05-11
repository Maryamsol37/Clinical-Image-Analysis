import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import global_threshold


class TestGlobalThreshold(unittest.TestCase):
    """Tests for global_threshold function"""
    
    def setUp(self):
        """Create test images"""
        # 4x4 test image with known values
        self.test_image = np.array([
            [50, 100, 150, 200],
            [75, 125, 175, 225],
            [100, 150, 200, 250],
            [125, 175, 225, 255]
        ], dtype=np.uint8)
        
        # Gradient image
        self.gradient = np.array([
            [0, 85, 170, 255],
            [0, 85, 170, 255],
            [0, 85, 170, 255],
            [0, 85, 170, 255]
        ], dtype=np.uint8)
        
        # Color image
        self.color_image = np.zeros((4, 4, 3), dtype=np.uint8)
        self.color_image[:, :, 0] = 100
    
    def test_threshold_128(self):
        """Threshold at 128 - middle value"""
        result = global_threshold(self.test_image, 128)
        expected = np.array([
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 255, 255, 255],
            [0, 255, 255, 255]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Threshold at 128 correct")
    
    def test_threshold_0(self):
        """Threshold at 0 - minimum value"""
        result = global_threshold(self.test_image, 0)
        expected = np.ones_like(self.test_image) * 255
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Threshold at 0 gives all white")
    
    def test_threshold_255(self):
        """Threshold at 255 - maximum value"""
        result = global_threshold(self.test_image, 255)
        expected = np.zeros_like(self.test_image)
        expected[self.test_image >= 255] = 255
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Threshold at 255 correct")
    
    def test_threshold_exact_values(self):
        """Test pixels exactly at threshold value become white"""
        test = np.array([[128]], dtype=np.uint8)
        result = global_threshold(test, 128)
        self.assertEqual(result[0, 0], 255)
        print("✓ PASS: Pixels at threshold value become white")
    
    def test_threshold_below_value(self):
        """Test pixels below threshold become black"""
        test = np.array([[127]], dtype=np.uint8)
        result = global_threshold(test, 128)
        self.assertEqual(result[0, 0], 0)
        print("✓ PASS: Pixels below threshold become black")
    
    def test_threshold_above_value(self):
        """Test pixels above threshold become white"""
        test = np.array([[129]], dtype=np.uint8)
        result = global_threshold(test, 128)
        self.assertEqual(result[0, 0], 255)
        print("✓ PASS: Pixels above threshold become white")
    
    def test_threshold_with_color_image(self):
        """Threshold should work with color images (convert first)"""
        result = global_threshold(self.color_image, 50)
        self.assertEqual(result.shape, (4, 4))
        print("✓ PASS: Color image threshold produces correct shape")
    
    def test_output_is_binary(self):
        """Output should only contain 0 and 255"""
        result = global_threshold(self.test_image, 128)
        unique_values = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_values, [0, 255])))
        print("✓ PASS: Output contains only 0 and 255")
    
    def test_output_dtype(self):
        """Output should be uint8"""
        result = global_threshold(self.test_image, 128)
        self.assertEqual(result.dtype, np.uint8)
        print("✓ PASS: Output dtype is uint8")
    
    def test_different_thresholds(self):
        """Test various threshold values"""
        test = np.array([[100]], dtype=np.uint8)
        
        # Threshold below pixel value
        result = global_threshold(test, 50)
        self.assertEqual(result[0, 0], 255)
        print("✓ PASS: Threshold 50 on value 100 → white")
        
        # Threshold equal to pixel value
        result = global_threshold(test, 100)
        self.assertEqual(result[0, 0], 255)
        print("✓ PASS: Threshold 100 on value 100 → white")
        
        # Threshold above pixel value
        result = global_threshold(test, 150)
        self.assertEqual(result[0, 0], 0)
        print("✓ PASS: Threshold 150 on value 100 → black")
    
    def test_threshold_on_uniform_image(self):
        """Test threshold on image with uniform values"""
        uniform = np.ones((5, 5), dtype=np.uint8) * 128
        result = global_threshold(uniform, 129)
        expected = np.zeros((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Uniform image threshold correct")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: global_threshold")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGlobalThreshold)
    unittest.TextTestRunner(verbosity=2).run(suite)