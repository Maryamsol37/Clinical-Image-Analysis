import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import ensure_binary


class TestEnsureBinary(unittest.TestCase):
    """Tests for ensure_binary function"""
    
    def setUp(self):
        """Create test images"""
        # Pure binary image
        self.binary_image = np.array([
            [0, 255, 0],
            [255, 0, 255],
            [0, 255, 0]
        ], dtype=np.uint8)
        
        # Grayscale image with various values
        self.grayscale_image = np.array([
            [0, 50, 100],
            [150, 200, 250],
            [1, 128, 255]
        ], dtype=np.uint8)
        
        # All zeros
        self.all_zeros = np.zeros((3, 3), dtype=np.uint8)
        
        # All 255
        self.all_255 = np.ones((3, 3), dtype=np.uint8) * 255
    
    def test_binary_unchanged(self):
        """Already binary image should stay the same"""
        result = ensure_binary(self.binary_image)
        np.testing.assert_array_equal(result, self.binary_image)
        print("✓ PASS: Binary input unchanged")
    
    def test_grayscale_to_binary(self):
        """Grayscale values should become binary"""
        result = ensure_binary(self.grayscale_image)
        expected = np.array([
            [0, 255, 255],
            [255, 255, 255],
            [255, 255, 255]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Grayscale converted to binary")
    
    def test_zeros_stay_zero(self):
        """Zero values should remain zero"""
        result = ensure_binary(self.all_zeros)
        np.testing.assert_array_equal(result, self.all_zeros)
        print("✓ PASS: Zeros stay zero")
    
    def test_255_stay_255(self):
        """255 values should remain 255"""
        result = ensure_binary(self.all_255)
        np.testing.assert_array_equal(result, self.all_255)
        print("✓ PASS: 255s stay 255")
    
    def test_value_1_becomes_255(self):
        """Value of 1 should become 255"""
        test = np.array([[1]], dtype=np.uint8)
        result = ensure_binary(test)
        self.assertEqual(result[0, 0], 255)
        print("✓ PASS: Value 1 → 255")
    
    def test_value_254_becomes_255(self):
        """Value of 254 should become 255"""
        test = np.array([[254]], dtype=np.uint8)
        result = ensure_binary(test)
        self.assertEqual(result[0, 0], 255)
        print("✓ PASS: Value 254 → 255")
    
    def test_value_128_becomes_255(self):
        """Value of 128 should become 255"""
        test = np.array([[128]], dtype=np.uint8)
        result = ensure_binary(test)
        self.assertEqual(result[0, 0], 255)
        print("✓ PASS: Value 128 → 255")
    
    def test_mixed_values(self):
        """All non-zero values become 255"""
        test = np.array([[0, 1, 2], [100, 200, 255], [50, 0, 75]], dtype=np.uint8)
        result = ensure_binary(test)
        expected = np.array([[0, 255, 255], [255, 255, 255], [255, 0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Mixed values correctly binarized")
    
    def test_output_is_binary(self):
        """Output should only contain 0 and 255"""
        test = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = ensure_binary(test)
        unique_vals = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_vals, [0, 255])))
        print("✓ PASS: Output contains only 0 and 255")
    
    def test_output_dtype(self):
        """Output should be uint8"""
        result = ensure_binary(self.grayscale_image)
        self.assertEqual(result.dtype, np.uint8)
        print("✓ PASS: Output dtype is uint8")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: ensure_binary")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnsureBinary)
    unittest.TextTestRunner(verbosity=2).run(suite)