import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import pad_binary_image



class TestPadBinaryImage(unittest.TestCase):
    """Tests for pad_binary_image function"""
    
    def setUp(self):
        """Create test images"""
        # Simple 3x3 binary image
        self.binary_3x3 = np.array([
            [255, 0, 255],
            [0, 255, 0],
            [255, 0, 255]
        ], dtype=np.uint8)
        
        # Single pixel
        self.single_pixel = np.array([[255]], dtype=np.uint8)
        
        # 2x2 image
        self.binary_2x2 = np.array([
            [255, 0],
            [0, 255]
        ], dtype=np.uint8)
    
    def test_pad_1_3x3(self):
        """Pad 3x3 image with 1 pixel"""
        result = pad_binary_image(self.binary_3x3, 1, 1)
        
        # Expected shape: 5x5
        self.assertEqual(result.shape, (5, 5))
        
        # Original image should be in center
        np.testing.assert_array_equal(result[1:4, 1:4], self.binary_3x3)
        
        # Border should be zeros
        self.assertTrue(np.all(result[0, :] == 0))    # Top row
        self.assertTrue(np.all(result[-1, :] == 0))   # Bottom row
        self.assertTrue(np.all(result[:, 0] == 0))    # Left column
        self.assertTrue(np.all(result[:, -1] == 0))   # Right column
        
        print("✓ PASS: Pad 1 on 3x3")
    
    def test_pad_2_3x3(self):
        """Pad 3x3 image with 2 pixels"""
        result = pad_binary_image(self.binary_3x3, 2, 2)
        
        # Expected shape: 7x7
        self.assertEqual(result.shape, (7, 7))
        
        # Original image should be in center
        np.testing.assert_array_equal(result[2:5, 2:5], self.binary_3x3)
        
        print("✓ PASS: Pad 2 on 3x3")
    
    def test_pad_asymmetric(self):
        """Pad with different horizontal and vertical padding"""
        result = pad_binary_image(self.binary_3x3, 1, 2)
        
        # Expected: height = 3+2=5, width = 3+4=7
        self.assertEqual(result.shape, (5, 7))
        
        print("✓ PASS: Asymmetric padding")
    
    def test_pad_single_pixel(self):
        """Pad single pixel image"""
        result = pad_binary_image(self.single_pixel, 1, 1)
        
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result[1, 1], 255)
        
        # All border pixels should be 0
        result[1, 1] = 0  # Temporarily remove center
        self.assertTrue(np.all(result == 0))
        
        print("✓ PASS: Pad single pixel")
    
    def test_pad_zero_padding(self):
        """Pad with 0 (should return original)"""
        result = pad_binary_image(self.binary_3x3, 0, 0)
        np.testing.assert_array_equal(result, self.binary_3x3)
        print("✓ PASS: Zero padding returns original")
    
    def test_pad_3_2x2(self):
        """Pad 2x2 image with 3 pixels"""
        result = pad_binary_image(self.binary_2x2, 3, 3)
        
        # Expected shape: 8x8
        self.assertEqual(result.shape, (8, 8))
        
        # Original should be in center
        np.testing.assert_array_equal(result[3:5, 3:5], self.binary_2x2)
        
        print("✓ PASS: Pad 3 on 2x2")
    
    def test_pad_large_padding(self):
        """Test with large padding values"""
        result = pad_binary_image(self.single_pixel, 5, 5)
        
        self.assertEqual(result.shape, (11, 11))
        self.assertEqual(result[5, 5], 255)
        
        # Count non-zero pixels (should be just 1)
        self.assertEqual(np.sum(result == 255), 1)
        
        print("✓ PASS: Large padding")
    
    def test_pad_dtype(self):
        """Padded image should be uint8"""
        result = pad_binary_image(self.binary_3x3, 1, 1)
        self.assertEqual(result.dtype, np.uint8)
        print("✓ PASS: Output dtype is uint8")
    
    def test_pad_preserves_values(self):
        """Padding should not change original pixel values"""
        result = pad_binary_image(self.binary_3x3, 2, 2)
        
        # Check original values are preserved
        self.assertEqual(result[2, 2], 255)
        self.assertEqual(result[2, 3], 0)
        self.assertEqual(result[2, 4], 255)
        self.assertEqual(result[3, 2], 0)
        self.assertEqual(result[3, 3], 255)
        self.assertEqual(result[3, 4], 0)
        self.assertEqual(result[4, 2], 255)
        self.assertEqual(result[4, 3], 0)
        self.assertEqual(result[4, 4], 255)
        
        print("✓ PASS: Original values preserved")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: pad_binary_image")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPadBinaryImage)
    unittest.TextTestRunner(verbosity=2).run(suite)