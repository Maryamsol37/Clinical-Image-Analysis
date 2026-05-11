import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import dilate, create_structuring_element


class TestDilation(unittest.TestCase):
    """Tests for dilate function"""
    
    def setUp(self):
        """Create test images and structuring elements"""
        # 3x3 square SE
        self.square_se_3x3 = create_structuring_element(3, "square")
        
        # 3x3 cross SE
        self.cross_se_3x3 = create_structuring_element(3, "cross")
        
        # 5x5 square SE
        self.square_se_5x5 = create_structuring_element(5, "square")
        
        # Single white pixel in 5x5 image
        self.single_pixel = np.zeros((5, 5), dtype=np.uint8)
        self.single_pixel[2, 2] = 255
        
        # Single pixel in 7x7 (for larger SE tests)
        self.single_pixel_7x7 = np.zeros((7, 7), dtype=np.uint8)
        self.single_pixel_7x7[3, 3] = 255
        
        # All black
        self.all_black = np.zeros((5, 5), dtype=np.uint8)
        
        # All white
        self.all_white = np.ones((5, 5), dtype=np.uint8) * 255
        
        # Two separate white pixels close together
        self.two_pixels = np.zeros((7, 7), dtype=np.uint8)
        self.two_pixels[2, 3] = 255
        self.two_pixels[4, 3] = 255
    
    def test_dilation_single_pixel_square_se(self):
        """Single pixel dilated by 3x3 square SE becomes 3x3 square"""
        result = dilate(self.single_pixel, self.square_se_3x3)
        
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1:4, 1:4] = 255
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Single pixel → 3x3 square")
    
    def test_dilation_single_pixel_cross_se(self):
        """Single pixel dilated by 3x3 cross SE becomes cross"""
        result = dilate(self.single_pixel, self.cross_se_3x3)
        
        # Single pixel at (2,2) in 5x5 image dilates to a cross
        # that extends 1 pixel in each direction from center
        # The cross CANNOT reach the image borders (rows 0,4 and cols 0,4)
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 255, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 255, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Single pixel → cross shape")
        
    def test_dilation_single_pixel_5x5_se(self):
        """Single pixel dilated by 5x5 SE becomes 5x5 square"""
        result = dilate(self.single_pixel_7x7, self.square_se_5x5)
        
        expected = np.zeros((7, 7), dtype=np.uint8)
        expected[1:6, 1:6] = 255
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Single pixel → 5x5 square")
    
    def test_dilation_all_black_stays_black(self):
        """All black image should stay black"""
        result = dilate(self.all_black, self.square_se_3x3)
        np.testing.assert_array_equal(result, self.all_black)
        print("✓ PASS: All black stays black")
    
    def test_dilation_all_white_stays_white(self):
        """All white image should stay white"""
        result = dilate(self.all_white, self.square_se_3x3)
        np.testing.assert_array_equal(result, self.all_white)
        print("✓ PASS: All white stays white")
    
    def test_dilation_expands_boundary(self):
        """Dilation should expand white regions"""
        # 3x3 white square in center of 7x7
        test_img = np.zeros((7, 7), dtype=np.uint8)
        test_img[2:5, 2:5] = 255
        
        result = dilate(test_img, self.square_se_3x3)
        
        # Should expand by 1 pixel on all sides
        expected = np.zeros((7, 7), dtype=np.uint8)
        expected[1:6, 1:6] = 255
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Boundary expanded by 1 pixel")
    
    def test_dilation_connects_objects(self):
        """Dilation should connect nearby objects"""
        result = dilate(self.two_pixels, self.square_se_3x3)
        
        # The two dilated regions should overlap (connect)
        middle_region = result[3, 2:5]  # Between the two original pixels
        self.assertTrue(np.all(middle_region == 255))
        
        print("✓ PASS: Dilation connects nearby objects")
    
    def test_dilation_fills_gaps(self):
        """Dilation should fill small gaps"""
        # Two white regions with 1-pixel gap
        test_img = np.zeros((5, 5), dtype=np.uint8)
        test_img[:, :2] = 255  # Left 2 columns white
        test_img[:, 3:] = 255  # Right 2 columns white
        # Gap at column 2
        
        result = dilate(test_img, self.square_se_3x3)
        
        # The gap should be filled
        self.assertTrue(np.all(result[:, 2] == 255))
        
        print("✓ PASS: Small gap filled")
    
    def test_dilation_corner_pixel(self):
        """Dilation of corner pixel"""
        corner_pixel = np.zeros((5, 5), dtype=np.uint8)
        corner_pixel[0, 0] = 255
        
        result = dilate(corner_pixel, self.square_se_3x3)
        
        # Should create 2x2 in corner (limited by image boundaries)
        self.assertEqual(result[0, 0], 255)
        self.assertEqual(result[0, 1], 255)
        self.assertEqual(result[1, 0], 255)
        self.assertEqual(result[1, 1], 255)
        
        print("✓ PASS: Corner pixel dilation correct")
    
    def test_dilation_output_is_binary(self):
        """Output should only contain 0 and 255"""
        result = dilate(self.single_pixel, self.square_se_3x3)
        unique_vals = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_vals, [0, 255])))
        print("✓ PASS: Output is binary")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: dilate")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDilation)
    unittest.TextTestRunner(verbosity=2).run(suite)