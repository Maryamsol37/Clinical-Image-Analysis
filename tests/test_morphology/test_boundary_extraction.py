import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import boundary_extraction, create_structuring_element


class TestBoundaryExtraction(unittest.TestCase):
    """Tests for boundary_extraction function"""
    
    def setUp(self):
        """Create test images and structuring elements"""
        # 3x3 square SE
        self.square_se_3x3 = create_structuring_element(3, "square")
        
        # 3x3 cross SE
        self.cross_se_3x3 = create_structuring_element(3, "cross")
        
        # White square on black background (5x5 white in 7x7)
        self.white_square = np.zeros((7, 7), dtype=np.uint8)
        self.white_square[2:5, 2:5] = 255
        
        # White rectangle
        self.white_rectangle = np.zeros((7, 9), dtype=np.uint8)
        self.white_rectangle[1:6, 2:7] = 255
        
        # All white
        self.all_white = np.ones((5, 5), dtype=np.uint8) * 255
        
        # All black
        self.all_black = np.zeros((5, 5), dtype=np.uint8)
        
        # Complex shape (cross)
        self.cross_shape = np.zeros((7, 7), dtype=np.uint8)
        self.cross_shape[3, :] = 255  # Horizontal line
        self.cross_shape[:, 3] = 255  # Vertical line
    
    def test_boundary_of_square(self):
        """Boundary of white square should be outer edge"""
        result = boundary_extraction(self.white_square, self.square_se_3x3)
        
        # Expected: outer border of the 3x3 square
        expected = np.zeros((7, 7), dtype=np.uint8)
        expected[2:5, 2:5] = 255  # All of square
        expected[3, 3] = 0        # Inner pixel removed
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: Square boundary correct")
    
    def test_boundary_of_rectangle(self):
        """Boundary of rectangle should be its outer perimeter"""
        result = boundary_extraction(self.white_rectangle, self.square_se_3x3)
        
        # Check that boundary exists (non-zero pixels)
        boundary_pixels = np.sum(result == 255)
        self.assertGreater(boundary_pixels, 0)
        
        # Check that interior has zeros
        # The interior (3:4, 3:6) should be 0 after boundary extraction
        interior_region = result[3:4, 3:6]
        self.assertTrue(np.all(interior_region == 0))
        
        print(f"✓ PASS: Rectangle boundary has {boundary_pixels} pixels")
    
    def test_boundary_one_pixel_thick(self):
        """Boundary should be approximately one pixel thick"""
        result = boundary_extraction(self.white_square, self.square_se_3x3)
        
        # Check that center of original square is 0
        self.assertEqual(result[3, 3], 0)
        
        # But boundary pixels exist
        self.assertEqual(result[2, 2], 255)  # Corner
        self.assertEqual(result[2, 3], 255)  # Edge
        self.assertEqual(result[2, 4], 255)  # Corner
        
        print("✓ PASS: Boundary is one pixel thick")
    
    def test_boundary_all_white(self):
        """Boundary of all white image should be outer border only"""
        result = boundary_extraction(self.all_white, self.square_se_3x3)
        
        expected = np.ones((5, 5), dtype=np.uint8) * 255
        expected[1:4, 1:4] = 0  # Interior becomes black
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: All white boundary is outer border")
    
    def test_boundary_all_black(self):
        """Boundary of all black should be all black"""
        result = boundary_extraction(self.all_black, self.square_se_3x3)
        np.testing.assert_array_equal(result, self.all_black)
        print("✓ PASS: All black boundary is all black")
    
    def test_boundary_cross_se(self):
        """Boundary extraction with cross SE"""
        result = boundary_extraction(self.white_square, self.cross_se_3x3)
        
        # Should still extract boundary, shape might differ
        boundary_pixels = np.sum(result == 255)
        self.assertGreater(boundary_pixels, 0)
        
        print(f"✓ PASS: Cross SE boundary has {boundary_pixels} pixels")
    
    def test_boundary_complex_shape(self):
        """Boundary of cross shape with square SE"""
        result = boundary_extraction(self.cross_shape, self.square_se_3x3)
        
        # The cross shape has arms that are only 1 pixel thick
        # With a 3x3 square SE, NO pixel survives erosion
        # Because every white pixel has at least one diagonal black neighbor
        # 
        # Boundary = Original - Eroded = Original - 0 = Original
        # The entire cross shape becomes the boundary
        expected = self.cross_shape.copy()
        np.testing.assert_array_equal(result, expected)
        
        # Center intersection SHOULD be 255 (it's part of the boundary now)
        self.assertEqual(result[3, 3], 255, "Center is boundary because it gets eroded")
        
        boundary_pixels = np.sum(result == 255)
        print(f"✓ PASS: Complex shape boundary extracted ({boundary_pixels} pixels)")
        
    def test_boundary_output_is_binary(self):
        """Output should be binary"""
        result = boundary_extraction(self.white_square, self.square_se_3x3)
        unique_vals = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_vals, [0, 255])))
        print("✓ PASS: Output is binary")
    
    def test_boundary_single_pixel(self):
        """Boundary of single pixel should be that pixel"""
        single = np.zeros((5, 5), dtype=np.uint8)
        single[2, 2] = 255
        
        result = boundary_extraction(single, self.square_se_3x3)
        
        # Single pixel gets eroded away, so boundary = original - 0 = original
        self.assertEqual(result[2, 2], 255)
        
        print("✓ PASS: Single pixel boundary is itself")
    
    def test_boundary_subtraction_logic(self):
        """Verify boundary = original - eroded"""
        from processing.morphology.binary_morphology import erode
        
        original = self.white_square.copy()
        eroded = erode(original, self.square_se_3x3)
        
        # Manual boundary calculation
        manual_boundary = np.zeros_like(original)
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                val = int(original[i, j]) - int(eroded[i, j])
                if val < 0:
                    val = 0
                manual_boundary[i, j] = val
        
        result = boundary_extraction(original, self.square_se_3x3)
        np.testing.assert_array_equal(result, manual_boundary)
        print("✓ PASS: Boundary = original - eroded")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: boundary_extraction")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBoundaryExtraction)
    unittest.TextTestRunner(verbosity=2).run(suite)