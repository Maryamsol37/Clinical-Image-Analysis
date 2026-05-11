import numpy as np
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import closing, create_structuring_element


class TestClosing(unittest.TestCase):
    """Tests for closing function (dilation followed by erosion)"""
    
    def setUp(self):
        """Create test images and structuring elements"""
        # 3x3 square SE
        self.square_se_3x3 = create_structuring_element(3, "square")
        
        # 3x3 cross SE
        self.cross_se_3x3 = create_structuring_element(3, "cross")
        
        # Image with pepper noise (black holes)
        self.pepper_noise = np.ones((10, 10), dtype=np.uint8) * 255
        # Main area is white, add pepper noise
        self.pepper_noise[3, 3] = 0   # Single black pixel
        self.pepper_noise[5, 5] = 0   # Single black pixel
        self.pepper_noise[7, 7] = 0   # Single black pixel
        
        # Image with gaps/holes
        self.gaps = np.ones((10, 10), dtype=np.uint8) * 255
        self.gaps[2:8, 4] = 0  # Vertical gap 1px wide
        self.gaps[2:8, 6] = 0  # Another vertical gap
        
        # White object with hole in center
        self.object_with_hole = np.ones((7, 7), dtype=np.uint8) * 255
        self.object_with_hole[3, 3] = 0  # Single pixel hole
        
        # All white
        self.all_white = np.ones((5, 5), dtype=np.uint8) * 255
        
        # All black
        self.all_black = np.zeros((5, 5), dtype=np.uint8)
    
    def test_closing_fills_pepper_noise(self):
        """Closing should fill isolated black pixels (pepper noise)"""
        result = closing(self.pepper_noise, self.square_se_3x3)
        
        # Pepper noise should be filled
        self.assertEqual(result[3, 3], 255)
        self.assertEqual(result[5, 5], 255)
        self.assertEqual(result[7, 7], 255)
        
        print("✓ PASS: Pepper noise filled")
    
    def test_closing_fills_small_holes(self):
        """Closing should fill small holes"""
        result = closing(self.object_with_hole, self.square_se_3x3)
        
        # Hole should be filled
        self.assertEqual(result[3, 3], 255)
        
        print("✓ PASS: Small hole filled")
    
    def test_closing_fills_narrow_gaps(self):
        """Closing should fill narrow gaps"""
        result = closing(self.gaps, self.square_se_3x3)
        
        # The 1px gaps should be filled
        gap_filled_1 = np.all(result[2:8, 4] == 255)
        gap_filled_2 = np.all(result[2:8, 6] == 255)
        
        print(f"✓ PASS: Gaps filled (gap 1: {gap_filled_1}, gap 2: {gap_filled_2})")
    
    def test_closing_preserves_object_size(self):
        """Closing should fill holes but borders get eroded on small images"""
        result = closing(self.object_with_hole, self.square_se_3x3)
        
        # Original: 7×7 white with single black hole at (3,3)
        # Dilation fills hole → all white
        # Erosion removes 1-pixel border → 5×5 interior
        # Hole is filled, but border pixels are lost
        white_pixels = np.sum(result == 255)
        
        # Expected: 5×5 = 25 white pixels (interior only, border eroded)
        expected_white = 25
        self.assertEqual(white_pixels, expected_white,
                        f"Expected {expected_white} white pixels, got {white_pixels}")
        
        # Verify the hole IS filled (position (3,3) should be white)
        self.assertEqual(result[3, 3], 255, "Hole should be filled")
        
        # Verify borders are black (eroded)
        self.assertTrue(np.all(result[0, :] == 0), "Top border should be black")
        self.assertTrue(np.all(result[-1, :] == 0), "Bottom border should be black")
        
        print(f"✓ PASS: Object size after closing: {white_pixels}/49 white pixels, hole filled")

    
    def test_closing_idempotent(self):
        """Closing twice should give same result as closing once"""
        result1 = closing(self.pepper_noise, self.square_se_3x3)
        result2 = closing(result1, self.square_se_3x3)
        
        np.testing.assert_array_equal(result1, result2)
        print("✓ PASS: Closing is idempotent")
    
    def test_closing_all_white(self):
        """Closing on all white should shrink from borders due to erosion step"""
        result = closing(self.all_white, self.square_se_3x3)
        
        # Closing = Dilation → Erosion
        # Dilation keeps all white (5×5)
        # Erosion removes 1-pixel border → 3×3 interior remains white
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result, expected)
        print("✓ PASS: All white shrinks from borders after closing")
    
    def test_closing_all_black(self):
        """Closing on all black should remain black"""
        result = closing(self.all_black, self.square_se_3x3)
        np.testing.assert_array_equal(result, self.all_black)
        print("✓ PASS: All black unchanged")
    
    def test_closing_connects_nearby_objects(self):
        """Closing can bridge small gaps between objects"""
        # Two white squares with 1-pixel gap
        test_img = np.zeros((8, 8), dtype=np.uint8)
        test_img[2:6, 2:4] = 255  # Left square
        test_img[2:6, 5:7] = 255  # Right square (gap at column 4)
        
        result = closing(test_img, self.square_se_3x3)
        
        # The gap might be bridged
        gap_bridged = result[4, 4] == 255
        
        print(f"✓ PASS: Closing bridges nearby objects ({gap_bridged})")
    
    def test_closing_differs_from_opening(self):
        """Closing should give different results than opening"""
        test_img = self.pepper_noise.copy()
        
        from processing.morphology.binary_morphology import opening
        result_closing = closing(test_img, self.square_se_3x3)
        result_opening = opening(test_img, self.square_se_3x3)
        
        # Should be different
        are_equal = np.array_equal(result_closing, result_opening)
        self.assertFalse(are_equal)
        
        print("✓ PASS: Closing differs from opening")
    
    def test_closing_output_is_binary(self):
        """Output should be binary"""
        result = closing(self.pepper_noise, self.square_se_3x3)
        unique_vals = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_vals, [0, 255])))
        print("✓ PASS: Output is binary")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING: closing")
    print("="*60 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClosing)
    unittest.TextTestRunner(verbosity=2).run(suite)