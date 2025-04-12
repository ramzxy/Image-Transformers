import os
import sys
import unittest
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_transformer import ImageTransformer

class TestImageTransformer(unittest.TestCase):
    def setUp(self):
        # Create a small test image
        self.test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        img = Image.new('RGB', (100, 100), color=(73, 109, 137))
        img.save(self.test_image_path)
        
        self.transformer = ImageTransformer(self.test_image_path)
        
    def tearDown(self):
        # Clean up test image
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_load_image(self):
        """Test image loading"""
        image = self.transformer.load_image()
        self.assertIsNotNone(image)
        self.assertEqual(image.size, (100, 100))
    
    def test_to_matrix(self):
        """Test conversion to matrix"""
        matrix = self.transformer.to_matrix()
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape, (100, 100, 3))
        # Check the color values
        self.assertEqual(matrix[0, 0, 0], 73)  # R
        self.assertEqual(matrix[0, 0, 1], 109)  # G
        self.assertEqual(matrix[0, 0, 2], 137)  # B
    
    def test_resize(self):
        """Test resize operation"""
        self.transformer.resize(50, 50)
        self.assertEqual(self.transformer.matrix.shape, (50, 50, 3))
    
    def test_to_grayscale(self):
        """Test grayscale conversion"""
        self.transformer.to_matrix()  # Ensure matrix is loaded
        self.transformer.to_grayscale()
        # Should now be a 2D array
        self.assertEqual(len(self.transformer.matrix.shape), 2)
        self.assertEqual(self.transformer.matrix.shape, (100, 100))
    
    def test_method_chaining(self):
        """Test method chaining"""
        result = self.transformer.resize(50, 50).to_grayscale()
        # Should be the same object
        self.assertEqual(id(result), id(self.transformer))
        # Shape should be updated
        self.assertEqual(self.transformer.matrix.shape, (50, 50))

if __name__ == '__main__':
    unittest.main() 