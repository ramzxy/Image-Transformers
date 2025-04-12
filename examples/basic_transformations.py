#!/usr/bin/env python3
"""
Example demonstrating basic transformations using the ImageTransformer.
"""

import os
import sys
# Add the parent directory to sys.path for direct running of examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_transformer import ImageTransformer

def main():
    # Path to sample image (assumed to be in the examples directory)
    sample_path = os.path.join(os.path.dirname(__file__), "sample_image.jpg")
    
    # Check if sample image exists, otherwise use the one from project root
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "sample_image.jpg")
        # Copy to examples directory if needed
        if os.path.exists(sample_path):
            import shutil
            shutil.copy(sample_path, os.path.join(os.path.dirname(__file__), "sample_image.jpg"))
    
    if not os.path.exists(sample_path):
        print("Sample image not found. Please provide a sample image named 'sample_image.jpg'")
        return
    
    # Initialize transformer
    transformer = ImageTransformer(sample_path)
    
    # Load the image
    image = transformer.load_image()
    if image:
        print(f"Image loaded: {image.format}, size: {image.size}, mode: {image.mode}")
    else:
        print("Failed to load image")
        return
    
    # Directory for output images
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Resize
    resized = ImageTransformer(sample_path)
    resized.load_image()
    resized.resize(200, 200)
    resized.save_image(os.path.join(output_dir, "resized.jpg"))
    print("Saved resized image")
    
    # Example 2: Grayscale and edge detection
    edges = ImageTransformer(sample_path)
    edges.load_image()
    edges.to_grayscale().detect_edges_sobel()
    edges.save_image(os.path.join(output_dir, "edges.jpg"))
    print("Saved edge detection image")
    
    # Example 3: Multiple transformations
    multi = ImageTransformer(sample_path)
    multi.load_image()
    multi.resize(300, 300) \
         .adjust_brightness(1.2) \
         .adjust_contrast(1.5) \
         .blur_box(3)
    multi.save_image(os.path.join(output_dir, "multi_transformed.jpg"))
    print("Saved multi-transformed image")
    
    # Example 4: Sepia tone
    sepia = ImageTransformer(sample_path)
    sepia.load_image()
    sepia.apply_sepia()
    sepia.save_image(os.path.join(output_dir, "sepia.jpg"))
    print("Saved sepia image")
    
    # Example 5: Rotate
    rotated = ImageTransformer(sample_path)
    rotated.load_image()
    rotated.rotate(45)
    rotated.save_image(os.path.join(output_dir, "rotated.jpg"))
    print("Saved rotated image")
    
    print(f"All transformed images saved to {output_dir}")

if __name__ == "__main__":
    main() 