#!/usr/bin/env python3
"""
Main entry point for the image_transformer package.
Allows running the package as a module with: python -m image_transformer
"""

import os
import sys
import argparse
from image_transformer import ImageTransformer

def main():
    parser = argparse.ArgumentParser(description='Image Transformer - Apply matrix transformations to images')
    
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('-o', '--output', help='Output image file path', default='transformed_output.jpg')
    
    # Add transformation arguments
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), 
                        help='Resize image to specified width and height')
    parser.add_argument('--rotate', type=float, help='Rotate image by specified angle in degrees')
    parser.add_argument('--grayscale', action='store_true', help='Convert to grayscale')
    parser.add_argument('--brightness', type=float, help='Adjust brightness (1.0 is unchanged)')
    parser.add_argument('--contrast', type=float, help='Adjust contrast (1.0 is unchanged)')
    parser.add_argument('--blur', type=int, help='Apply box blur with specified kernel size')
    parser.add_argument('--edges', action='store_true', help='Apply edge detection')
    parser.add_argument('--threshold', type=int, help='Apply threshold with specified value (0-255)')
    parser.add_argument('--sepia', action='store_true', help='Apply sepia tone')
    parser.add_argument('--flip-h', action='store_true', help='Flip horizontally')
    parser.add_argument('--flip-v', action='store_true', help='Flip vertically')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
        
    # Initialize transformer
    transformer = ImageTransformer(args.input)
    
    # Load the image
    image = transformer.load_image()
    if not image:
        print("Error: Failed to load image")
        sys.exit(1)
    
    print(f"Image loaded: {image.format}, size: {image.size}, mode: {image.mode}")
    
    # Apply transformations in a sensible order
    transformations_applied = 0
    
    if args.resize:
        width, height = args.resize
        transformer.resize(width, height)
        print(f"Resized to {width}x{height}")
        transformations_applied += 1
    
    if args.rotate:
        transformer.rotate(args.rotate)
        print(f"Rotated by {args.rotate} degrees")
        transformations_applied += 1
    
    if args.flip_h:
        transformer.flip_horizontal()
        print("Flipped horizontally")
        transformations_applied += 1
    
    if args.flip_v:
        transformer.flip_vertical()
        print("Flipped vertically")
        transformations_applied += 1
    
    if args.grayscale:
        transformer.to_grayscale()
        print("Converted to grayscale")
        transformations_applied += 1
    
    if args.brightness:
        transformer.adjust_brightness(args.brightness)
        print(f"Adjusted brightness by factor of {args.brightness}")
        transformations_applied += 1
    
    if args.contrast:
        transformer.adjust_contrast(args.contrast)
        print(f"Adjusted contrast by factor of {args.contrast}")
        transformations_applied += 1
    
    if args.blur:
        transformer.blur_box(args.blur)
        print(f"Applied blur with kernel size {args.blur}")
        transformations_applied += 1
    
    if args.edges:
        transformer.detect_edges_sobel()
        print("Applied edge detection")
        transformations_applied += 1
    
    if args.threshold:
        transformer.threshold(args.threshold)
        print(f"Applied threshold with value {args.threshold}")
        transformations_applied += 1
    
    if args.sepia:
        transformer.apply_sepia()
        print("Applied sepia tone")
        transformations_applied += 1
    
    # Save the result
    if transformations_applied > 0:
        if transformer.save_image(args.output):
            print(f"Transformed image saved to {args.output}")
        else:
            print(f"Error: Failed to save image to {args.output}")
            sys.exit(1)
    else:
        print("No transformations specified. Use --help to see available options.")

if __name__ == "__main__":
    main() 