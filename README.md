# Matrix Image Transformer

A Python program that applies various image transformations using matrix operations. All transformations are implemented using custom matrix operations (no NumPy) to demonstrate linear algebra concepts.

## Features

- **Brightness Adjustment**: Add or subtract intensity from all pixels
- **Flip**: Transpose the image matrix
- **Negative**: Invert pixel values (255 - pixel_value)
- **Rotate**: 90-degree rotations using matrix transpose and row reversal
- **Skew**: Apply horizontal and vertical skew transformations
- **Scale**: Resize images using scaling matrices

## Requirements

- Python 3.10+ (uses match-case statements)
- Pillow (PIL) for image I/O

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create an `output` directory (if it doesn't exist):
```bash
mkdir output
```

## Usage

1. Place your input image in the project directory (or specify the full path)
2. Run the program:
```bash
python main.py
```

3. Follow the prompts:
   - Enter the image path (or press Enter for 'input.jpg')
   - Choose a transformation (1-6)
   - Enter required parameters for the chosen transformation

## Transformations

### 1. Brightness Adjustment
- **Range**: -255 to 255
- **Effect**: Positive values brighten, negative values darken

### 2. Flip
- **Effect**: Transposes the image matrix (swaps rows and columns)

### 3. Negative
- **Effect**: Creates a negative image by subtracting each pixel from 255

### 4. Rotate
- **Input**: Angle in multiples of 90 degrees
- **Effect**: Rotates the image clockwise

### 5. Skew
- **Horizontal Skew**: -2 to 2 (affects x-coordinates)
- **Vertical Skew**: -2 to 2 (affects y-coordinates)
- **Effect**: Applies shear transformation

### 6. Scale
- **Horizontal Scale**: 0.1 to 5 (width scaling factor)
- **Vertical Scale**: 0.1 to 5 (height scaling factor)
- **Effect**: Resizes the image

## Output

Transformed images are saved in the `output/` directory with descriptive filenames indicating the transformation and parameters used.

## Mathematical Foundation

All transformations use fundamental matrix operations:
- Matrix addition and subtraction
- Matrix multiplication
- Matrix transpose
- Coordinate transformations using 2D transformation matrices

The image is represented as a 2D matrix where each element represents a grayscale pixel value (0-255).

## Example

```bash
python main.py
# Enter input image path: sample.jpg
# Choose transformation: 5 (Skew)
# Enter horizontal skew: 0.5
# Enter vertical skew: 0.0
# Output: output/skewed_0.5_0.0.jpg
```
