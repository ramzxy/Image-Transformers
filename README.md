# Image Transformer

A Python library for image transformation using matrix operations and linear algebra. This project implements various image transformations from scratch using NumPy, without relying on prebuilt image processing functions.

## Features

- **Basic Matrix Operations**: Converts images to matrices and applies transformations directly to the pixel values
- **No Dependencies on Image Processing Libraries**: All transformations are implemented using matrix operations
- **Supported Transformations**:
  - Resize (nearest neighbor interpolation)
  - Rotate (using rotation matrices)
  - Flip (horizontal and vertical)
  - Grayscale conversion (weighted RGB method)
  - Brightness/contrast adjustment
  - Blur (box blur using manual convolution)
  - Edge detection (Sobel operator)
  - Thresholding
  - Crop
  - Sepia tone effect

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-transformer.git
cd image-transformer

# Install dependencies
pip install numpy pillow
```

## Usage

```python
from Transformer import ImageTransformer

# Initialize with an image path
transformer = ImageTransformer("path_to_your_image.jpg")

# Load the image
transformer.load_image()

# Apply transformations (methods can be chained)
transformer.resize(400, 300) \
          .to_grayscale() \
          .adjust_contrast(1.5) \
          .blur_box(3)

# Save the result
transformer.save_image("transformed_image.jpg")
```

## Example Transformations

### Resize
```python
# Resize to 50% of original dimensions
width, height = image.size
transformer.resize(width // 2, height // 2)
```

### Rotate
```python
# Rotate image by 45 degrees
transformer.rotate(45)
```

### Grayscale and Edge Detection
```python
# Convert to grayscale and find edges
transformer.to_grayscale().detect_edges_sobel()
```

## How It Works

Unlike most image processing libraries, this project implements transformations using fundamental matrix operations:

1. Images are loaded and converted to NumPy matrices
2. Transformations are applied as matrix operations
3. The resulting matrix is converted back to an image

For example, the sepia filter applies a 3×3 color transformation matrix:

```python
sepia_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131]
])
```

## Requirements

- Python 3.6+
- NumPy
- Pillow (PIL)

## Educational Purpose

This project is designed to demonstrate how image processing algorithms work at a fundamental level using matrix operations. It's ideal for:

- Understanding image processing fundamentals
- Learning matrix transformations
- Studying digital image representation

## License

MIT 