# Matrix Image Transformations

A simple Python project that demonstrates basic matrix operations on images. The project converts images to matrices, applies matrix transformations, and converts them back to images.

## Matrix Operations

1. **Matrix Addition**
```python
# Example:
(1, 2)   +   (5, 6)   =   (1+5, 2+6)
(3, 4)       (7, 8)       (3+7, 4+8)
# Result: (6, 8, 10, 12)
```

2. **Matrix Subtraction**
```python
A - B = Matrix of (A[i][j] - B[i][j])
```

3. **Matrix Multiplication**
```python
# Multiplies corresponding elements
result[i][j] = sum(matrix1[i][k] * matrix2[k][j])
```

4. **Matrix Transpose**
```python
# Flips matrix over its diagonal
(1, 2)  ->  (1, 3)
(3, 4)      (2, 4)
```

5. **Scalar Multiplication**
```python
# Multiplies each element by a number
matrix * 2 = Matrix of (element * 2)
```

## Image Transformations

The project includes four basic image transformations:
1. **Brighten**: Adds 50 to each pixel value using matrix addition
2. **Darken**: Multiplies each pixel value by 0.7 using scalar multiplication
3. **Flip**: Transposes the image matrix
4. **Negative**: Subtracts pixel values from 255 to invert the image

## Usage

1. Install the required package:
```bash
pip install pillow
```

2. Place your input image as "input.jpg" in the project folder

3. Run the program:
```bash
python main.py
```

4. Check the output files:
- brighter.jpg
- darker.jpg
- flipped.jpg
- negative.jpg

## Note
This project works with grayscale images only. Color images will be automatically converted to grayscale during processing. 