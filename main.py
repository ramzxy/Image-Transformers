from PIL import Image

def matrix_addition(matrix1, matrix2):
    """Add two matrices."""
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[i])):
            row.append(matrix1[i][j] + matrix2[i][j]) #      (1, 2) (3, 4)    (5, 6) (7, 8)       (1+5, 2+6) (3+7, 4+8) 
        result.append(row)                                          # (6, 8, 10, 12)
    return result

def matrix_subtraction(matrix1, matrix2):
    """Subtract two matrices."""
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[i])):
            row.append(matrix1[i][j] - matrix2[i][j])
        result.append(row)
    return result

def matrix_multiplication(matrix1, matrix2):
    """Multiply two matrices."""
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix2[0])):
            sum = 0
            for k in range(len(matrix2)):
                sum += matrix1[i][k] * matrix2[k][j]
            row.append(sum)
        result.append(row)
    return result

def matrix_transpose(matrix):
    """Transpose a matrix."""
    result = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        result.append(row)
    return result

def matrix_scalar_multiplication(matrix, scalar):
    """Multiply a matrix by a scalar."""
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            row.append(matrix[i][j] * scalar)
        result.append(row)
    return result

def get_image_matrix(image_path):
    """Convert image to grayscale matrix."""
    with Image.open(image_path) as img:
        # Convert to grayscale
        img = img.convert('L')
        # Convert to matrix
        matrix = []
        for i in range(img.height):
            row = []
            for j in range(img.width):
                row.append(img.getpixel((j, i)))
            matrix.append(row)
        return matrix

def save_image_matrix(matrix, output_path):
    """Save matrix as grayscale image."""
    height = len(matrix)
    width = len(matrix[0])
    
    # Create new image
    img = Image.new('L', (width, height))
    
    # Set pixel values
    for i in range(height):
        for j in range(width):
            # Ensure value is in valid range (0-255)
            pixel_value = min(max(int(matrix[i][j]), 0), 255)
            img.putpixel((j, i), pixel_value)
    
    img.save(output_path)

def main():
    # Load image as matrix
    input_matrix = get_image_matrix("input.jpg")
    
    # 1. Make image brighter (add 50 to all pixels)
    bright_matrix = matrix_addition(input_matrix, 
                                  [[50 for _ in range(len(input_matrix[0]))] 
                                   for _ in range(len(input_matrix))])
    save_image_matrix(bright_matrix, "brighter.jpg")
    
    # 2. Make image darker (multiply by 0.7)
    dark_matrix = matrix_scalar_multiplication(input_matrix, 0.7)
    save_image_matrix(dark_matrix, "darker.jpg")
    
    # 3. Flip image (transpose)
    flip_matrix = matrix_transpose(input_matrix)
    save_image_matrix(flip_matrix, "flipped.jpg")
    
    # 4. Create negative (subtract from 255)
    negative_matrix = matrix_subtraction([[255 for _ in range(len(input_matrix[0]))] 
                                        for _ in range(len(input_matrix))],
                                       input_matrix)
    save_image_matrix(negative_matrix, "negative.jpg")

if __name__ == "__main__":
    main()

