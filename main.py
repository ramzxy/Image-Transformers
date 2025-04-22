from PIL import Image
import math

def matrix_addition(matrix1, matrix2):
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[i])):
            row.append(matrix1[i][j] + matrix2[i][j])       
        result.append(row)                                         
    return result

def matrix_subtraction(matrix1, matrix2):
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[i])):
            row.append(matrix1[i][j] - matrix2[i][j])
        result.append(row)
    return result

def matrix_multiplication(matrix1, matrix2):
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
    result = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        result.append(row)
    return result

def matrix_scalar_multiplication(matrix, scalar):
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            row.append(matrix[i][j] * scalar)
        result.append(row)
    return result

def get_image_matrix(image_path):
    with Image.open(image_path) as img:
        # Convert to grayscale
        img = img.convert('L')
        # Convert to matrix using list comprehension
        return [[img.getpixel((j, i)) for j in range(img.width)] for i in range(img.height)]

def save_image_matrix(matrix, output_path):
    height = len(matrix)
    width = len(matrix[0])
    
    # Create new image
    img = Image.new('L', (width, height))
    
    # Set pixel values using list comprehension
    for i in range(height):
        for j in range(width):
            # Ensure value is in valid range (0-255)
            img.putpixel((j, i), min(max(int(matrix[i][j]), 0), 255))
    
    img.save(output_path)

def brighten_image(matrix, amount=50):
    height = len(matrix)
    width = len(matrix[0])
    brightness_matrix = [[amount for _ in range(width)] for _ in range(height)]
    return matrix_addition(matrix, brightness_matrix)

def darken_image(matrix, factor=0.7):
    """Make image darker by multiplying with a factor less than 1."""
    return matrix_scalar_multiplication(matrix, factor)

def flip_image(matrix):
    """Flip image by transposing the matrix."""
    return matrix_transpose(matrix)

def negative_image(matrix):
    """Create negative by subtracting from 255."""
    height = len(matrix)
    width = len(matrix[0])
    max_matrix = [[255 for _ in range(width)] for _ in range(height)]
    return matrix_subtraction(max_matrix, matrix)

def rotate_image(matrix):
    """Rotate image by 90 degrees clockwise using matrix transpose."""
    # First transpose the matrix
    transposed = matrix_transpose(matrix)
    # Then reverse each row for 90-degree clockwise rotation
    return [row[::-1] for row in transposed]

def skew_image(matrix, skew_factor_x=0.5, skew_factor_y=0.0):
    height = len(matrix)
    width = len(matrix[0])
    
    # Create skew matrix
    skew_matrix = [
        [1, skew_factor_x],
        [skew_factor_y, 1]
    ]
    
    # Calculate new dimensions
    new_width = width + int(height * abs(skew_factor_x))
    new_height = height + int(width * abs(skew_factor_y))
    
    # Create result matrix
    result = [[0 for _ in range(new_width)] for _ in range(new_height)]
    
    # Reference point (origin for skew transformation)
    ref_x = 0
    ref_y = 0
    
    # Apply skew transformation using matrix multiplication
    for i in range(height):
        for j in range(width):
            # Create coordinate matrix relative to reference point
            coord_matrix = [
                [j - ref_x],
                [i - ref_y]
            ]
            
            # Apply skew using matrix_multiplication
            skewed_coord = matrix_multiplication(skew_matrix, coord_matrix)
            
            # Add reference point offset back
            new_j = int(skewed_coord[0][0]) + ref_x
            new_i = int(skewed_coord[1][0]) + ref_y
            
            # Check if new position is within bounds
            if 0 <= new_i < new_height and 0 <= new_j < new_width:
                result[new_i][new_j] = matrix[i][j]
    
    return result

def main():
    # Load image as matrix
    input_matrix = get_image_matrix("input.jpg")
    
    # Apply various transformations
    save_image_matrix(brighten_image(input_matrix), "brighter.jpg")
    save_image_matrix(darken_image(input_matrix), "darker.jpg")
    save_image_matrix(flip_image(input_matrix), "flipped.jpg")
    save_image_matrix(negative_image(input_matrix), "negative.jpg")
    save_image_matrix(rotate_image(input_matrix), "rotated90.jpg")
    save_image_matrix(skew_image(input_matrix, 0.5, 0.0), "skewed_horizontal.jpg")
    save_image_matrix(skew_image(input_matrix, 0.0, 0.5), "skewed_vertical.jpg")

if __name__ == "__main__":
    main()

