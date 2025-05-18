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
    """
    Convert an image to a grayscale matrix representation.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        list: 2D matrix representing the grayscale image
    """
    with Image.open(image_path) as img:
        # Convert to grayscale
        img = img.convert('L')
        # Convert to matrix using list comprehension
        return [[img.getpixel((j, i)) for j in range(img.width)] for i in range(img.height)]

def save_image_matrix(matrix, output_path):
    """
    Convert a matrix back to an image and save it.
    
    Args:
        matrix (list): 2D matrix representing the image
        output_path (str): Path where the output image will be saved
    """
    height = len(matrix)
    width = len(matrix[0])
    
    # Create new image
    img = Image.new('L', (width, height))
    
    # Set pixel values using list comprehension
    for i in range(height):
        for j in range(width):
            # Ensure value is in valid range (0-255)
            img.putpixel((j, i), min(max(int(matrix[i][j]), 0), 255))
    
    img.save(f"./output/{output_path}")

# Image Transformation Functions
def brighten_image(matrix, amount=50):
    """
    Increase the brightness of an image by adding a constant value.
    
    Args:
        matrix (list): Input image matrix
        amount (int): Brightness increase amount (default: 50)
        
    Returns:
        list: Brightened image matrix
    """
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
    """
    Apply skew transformation to an image.
    
    Args:
        matrix (list): Input image matrix
        skew_factor_x (float): Horizontal skew factor (default: 0.5)
        skew_factor_y (float): Vertical skew factor (default: 0.0)
        
    Returns:
        list: Skewed image matrix
    """
    height = len(matrix)
    width = len(matrix[0])
    
    # Create skew transformation matrix
    skew_matrix = [
        [1, skew_factor_x],
        [skew_factor_y, 1]
    ]
    
    # Calculate dimensions after skewing
    new_width = width + int(height * abs(skew_factor_x))
    new_height = height + int(width * abs(skew_factor_y))
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(new_width)] for _ in range(new_height)]
    
    # Reference point (origin for skew transformation)
    ref_x = 0
    ref_y = 0
    
    # Apply skew transformation to each pixel
    for i in range(height):
        for j in range(width):
            # Create coordinate matrix relative to reference point
            coord_matrix = [
                [j - ref_x],
                [i - ref_y]
            ]
            
            # Apply skew using matrix multiplication
            skewed_coord = matrix_multiplication(skew_matrix, coord_matrix)
            
            # Add reference point offset back
            new_j = int(skewed_coord[0][0]) + ref_x
            new_i = int(skewed_coord[1][0]) + ref_y
            
            # Check if new position is within bounds
            if 0 <= new_i < new_height and 0 <= new_j < new_width:
                result[new_i][new_j] = matrix[i][j]
    
    return result

def scale_image(matrix, scale_x=1.5, scale_y=1.5):
    """Scale image using a scaling matrix transformation."""
    height = len(matrix)
    width = len(matrix[0])
    
    # Create scaling matrix
    scaling_matrix = [
        [scale_x, 0],
        [0, scale_y]
    ]
    
    # Calculate new dimensions
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)
    
    # Create result matrix
    result = [[0 for _ in range(new_width)] for _ in range(new_height)]
    
    # Apply inverse mapping to find source pixels
    for new_i in range(new_height):
        for new_j in range(new_width):
            # Create coordinate matrix for destination
            dest_coord = [
                [new_j],
                [new_i]
            ]
            
            # Inverse scaling matrix (1/scale_x, 1/scale_y)
            inverse_scaling = [
                [1/scale_x, 0],
                [0, 1/scale_y]
            ]
            
            # Apply inverse scaling to find source pixel
            src_coord = matrix_multiplication(inverse_scaling, dest_coord)
            
            # Get original coordinates
            src_j = int(src_coord[0][0])
            src_i = int(src_coord[1][0])
            
            # Check if source pixel is within bounds
            if 0 <= src_i < height and 0 <= src_j < width:
                result[new_i][new_j] = matrix[src_i][src_j]
    
    return result

def main():
    """
    Main function to demonstrate various image transformations.
    Loads an input image and applies different transformations,
    saving the results as separate files.
    """
    # Load image as matrix
    input_matrix = get_image_matrix("input.jpg")
    
    # Apply and save various transformations
    transformations = [
        (brighten_image(input_matrix), "brighter.jpg"),
        (darken_image(input_matrix), "darker.jpg"),
        (flip_image(input_matrix), "flipped.jpg"),
        (negative_image(input_matrix), "negative.jpg"),
        (rotate_image(input_matrix), "rotated90.jpg"),
        (skew_image(input_matrix, 0.5, 0.0), "skewed_horizontal.jpg"),
        (skew_image(input_matrix, 0.0, 0.5), "skewed_vertical.jpg"),
        (scale_image(input_matrix, 1.5, 1.5), "scaled.jpg"),
        (scale_image(input_matrix, 0.5, 0.5), "scaled_down.jpg")
    ]
    
    # Save all transformations
    for transformed_matrix, output_path in transformations:
        save_image_matrix(transformed_matrix, output_path)

if __name__ == "__main__":
    main()

