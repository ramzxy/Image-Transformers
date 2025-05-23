from PIL import Image

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
    
    img = Image.new('L', (width, height))
    
    # Set pixel values using list comprehension
    for i in range(height):
        for j in range(width):
            # Ensure value is in valid range (0-255)
            img.putpixel((j, i), min(max(int(matrix[i][j]), 0), 255))
    
    img.save(f"./output/{output_path}")

# Image Transformation Functions
def brightness_image(matrix, intensity):
    height = len(matrix)
    width = len(matrix[0])
    brightness_matrix = [[intensity for _ in range(width)] for _ in range(height)]
    return matrix_addition(matrix, brightness_matrix)

def flip_image(matrix):
    return matrix_transpose(matrix)

def negative_image(matrix):
    height = len(matrix)
    width = len(matrix[0])
    max_matrix = [[255 for _ in range(width)] for _ in range(height)]
    return matrix_subtraction(max_matrix, matrix)

def rotate_image(matrix):
    # First transpose the matrix
    transposed = matrix_transpose(matrix)
    # Then reverse each row for 90-degree clockwise rotation
    return [row[::-1] for row in transposed]

def skew_image(matrix, skew_factor_x=0.5, skew_factor_y=0.0):
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
    
    
    # Apply skew transformation to each pixel
    for i in range(height):
        for j in range(width):
            # Create coordinate matrix relative to reference point
            coord_matrix = [
                [j],
                [i]
            ]
            
            # Apply skew using matrix multiplication
            skewed_coord = matrix_multiplication(skew_matrix, coord_matrix)
            
            # Add reference point offset back
            new_j = int(skewed_coord[0][0])
            new_i = int(skewed_coord[1][0])
            
            # Check if new position is within bounds
            if 0 <= new_i < new_height and 0 <= new_j < new_width:
                result[new_i][new_j] = matrix[i][j]
    
    return result

def scale_image(matrix, scale_x=1.5, scale_y=1.5):
    height = len(matrix)
    width = len(matrix[0])
    
    
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

def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")

def get_rotation_input():
    while True:
        angle = get_float_input("Enter rotation angle (must be a multiple of 90 degrees): ")
        if angle % 90 == 0:
            return int(angle // 90)
        print("Angle must be a multiple of 90 degrees")

def main():
    # Load image as matrix
    input_path = input("Enter input image path (or press Enter for 'input.jpg'): ").strip()
    if not input_path:
        input_path = "input.jpg"
    
    input_matrix = get_image_matrix(input_path)
    
    print("\nAvailable transformations:")
    print("1. Brightness adjustment")
    print("2. Flip")
    print("3. Negative")
    print("4. Rotate")
    print("5. Skew")
    print("6. Scale")
    
    choice = input("\nEnter transformation number (1-6): ").strip()
    
    transformed_matrix = None
    output_name = None
    
    match choice:
        case "1":
            intensity = get_float_input("Enter brightness intensity (-255 to 255): ", -255, 255)
            transformed_matrix = brightness_image(input_matrix, intensity)
            output_name = f"brightness_{intensity}.jpg"
        
        case "2":
            transformed_matrix = flip_image(input_matrix)
            output_name = "flipped.jpg"
        
        case "3":
            transformed_matrix = negative_image(input_matrix)
            output_name = "negative.jpg"
        
        case "4":
            rotations = get_rotation_input()
            transformed_matrix = input_matrix
            for _ in range(rotations % 4): 
                transformed_matrix = rotate_image(transformed_matrix)
            output_name = f"rotated_{rotations * 90}.jpg"
        
        case "5":
            skew_x = get_float_input("Enter horizontal skew factor (-2 to 2): ", -2, 2)
            skew_y = get_float_input("Enter vertical skew factor (-2 to 2): ", -2, 2)
            transformed_matrix = skew_image(input_matrix, skew_x, skew_y)
            output_name = f"skewed_{skew_x}_{skew_y}.jpg"
        
        case "6":
            scale_x = get_float_input("Enter horizontal scale factor (0.1 to 5): ", 0.1, 5)
            scale_y = get_float_input("Enter vertical scale factor (0.1 to 5): ", 0.1, 5)
            transformed_matrix = scale_image(input_matrix, scale_x, scale_y)
            output_name = f"scaled_{scale_x}_{scale_y}.jpg"
        
        case _:
            print("Invalid choice!")
            return
    
    if transformed_matrix and output_name:
        save_image_matrix(transformed_matrix, output_name)
        print(f"\nTransformed image saved as: output/{output_name}")

if __name__ == "__main__":
    main()

