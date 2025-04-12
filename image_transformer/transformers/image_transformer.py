import numpy as np
from PIL import Image

class ImageTransformer:
    def __init__(self, image_path):
        """
        Initialize the ImageTransformer with the path to an image file.
        
        Args:
            image_path (str): Path to the image file
        """
        self.image_path = image_path
        self.image = None
        self.matrix = None
        
    def load_image(self):
        """
        Load the image from the specified path.
        
        Returns:
            Image: The loaded PIL Image object
        """
        try:
            self.image = Image.open(self.image_path)
            return self.image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def to_matrix(self):
        """
        Convert the loaded image to a numpy matrix (array).
        
        Returns:
            numpy.ndarray: The image as a numpy array
        """
        if self.image is None:
            self.load_image()
            
        if self.image:
            # Convert image to numpy array
            self.matrix = np.array(self.image)
            return self.matrix
        return None
    
    def from_matrix_to_image(self):
        """
        Convert the current matrix back to a PIL Image.
        
        Returns:
            PIL.Image: The image created from the matrix
        """
        if self.matrix is not None:
            # Ensure values are in valid range
            matrix = np.clip(self.matrix, 0, 255).astype(np.uint8)
            self.image = Image.fromarray(matrix)
            return self.image
        return None
        
    def save_image(self, output_path):
        """
        Save the current image to a file.
        
        Args:
            output_path (str): Path where the image will be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.image is None:
            self.from_matrix_to_image()
            
        if self.image:
            try:
                self.image.save(output_path)
                return True
            except Exception as e:
                print(f"Error saving image: {e}")
                return False
        return False
    
    def resize(self, new_width, new_height):
        """
        Resize the image using manual nearest neighbor interpolation.
        
        Args:
            new_width (int): Target width
            new_height (int): Target height
            
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            original_height, original_width = self.matrix.shape[0], self.matrix.shape[1]
            
            # Create empty matrix for resized image
            if len(self.matrix.shape) == 3:  # Color image
                channels = self.matrix.shape[2]
                resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)
            else:  # Grayscale
                resized = np.zeros((new_height, new_width), dtype=np.uint8)
            
            # Calculate scaling factors
            x_ratio = original_width / new_width
            y_ratio = original_height / new_height
            
            # Manual nearest neighbor sampling
            for y in range(new_height):
                for x in range(new_width):
                    # Find corresponding position in original image
                    orig_x = min(int(x * x_ratio), original_width - 1)
                    orig_y = min(int(y * y_ratio), original_height - 1)
                    
                    # Copy pixel value
                    resized[y, x] = self.matrix[orig_y, orig_x]
            
            self.matrix = resized
            self.from_matrix_to_image()
        
        return self
    
    def rotate(self, angle_degrees):
        """
        Rotate the image by the specified angle using a rotation matrix.
        
        Args:
            angle_degrees (float): Rotation angle in degrees (counter-clockwise)
            
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # Convert angle to radians
            angle_radians = np.radians(angle_degrees)
            
            # Get image dimensions
            height, width = self.matrix.shape[0], self.matrix.shape[1]
            
            # Calculate the center of the image
            center_x, center_y = width // 2, height // 2
            
            # Create rotation matrix
            cos_theta = np.cos(angle_radians)
            sin_theta = np.sin(angle_radians)
            
            # Calculate new image dimensions to fit the rotated image
            new_width = int(abs(width * cos_theta) + abs(height * sin_theta))
            new_height = int(abs(width * sin_theta) + abs(height * cos_theta))
            
            # Create a blank canvas for the rotated image
            if len(self.matrix.shape) == 3:  # Color image
                channels = self.matrix.shape[2]
                rotated = np.zeros((new_height, new_width, channels), dtype=np.uint8)
            else:  # Grayscale
                rotated = np.zeros((new_height, new_width), dtype=np.uint8)
            
            # Calculate the center of the new image
            new_center_x, new_center_y = new_width // 2, new_height // 2
            
            # Perform the rotation by applying the inverse transformation
            for y in range(new_height):
                for x in range(new_width):
                    # Convert to coordinates relative to the center of the rotated image
                    rel_x = x - new_center_x
                    rel_y = y - new_center_y
                    
                    # Apply inverse rotation
                    orig_x = int(rel_x * cos_theta - rel_y * sin_theta + center_x)
                    orig_y = int(rel_x * sin_theta + rel_y * cos_theta + center_y)
                    
                    # Check if the original coordinates are within the image bounds
                    if 0 <= orig_x < width and 0 <= orig_y < height:
                        rotated[y, x] = self.matrix[orig_y, orig_x]
            
            self.matrix = rotated
            self.from_matrix_to_image()
        
        return self
    
    def flip_horizontal(self):
        """
        Flip the image horizontally using matrix manipulation.
        
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # Flip by reversing the order of columns
            self.matrix = self.matrix[:, ::-1]
            self.from_matrix_to_image()
        
        return self
    
    def flip_vertical(self):
        """
        Flip the image vertically using matrix manipulation.
        
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # Flip by reversing the order of rows
            self.matrix = self.matrix[::-1, :]
            self.from_matrix_to_image()
        
        return self
    
    def to_grayscale(self):
        """
        Convert the image to grayscale using weighted RGB conversion.
        
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None and len(self.matrix.shape) == 3:
            # Use standard RGB to grayscale formula: 0.299*R + 0.587*G + 0.114*B
            weights = np.array([0.299, 0.587, 0.114])
            grayscale = np.dot(self.matrix[..., :3], weights)
            self.matrix = grayscale.astype(np.uint8)
            self.from_matrix_to_image()
        
        return self
    
    def adjust_brightness(self, factor):
        """
        Adjust the brightness of the image using matrix scaling.
        
        Args:
            factor (float): Brightness factor (0.0 darkens, 1.0 original, >1.0 brightens)
            
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # Scale all pixel values by the factor
            adjusted = self.matrix.astype(float) * factor
            # Clip values to valid range and convert back to uint8
            self.matrix = np.clip(adjusted, 0, 255).astype(np.uint8)
            self.from_matrix_to_image()
        
        return self
    
    def adjust_contrast(self, factor):
        """
        Adjust the contrast of the image using matrix scaling around the mean.
        
        Args:
            factor (float): Contrast factor (0.0 solid gray, 1.0 original, >1.0 more contrast)
            
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # Calculate mean intensity as the pivot point
            mean = np.mean(self.matrix)
            
            # Apply contrast adjustment: (pixel - mean) * factor + mean
            adjusted = (self.matrix.astype(float) - mean) * factor + mean
            # Clip values to valid range and convert back to uint8
            self.matrix = np.clip(adjusted, 0, 255).astype(np.uint8)
            self.from_matrix_to_image()
        
        return self
    
    def blur_box(self, kernel_size=3):
        """
        Apply box blur using a manually implemented convolution.
        
        Args:
            kernel_size (int): Size of the blur kernel (odd number)
            
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            # Create a box blur kernel
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            
            # Get original dimensions
            if len(self.matrix.shape) == 3:  # Color image
                height, width, channels = self.matrix.shape
                # Create output matrix
                blurred = np.zeros_like(self.matrix)
                
                # Apply convolution to each channel
                for c in range(channels):
                    for y in range(height):
                        for x in range(width):
                            # Define the kernel boundaries
                            k_radius = kernel_size // 2
                            y_min = max(0, y - k_radius)
                            y_max = min(height, y + k_radius + 1)
                            x_min = max(0, x - k_radius)
                            x_max = min(width, x + k_radius + 1)
                            
                            # Extract neighborhood
                            neighborhood = self.matrix[y_min:y_max, x_min:x_max, c]
                            
                            # Apply kernel (average of neighborhood)
                            blurred[y, x, c] = np.mean(neighborhood)
            else:  # Grayscale
                height, width = self.matrix.shape
                # Create output matrix
                blurred = np.zeros_like(self.matrix)
                
                # Apply convolution
                for y in range(height):
                    for x in range(width):
                        # Define the kernel boundaries
                        k_radius = kernel_size // 2
                        y_min = max(0, y - k_radius)
                        y_max = min(height, y + k_radius + 1)
                        x_min = max(0, x - k_radius)
                        x_max = min(width, x + k_radius + 1)
                        
                        # Extract neighborhood
                        neighborhood = self.matrix[y_min:y_max, x_min:x_max]
                        
                        # Apply kernel (average of neighborhood)
                        blurred[y, x] = np.mean(neighborhood)
            
            self.matrix = blurred.astype(np.uint8)
            self.from_matrix_to_image()
        
        return self
    
    def detect_edges_sobel(self):
        """
        Apply Sobel edge detection using manual convolution.
        
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # If image is color, convert to grayscale first
            if len(self.matrix.shape) == 3:
                # Use RGB to grayscale conversion
                weights = np.array([0.299, 0.587, 0.114])
                gray = np.dot(self.matrix[..., :3], weights).astype(np.uint8)
            else:
                gray = self.matrix.copy()
            
            # Define Sobel kernels
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            height, width = gray.shape
            edges = np.zeros_like(gray)
            
            # Apply convolution
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    # Extract 3x3 neighborhood
                    neighborhood = gray[y-1:y+2, x-1:x+2]
                    
                    # Apply Sobel operators
                    gx = np.sum(neighborhood * sobel_x)
                    gy = np.sum(neighborhood * sobel_y)
                    
                    # Compute gradient magnitude
                    magnitude = np.sqrt(gx**2 + gy**2)
                    edges[y, x] = np.clip(magnitude, 0, 255)
            
            self.matrix = edges
            self.from_matrix_to_image()
        
        return self
    
    def threshold(self, threshold_value=128):
        """
        Apply binary threshold to the image.
        
        Args:
            threshold_value (int): Threshold value (0-255)
            
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # If image is color, convert to grayscale first
            if len(self.matrix.shape) == 3:
                # Use RGB to grayscale conversion
                weights = np.array([0.299, 0.587, 0.114])
                gray = np.dot(self.matrix[..., :3], weights).astype(np.uint8)
            else:
                gray = self.matrix.copy()
            
            # Apply threshold
            binary = np.zeros_like(gray)
            binary[gray >= threshold_value] = 255
            
            self.matrix = binary
            self.from_matrix_to_image()
        
        return self
    
    def crop(self, left, top, right, bottom):
        """
        Crop the image to the specified rectangle.
        
        Args:
            left (int): Left coordinate
            top (int): Top coordinate
            right (int): Right coordinate
            bottom (int): Bottom coordinate
            
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None:
            # Ensure coordinates are within bounds
            height, width = self.matrix.shape[0], self.matrix.shape[1]
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            
            # Extract the sub-matrix
            self.matrix = self.matrix[top:bottom, left:right]
            self.from_matrix_to_image()
        
        return self
    
    def apply_sepia(self):
        """
        Apply sepia filter using a transformation matrix.
        
        Returns:
            ImageTransformer: Self for method chaining
        """
        if self.matrix is None:
            self.to_matrix()
            
        if self.matrix is not None and len(self.matrix.shape) == 3:
            # Sepia transformation matrix
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            
            # Reshape the image for matrix multiplication
            height, width, _ = self.matrix.shape
            flat_img = self.matrix.reshape(-1, 3)
            
            # Apply the sepia transformation
            sepia_flat = np.dot(flat_img, sepia_matrix.T)
            
            # Clip values to valid range
            sepia_flat = np.clip(sepia_flat, 0, 255).astype(np.uint8)
            
            # Reshape back to original shape
            self.matrix = sepia_flat.reshape(height, width, 3)
            self.from_matrix_to_image()
        
        return self 