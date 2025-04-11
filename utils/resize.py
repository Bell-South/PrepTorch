"""
Image resizing utilities for YOLO dataset preparation.
"""

from PIL import Image
import numpy as np


def resize_image(image_path, target_size):
    """
    Resize an image to the target size while maintaining aspect ratio through padding.
    
    Args:
        image_path (str): Path to the input image
        target_size (int): Target size for both dimensions (square output)
        
    Returns:
        tuple: (resized_image, original_width, original_height)
            - resized_image: PIL Image object of the resized image
            - original_width: Width of the original image
            - original_height: Height of the original image
    """
    # Open the image
    img = Image.open(image_path)
    original_width, original_height = img.size
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_size / original_width, target_size / original_height)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new square image with black background
    square_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    
    # Calculate position to paste the resized image (centered)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # Paste the resized image onto the square canvas
    square_img.paste(resized_img, (paste_x, paste_y))
    
    return square_img, original_width, original_height