"""
Bounding box utilities for YOLO dataset preparation.
"""

import math


def adjust_bbox_coordinates(x_center, y_center, width, height, 
                           original_width, original_height, target_size):
    """
    Adjust bounding box coordinates for a resized image.
    
    Args:
        x_center, y_center, width, height: Normalized YOLO format coordinates (0-1)
        original_width, original_height: Dimensions of the original image
        target_size: Target size of the square image
        
    Returns:
        tuple: (new_x_center, new_y_center, new_width, new_height) - Adjusted normalized coordinates
    """
    # Calculate scaling factor
    scale = min(target_size / original_width, target_size / original_height)
    
    # Calculate new dimensions
    new_width_pixels = original_width * scale
    new_height_pixels = original_height * scale
    
    # Calculate padding
    pad_x = (target_size - new_width_pixels) / 2
    pad_y = (target_size - new_height_pixels) / 2
    
    # Convert original coordinates to pixel values
    x_center_px = x_center * original_width
    y_center_px = y_center * original_height
    width_px = width * original_width
    height_px = height * original_height
    
    # Adjust for scaling
    x_center_px_new = x_center_px * scale + pad_x
    y_center_px_new = y_center_px * scale + pad_y
    width_px_new = width_px * scale
    height_px_new = height_px * scale
    
    # Convert back to normalized coordinates for the square image
    new_x_center = x_center_px_new / target_size
    new_y_center = y_center_px_new / target_size
    new_width = width_px_new / target_size
    new_height = height_px_new / target_size
    
    return new_x_center, new_y_center, new_width, new_height


def expand_bbox(x_center, y_center, width, height, expand_percentage, target_size):
    """
    Expand a bounding box by a percentage while maintaining the center point.
    
    Args:
        x_center, y_center, width, height: Normalized YOLO format coordinates (0-1)
        expand_percentage: Percentage to expand the bounding box
        target_size: Size of the image (needed for boundary checks)
        
    Returns:
        tuple: (new_x_center, new_y_center, new_width, new_height) - Expanded coordinates
    """
    # Calculate expansion factor
    factor = 1.0 + (expand_percentage / 100.0)
    
    # Calculate new width and height
    new_width = width * factor
    new_height = height * factor
    
    # Ensure the bounding box stays within image boundaries
    # Convert to top-left, bottom-right coordinates for easier boundary checking
    half_width = new_width / 2
    half_height = new_height / 2
    
    x_min = x_center - half_width
    y_min = y_center - half_height
    x_max = x_center + half_width
    y_max = y_center + half_height
    
    # Clamp coordinates to image boundaries (0-1)
    x_min = max(0, min(x_min, 1.0))
    y_min = max(0, min(y_min, 1.0))
    x_max = max(0, min(x_max, 1.0))
    y_max = max(0, min(y_max, 1.0))
    
    # Convert back to center, width, height format
    new_width = x_max - x_min
    new_height = y_max - y_min
    new_x_center = x_min + new_width / 2
    new_y_center = y_min + new_height / 2
    
    return new_x_center, new_y_center, new_width, new_height