"""
Visualization utilities for YOLO dataset preparation.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random


# Define a color palette for different classes
# We'll use a fixed set of distinct colors to visualize different classes
def get_color_palette(num_classes=80):
    """Generate a color palette for visualization."""
    random.seed(42)  # For reproducible colors
    colors = []
    for i in range(num_classes):
        # Generate distinct, bright colors
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        colors.append((r, g, b))
    return colors


def draw_bboxes_on_image(image, bboxes, output_path):
    """
    Draw bounding boxes on an image and save it.
    
    Args:
        image: PIL Image object
        bboxes: List of tuples (class_id, (x_center, y_center, width, height))
        output_path: Path to save the visualized image
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    colors = get_color_palette()
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    for class_id, bbox in bboxes:
        x_center, y_center, w, h = bbox
        
        # Convert normalized coordinates to pixel values
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        w_px = int(w * width)
        h_px = int(h * height)
        
        # Calculate corner coordinates
        x1 = x_center_px - w_px // 2
        y1 = y_center_px - h_px // 2
        x2 = x_center_px + w_px // 2
        y2 = y_center_px + h_px // 2
        
        # Ensure coordinates are within image boundaries
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw class label
        label = f"Class {class_id}"
        text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:4]
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill=color)
        draw.text((x1, y1), label, fill="white", font=font)
    
    # Save the image
    image.save(output_path)