#!/usr/bin/env python3
"""
Original Dataset Preview Tool

This script generates preview images for the original YOLO dataset,
drawing bounding boxes on the original images to visualize the labels.
"""

import os
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate preview images for original YOLO-formatted dataset."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Input directory containing 'images' and 'labels' folders"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Output directory for preview images"
    )
    
    return parser.parse_args()


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


def draw_bboxes_on_original_image(img_path, label_path, output_path):
    """
    Draw original bounding boxes on an image and save it.
    
    Args:
        img_path: Path to the original image
        label_path: Path to the YOLO format label file
        output_path: Path to save the visualized image
    """
    # Open the image
    try:
        image = Image.open(img_path)
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return
    
    width, height = image.size
    draw = ImageDraw.Draw(image)
    colors = get_color_palette()
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # Read and parse label file
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return
    
    # Draw bounding boxes for each label
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Warning: Invalid label format in {label_path}, skipping line: {line}")
            continue
        
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
            
            # Convert normalized coordinates to pixel values
            x_center_px = int(x_center * width)
            y_center_px = int(y_center * height)
            w_px = int(box_width * width)
            h_px = int(box_height * height)
            
            # Calculate corner coordinates
            x1 = x_center_px - w_px // 2
            y1 = y_center_px - h_px // 2
            x2 = x_center_px + w_px // 2
            y2 = y_center_px + h_px // 2
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw class label
            label = f"Class {class_id}"
            text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:4]
            draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill=color)
            draw.text((x1, y1), label, fill="white", font=font)
            
        except Exception as e:
            print(f"Error processing label in {label_path}: {e}")
            continue
    
    # Save the image
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
    except Exception as e:
        print(f"Error saving preview image to {output_path}: {e}")


def process_dataset(input_dir, output_dir):
    """
    Process all images in the dataset and create preview images.
    
    Args:
        input_dir: Input directory containing 'images' and 'labels' folders
        output_dir: Output directory for preview images
    """
    # Verify input directories exist
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(images_dir, f"*.{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    # Count labels and missing label files
    total_images = len(image_files)
    images_with_labels = 0
    missing_labels = 0
    
    print(f"Processing {total_images} images...")
    for img_path in tqdm(image_files):
        # Get the base filename without extension
        img_basename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(img_basename)[0]
        
        # Construct label path
        label_path = os.path.join(labels_dir, f"{filename_no_ext}.txt")
        
        # Create preview path
        preview_path = os.path.join(output_dir, f"{filename_no_ext}_original.jpg")
        
        # Check if label file exists
        if os.path.exists(label_path):
            images_with_labels += 1
            draw_bboxes_on_original_image(img_path, label_path, preview_path)
        else:
            missing_labels += 1
            print(f"Warning: No label file found for {img_basename}")
    
    print(f"\nSummary:")
    print(f"Total images processed: {total_images}")
    print(f"Images with labels: {images_with_labels}")
    print(f"Images missing labels: {missing_labels}")
    print(f"\nPreview images saved to: {output_dir}")


def main():
    """Main function to run the preview tool."""
    args = parse_arguments()
    process_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()