#!/usr/bin/env python3
"""
Bounding Box Comparison Tool

This script compares original and processed images with their bounding boxes side by side.
It's useful for debugging and verifying that the dataset processing is working correctly.
"""

import os
import argparse
import glob
from PIL import Image, ImageDraw, ImageFont
import random


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare original and processed images with bounding boxes."
    )
    parser.add_argument(
        "--original_dir", 
        type=str, 
        required=True, 
        help="Original dataset directory with 'images' and 'labels' folders"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        required=True, 
        help="Processed dataset directory with 'train/images', 'train/labels', etc."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Output directory for comparison images"
    )
    parser.add_argument(
        "--image_name", 
        type=str, 
        help="Specific image name to compare (without extension)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5, 
        help="Number of random samples to compare if no specific image is provided"
    )
    
    return parser.parse_args()


def get_color_palette(num_classes=80):
    """Generate a color palette for visualization."""
    random.seed(42)  # For reproducible colors
    colors = []
    for i in range(num_classes):
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        colors.append((r, g, b))
    return colors


def find_processed_image(image_name, processed_dir):
    """Find the processed image in train or val directories."""
    # Try train directory first
    train_path = os.path.join(processed_dir, "train", "images", f"{image_name}.jpg")
    if os.path.exists(train_path):
        return train_path, os.path.join(processed_dir, "train", "labels", f"{image_name}.txt")
    
    # Try other image extensions
    for ext in ['jpeg', 'png', 'bmp']:
        train_path = os.path.join(processed_dir, "train", "images", f"{image_name}.{ext}")
        if os.path.exists(train_path):
            return train_path, os.path.join(processed_dir, "train", "labels", f"{image_name}.txt")
    
    # Try val directory
    val_path = os.path.join(processed_dir, "val", "images", f"{image_name}.jpg")
    if os.path.exists(val_path):
        return val_path, os.path.join(processed_dir, "val", "labels", f"{image_name}.txt")
    
    # Try other image extensions
    for ext in ['jpeg', 'png', 'bmp']:
        val_path = os.path.join(processed_dir, "val", "images", f"{image_name}.{ext}")
        if os.path.exists(val_path):
            return val_path, os.path.join(processed_dir, "val", "labels", f"{image_name}.txt")
    
    return None, None


def draw_bboxes_on_image(image_path, label_path):
    """Draw bounding boxes on an image and return the annotated image."""
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None
    
    width, height = image.size
    draw = ImageDraw.Draw(image)
    colors = get_color_palette()
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    
    # Read label file if it exists
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
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
                
                # Draw rectangle
                color = colors[class_id % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw class label
                label = f"C{class_id}"
                draw.rectangle([x1, y1, x1 + 25, y1 + 15], fill=color)
                draw.text((x1 + 2, y1), label, fill="white", font=font)
                
        except Exception as e:
            print(f"Error processing label file {label_path}: {e}")
    
    return image


def create_comparison(original_dir, processed_dir, output_dir, image_name=None, num_samples=5):
    """Create side-by-side comparison images."""
    os.makedirs(output_dir, exist_ok=True)
    
    if image_name:
        # Compare specific image
        compare_single_image(original_dir, processed_dir, output_dir, image_name)
    else:
        # Compare random samples
        compare_random_samples(original_dir, processed_dir, output_dir, num_samples)


def compare_single_image(original_dir, processed_dir, output_dir, image_name):
    """Compare a specific image from original and processed datasets."""
    # Find image files
    original_img_path = None
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        path = os.path.join(original_dir, "images", f"{image_name}.{ext}")
        if os.path.exists(path):
            original_img_path = path
            break
    
    if not original_img_path:
        print(f"Error: Original image {image_name}.* not found")
        return
    
    original_label_path = os.path.join(original_dir, "labels", f"{image_name}.txt")
    processed_img_path, processed_label_path = find_processed_image(image_name, processed_dir)
    
    if not processed_img_path:
        print(f"Error: Processed image for {image_name} not found")
        return
    
    # Draw bounding boxes
    original_with_boxes = draw_bboxes_on_image(original_img_path, original_label_path)
    processed_with_boxes = draw_bboxes_on_image(processed_img_path, processed_label_path)
    
    if original_with_boxes and processed_with_boxes:
        # Create side-by-side comparison
        max_height = max(original_with_boxes.height, processed_with_boxes.height)
        comparison = Image.new('RGB', (original_with_boxes.width + processed_with_boxes.width, max_height))
        comparison.paste(original_with_boxes, (0, 0))
        comparison.paste(processed_with_boxes, (original_with_boxes.width, 0))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((10, 10), "Original", fill="white", font=font)
        draw.text((original_with_boxes.width + 10, 10), "Processed", fill="white", font=font)
        
        # Save comparison
        comparison_path = os.path.join(output_dir, f"{image_name}_comparison.jpg")
        comparison.save(comparison_path)
        print(f"Comparison saved to {comparison_path}")


def compare_random_samples(original_dir, processed_dir, output_dir, num_samples):
    """Compare random samples from the dataset."""
    # Get list of image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_files.extend(glob.glob(os.path.join(original_dir, "images", f"*.{ext}")))
    
    if not image_files:
        print(f"No image files found in {original_dir}/images")
        return
    
    # Select random samples
    if len(image_files) <= num_samples:
        samples = image_files
    else:
        samples = random.sample(image_files, num_samples)
    
    # Process each sample
    for img_path in samples:
        img_basename = os.path.basename(img_path)
        image_name = os.path.splitext(img_basename)[0]
        print(f"Comparing {image_name}...")
        compare_single_image(original_dir, processed_dir, output_dir, image_name)


def main():
    """Main function to run the comparison tool."""
    args = parse_arguments()
    create_comparison(
        args.original_dir, 
        args.processed_dir, 
        args.output_dir, 
        args.image_name, 
        args.num_samples
    )


if __name__ == "__main__":
    main()