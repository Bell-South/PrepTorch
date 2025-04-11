#!/usr/bin/env python3
"""
YOLO Dataset Preparation Tool

This tool processes a YOLO-formatted dataset by:
- Resizing images to a square format
- Adjusting and expanding bounding boxes
- Splitting the dataset into train/val sets
- Generating preview images with drawn bounding boxes
- Creating a summary report
"""

import os
import argparse
import shutil
import random
from pathlib import Path
import glob
from tqdm import tqdm

# Import utility modules
from utils.resize import resize_image
from utils.bbox import adjust_bbox_coordinates, expand_bbox
from utils.visualization import draw_bboxes_on_image
from utils.report import generate_report


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process YOLO-formatted datasets for object detection training."
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
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--size", 
        type=int, 
        choices=[640, 1280], 
        default=640, 
        help="Target size for square images (640 or 1280)"
    )
    parser.add_argument(
        "--expand_percentage", 
        type=float, 
        default=20.0, 
        help="Percentage to expand bounding boxes (e.g., 20.0 for 20%%)"
    )
    parser.add_argument(
        "--train_split", 
        type=float, 
        default=0.7, 
        help="Proportion of images to use for training (default: 0.7)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducible train/val splits"
    )
    
    return parser.parse_args()


def create_directory_structure(output_dir):
    """Create the required directory structure for the output."""
    # Create main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train and validation directories
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")
    preview_dir = os.path.join(output_dir, "preview")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    
    return {
        "train_images": train_images_dir,
        "train_labels": train_labels_dir,
        "val_images": val_images_dir,
        "val_labels": val_labels_dir,
        "preview": preview_dir
    }


def process_dataset(args):
    """Process the dataset according to the specified arguments."""
    input_dir = args.input_dir
    output_dir = args.output_dir
    target_size = args.size
    expand_percentage = args.expand_percentage
    train_split = args.train_split
    random_seed = args.seed
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Verify input directories exist
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError(f"Input directory must contain 'images' and 'labels' subfolders")
    
    # Create output directory structure
    directories = create_directory_structure(output_dir)
    
    # Get list of image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*.{ext}")))
    
    # Create train/val split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    # Process images and labels
    class_counts = {}  # To track object counts per class
    total_objects = 0
    
    # Process training images
    print("Processing training images...")
    for img_path in tqdm(train_images):
        process_single_image(
            img_path, labels_dir, 
            directories["train_images"], directories["train_labels"], directories["preview"],
            target_size, expand_percentage, class_counts
        )
    
    # Process validation images
    print("Processing validation images...")
    for img_path in tqdm(val_images):
        process_single_image(
            img_path, labels_dir, 
            directories["val_images"], directories["val_labels"], directories["preview"],
            target_size, expand_percentage, class_counts
        )
    
    # Generate summary report
    total_objects = sum(class_counts.values())
    report_path = os.path.join(output_dir, "dataset_summary.txt")
    generate_report(
        report_path, 
        len(train_images), 
        len(val_images), 
        class_counts, 
        total_objects
    )
    
    print(f"Dataset processing complete. Summary saved to {report_path}")


def process_single_image(img_path, labels_dir, output_images_dir, output_labels_dir, preview_dir, 
                         target_size, expand_percentage, class_counts):
    """Process a single image and its label file."""
    # Get the base filename without extension
    img_basename = os.path.basename(img_path)
    filename_no_ext = os.path.splitext(img_basename)[0]
    
    # Corresponding label file
    label_path = os.path.join(labels_dir, f"{filename_no_ext}.txt")
    
    # Skip if label file doesn't exist
    if not os.path.exists(label_path):
        print(f"Warning: No label file found for {img_basename}, skipping.")
        return
    
    # Resize image
    resized_img, original_width, original_height = resize_image(img_path, target_size)
    
    # Save resized image
    output_img_path = os.path.join(output_images_dir, img_basename)
    resized_img.save(output_img_path)
    
    # Process label file
    with open(label_path, 'r') as f:
        label_lines = f.readlines()
    
    new_label_lines = []
    bboxes_for_preview = []
    
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Warning: Invalid label format in {label_path}, skipping line: {line}")
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Adjust coordinates for resized image
        adjusted_bbox = adjust_bbox_coordinates(
            x_center, y_center, width, height,
            original_width, original_height, target_size
        )
        
        # Expand bounding box
        expanded_bbox = expand_bbox(
            adjusted_bbox[0], adjusted_bbox[1], adjusted_bbox[2], adjusted_bbox[3],
            expand_percentage, target_size
        )
        
        # Format new label line
        new_line = f"{class_id} {expanded_bbox[0]:.6f} {expanded_bbox[1]:.6f} {expanded_bbox[2]:.6f} {expanded_bbox[3]:.6f}\n"
        new_label_lines.append(new_line)
        
        # Store bbox info for preview
        bboxes_for_preview.append((class_id, expanded_bbox))
        
        # Update class counts
        if class_id not in class_counts:
            class_counts[class_id] = 0
        class_counts[class_id] += 1
    
    # Save new label file
    output_label_path = os.path.join(output_labels_dir, f"{filename_no_ext}.txt")
    with open(output_label_path, 'w') as f:
        f.writelines(new_label_lines)
    
    # Create preview image
    preview_img_path = os.path.join(preview_dir, f"{filename_no_ext}_preview.jpg")
    draw_bboxes_on_image(resized_img.copy(), bboxes_for_preview, preview_img_path)


def main():
    """Main function to run the tool."""
    args = parse_arguments()
    process_dataset(args)


if __name__ == "__main__":
    main()