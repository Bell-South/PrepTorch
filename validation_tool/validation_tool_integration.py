#!/usr/bin/env python3
"""
This file integrates the street light validation module into the main validation tool.
This is a complete standalone script that incorporates all necessary parts from validation_tool.py.
"""

import argparse
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from tqdm import tqdm
import math
import json
from collections import defaultdict

# Import the street light validation module
from streetlight_validation import check_streetlight_boxes, visualize_streetlight_validation, integrate_with_validation_tool


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate bounding boxes in YOLO-formatted dataset and identify potential issues."
    )
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        required=True, 
        help="Dataset directory with 'images' and 'labels' folders"
    )
    parser.add_argument(
        "--preview_dir", 
        type=str, 
        required=False,
        help="Directory containing preview images (optional)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Output directory for issue reports and visualizations"
    )
    parser.add_argument(
        "--min_box_size", 
        type=float, 
        default=0.01, 
        help="Minimum relative box size (area) in image (0-1)"
    )
    parser.add_argument(
        "--max_box_size", 
        type=float, 
        default=0.9, 
        help="Maximum relative box size (area) in image (0-1)"
    )
    parser.add_argument(
        "--overlap_threshold", 
        type=float, 
        default=0.7, 
        help="IoU threshold for overlapping boxes (0-1)"
    )
    parser.add_argument(
        "--edge_threshold", 
        type=float, 
        default=0.01, 
        help="Distance from edge to flag edge boxes (0-1)"
    )
    parser.add_argument(
        "--aspect_ratio_threshold", 
        type=float, 
        default=2.0, 
        help="Threshold for flagging unusual aspect ratios"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output for debugging"
    )
    
    # Add street light validation arguments
    parser.add_argument(
        "--validate_streetlights", 
        action="store_true",
        help="Enable validation of street light annotations"
    )
    parser.add_argument(
        "--streetlight_class_id", 
        type=int, 
        default=None,
        help="Class ID for street lights (if not provided, will try to detect them)"
    )
    parser.add_argument(
        "--streetlight_confidence", 
        type=float, 
        default=0.5,
        help="Confidence threshold for street light detection (0-1)"
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


def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1: tuple (x_center, y_center, width, height) - normalized coordinates
        box2: tuple (x_center, y_center, width, height) - normalized coordinates
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Convert from center format to min/max format
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2
    
    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2
    
    # Calculate intersection area
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate union area
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    # Calculate IoU
    if union <= 0:
        return 0
    return intersection / union


def is_on_edge(box, threshold):
    """
    Check if a bounding box is close to the edge of the image.
    
    Args:
        box: tuple (x_center, y_center, width, height) - normalized coordinates
        threshold: distance from edge to consider (0-1)
        
    Returns:
        bool: True if box is on edge, False otherwise
    """
    x_center, y_center, width, height = box
    half_w = width / 2
    half_h = height / 2
    
    # Check if box edges are near image boundaries
    left_edge = x_center - half_w
    right_edge = x_center + half_w
    top_edge = y_center - half_h
    bottom_edge = y_center + half_h
    
    return (
        left_edge < threshold or 
        right_edge > (1 - threshold) or 
        top_edge < threshold or 
        bottom_edge > (1 - threshold)
    )


def check_label_issues(image_path, label_path, args, verbose=False):
    """
    Check a single image and its labels for potential issues.
    
    Args:
        image_path: Path to the image file
        label_path: Path to the corresponding YOLO format label file
        args: Command line arguments with validation thresholds
        verbose: Enable verbose output for debugging
        
    Returns:
        dict: Dictionary with issues found in this image
    """
    issues = {
        "small_boxes": [],
        "large_boxes": [],
        "overlapping_boxes": [],
        "edge_boxes": [],
        "unusual_aspect_ratio": [],
        "has_issues": False
    }
    
    # Skip if label file doesn't exist
    if not os.path.exists(label_path):
        if verbose:
            print(f"Label file not found: {label_path}")
        return issues, []
    
    # Open image to get dimensions
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        if verbose:
            print(f"Image {image_path} opened successfully. Size: {img_width}x{img_height}")
        img.close()
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return issues, []
    
    # Read label file
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if verbose:
            print(f"Label file {label_path} read successfully. {len(lines)} lines found.")
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return issues, []
    
    # Parse labels
    boxes = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            if verbose:
                print(f"Invalid format in label file {label_path}, line {i+1}. Expected 5 values, got {len(parts)}")
            continue
        
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            boxes.append({
                "id": i,
                "class_id": class_id,
                "coords": (x_center, y_center, width, height),
                "aspect_ratio": width / height if height > 0 else float('inf')
            })
            
            if verbose:
                print(f"Parsed box {i+1}: class={class_id}, coords=({x_center:.4f}, {y_center:.4f}, {width:.4f}, {height:.4f})")
            
        except Exception as e:
            print(f"Error parsing label in {label_path}, line {i+1}: {e}")
            continue
    
    # Check for issues with each box
    for i, box in enumerate(boxes):
        coords = box["coords"]
        box_area = coords[2] * coords[3]  # width * height
        
        if verbose:
            print(f"Checking box {i+1}: class={box['class_id']}, area={box_area:.6f}")
        
        # Check box size
        if box_area < args.min_box_size:
            issues["small_boxes"].append({
                "box_id": i,
                "class_id": box["class_id"],
                "area": box_area,
                "coords": coords
            })
            issues["has_issues"] = True
            if verbose:
                print(f"  ISSUE: Small box (area={box_area:.6f} < {args.min_box_size})")
            
        if box_area > args.max_box_size:
            issues["large_boxes"].append({
                "box_id": i,
                "class_id": box["class_id"],
                "area": box_area,
                "coords": coords
            })
            issues["has_issues"] = True
            if verbose:
                print(f"  ISSUE: Large box (area={box_area:.6f} > {args.max_box_size})")
            
        # Check if box is on edge
        if is_on_edge(coords, args.edge_threshold):
            issues["edge_boxes"].append({
                "box_id": i,
                "class_id": box["class_id"],
                "coords": coords
            })
            issues["has_issues"] = True
            if verbose:
                print(f"  ISSUE: Box on edge (threshold={args.edge_threshold})")
            
        # Check aspect ratio (will compare to class average later)
        box["aspect_ratio_info"] = {
            "box_id": i,
            "class_id": box["class_id"],
            "aspect_ratio": box["aspect_ratio"],
            "coords": coords
        }
    
    # Check for overlapping boxes
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            iou = calculate_iou(boxes[i]["coords"], boxes[j]["coords"])
            if verbose and iou > 0:
                print(f"  IoU between box {i+1} and box {j+1}: {iou:.4f}")
                
            if iou > args.overlap_threshold:
                issues["overlapping_boxes"].append({
                    "box1_id": i,
                    "box2_id": j,
                    "box1_class": boxes[i]["class_id"],
                    "box2_class": boxes[j]["class_id"],
                    "iou": iou,
                    "box1_coords": boxes[i]["coords"],
                    "box2_coords": boxes[j]["coords"]
                })
                issues["has_issues"] = True
                if verbose:
                    print(f"  ISSUE: Overlapping boxes {i+1} and {j+1} (IoU={iou:.4f} > {args.overlap_threshold})")
    
    return issues, [box["aspect_ratio_info"] for box in boxes]


def collect_class_aspect_ratios(dataset_issues, verbose=False):
    """
    Collect aspect ratio statistics for each class to identify outliers.
    
    Args:
        dataset_issues: Dictionary with issues for all images
        verbose: Enable verbose output for debugging
        
    Returns:
        dict: Class statistics for aspect ratios
    """
    if verbose:
        print("\nCollecting class aspect ratio statistics...")
        
    class_stats = defaultdict(list)
    
    # Collect all aspect ratios by class
    for img_path, (issues, boxes) in dataset_issues.items():
        for box in boxes:
            class_id = box["class_id"]
            aspect_ratio = box["aspect_ratio"]
            class_stats[class_id].append({
                "image": img_path,
                "aspect_ratio": aspect_ratio,
                "box_id": box["box_id"],
                "coords": box["coords"]
            })
    
    # Calculate statistics for each class
    class_metrics = {}
    for class_id, ratios in class_stats.items():
        if not ratios:
            continue
            
        aspect_values = [r["aspect_ratio"] for r in ratios]
        mean_ratio = np.mean(aspect_values)
        std_ratio = np.std(aspect_values)
        
        class_metrics[class_id] = {
            "count": len(ratios),
            "mean_aspect_ratio": mean_ratio,
            "std_aspect_ratio": std_ratio,
            "min_aspect_ratio": np.min(aspect_values),
            "max_aspect_ratio": np.max(aspect_values),
            "boxes": ratios
        }
        
        if verbose:
            print(f"Class {class_id}: {len(ratios)} objects, mean aspect ratio: {mean_ratio:.4f}±{std_ratio:.4f}")
    
    return class_metrics


def identify_aspect_ratio_outliers(dataset_issues, class_metrics, threshold, verbose=False):
    """
    Identify boxes with unusual aspect ratios compared to their class average.
    
    Args:
        dataset_issues: Dictionary with issues for all images
        class_metrics: Dictionary with aspect ratio statistics by class
        threshold: Z-score threshold for flagging unusual aspect ratios
        verbose: Enable verbose output for debugging
        
    Returns:
        dict: Updated dataset_issues with unusual aspect ratios marked
    """
    if verbose:
        print("\nIdentifying aspect ratio outliers...")
        
    outlier_count = 0
    
    for img_path, (issues, boxes) in dataset_issues.items():
        unusual_boxes = []
        
        for box in boxes:
            class_id = box["class_id"]
            aspect_ratio = box["aspect_ratio"]
            
            # Skip if we don't have enough samples for this class
            if class_id not in class_metrics or class_metrics[class_id]["count"] < 5:
                continue
                
            # Calculate z-score
            mean = class_metrics[class_id]["mean_aspect_ratio"]
            std = class_metrics[class_id]["std_aspect_ratio"]
            
            # Avoid division by zero
            if std == 0:
                continue
                
            z_score = abs((aspect_ratio - mean) / std)
            
            # Flag unusual aspect ratios
            if z_score > threshold:
                unusual_boxes.append({
                    "box_id": box["box_id"],
                    "class_id": class_id,
                    "aspect_ratio": aspect_ratio,
                    "mean_class_ratio": mean,
                    "z_score": z_score,
                    "coords": box["coords"]
                })
                issues["has_issues"] = True
                outlier_count += 1
                
                if verbose:
                    print(f"Outlier in {os.path.basename(img_path)}: box {box['box_id']} (class {class_id})")
                    print(f"  aspect_ratio={aspect_ratio:.4f}, mean={mean:.4f}, z-score={z_score:.4f}")
        
        issues["unusual_aspect_ratio"] = unusual_boxes
    
    if verbose:
        print(f"Found {outlier_count} aspect ratio outliers")
        
    return dataset_issues


def visualize_issues(image_path, label_path, issues, output_path, verbose=False):
    """
    Create a visualization of the image with problematic bounding boxes marked.
    
    Args:
        image_path: Path to the original image
        label_path: Path to the label file
        issues: Dictionary with issues found for this image
        output_path: Path to save the visualization
        verbose: Enable verbose output for debugging
    """
    if verbose:
        print(f"\nVisualizing issues for {os.path.basename(image_path)}")
        print(f"Output path: {output_path}")
    
    # Open the image
    try:
        image = Image.open(image_path)
        if verbose:
            print(f"Image opened successfully. Size: {image.size}")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return
    
    width, height = image.size
    draw = ImageDraw.Draw(image)
    colors = get_color_palette()
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # Read all boxes from label file
    all_boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
                all_boxes.append((class_id, (x_center, y_center, box_width, box_height)))
        
        if verbose:
            print(f"Read {len(all_boxes)} boxes from label file")
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
    
    # Draw all boxes in gray
    for class_id, (x_center, y_center, box_width, box_height) in all_boxes:
        # Convert to pixel coordinates
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        w_px = int(box_width * width)
        h_px = int(box_height * height)
        
        # Calculate corners
        x1 = x_center_px - w_px // 2
        y1 = y_center_px - h_px // 2
        x2 = x_center_px + w_px // 2
        y2 = y_center_px + h_px // 2
        
        # Draw rectangle in gray
        draw.rectangle([x1, y1, x2, y2], outline=(128, 128, 128), width=1)
    
    # Function to draw a problematic box
    def draw_problem_box(box_id, problem_type, color):
        if box_id >= len(all_boxes):
            if verbose:
                print(f"Warning: box_id {box_id} out of range (max: {len(all_boxes)-1})")
            return
            
        class_id, (x_center, y_center, box_width, box_height) = all_boxes[box_id]
        
        # Convert to pixel coordinates
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        w_px = int(box_width * width)
        h_px = int(box_height * height)
        
        # Calculate corners
        x1 = x_center_px - w_px // 2
        y1 = y_center_px - h_px // 2
        x2 = x_center_px + w_px // 2
        y2 = y_center_px + h_px // 2
        
        # Draw rectangle with problem color
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"C{class_id}:{problem_type}"
        text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:4]
        draw.rectangle([x1, y1 - text_height - 5, x1 + text_width, y1 - 5], fill=color)
        draw.text((x1, y1 - text_height - 5), label, fill="white", font=font)
        
        if verbose:
            print(f"Drew problem box: class={class_id}, type={problem_type}, coords=({x1}, {y1}, {x2}, {y2})")
    
    # Draw problematic boxes with different colors
    for item in issues["small_boxes"]:
        draw_problem_box(item["box_id"], "Small", (255, 0, 0))  # Red
        
    for item in issues["large_boxes"]:
        draw_problem_box(item["box_id"], "Large", (0, 0, 255))  # Blue
        
    for item in issues["edge_boxes"]:
        draw_problem_box(item["box_id"], "Edge", (255, 165, 0))  # Orange
        
    for item in issues["unusual_aspect_ratio"]:
        draw_problem_box(item["box_id"], "AspRatio", (128, 0, 128))  # Purple
    
    # Draw overlapping boxes with lines connecting them
    for item in issues["overlapping_boxes"]:
        box1_id = item["box1_id"]
        box2_id = item["box2_id"]
        
        if box1_id < len(all_boxes) and box2_id < len(all_boxes):
            _, (x1, y1, _, _) = all_boxes[box1_id]
            _, (x2, y2, _, _) = all_boxes[box2_id]
            
            # Convert to pixel coordinates
            x1_px = int(x1 * width)
            y1_px = int(y1 * height)
            x2_px = int(x2 * width)
            y2_px = int(y2 * height)
            
            # Draw a connecting line
            draw.line([x1_px, y1_px, x2_px, y2_px], fill=(255, 255, 0), width=2)  # Yellow
            
            # Draw boxes
            draw_problem_box(box1_id, "Overlap", (255, 255, 0))  # Yellow
            draw_problem_box(box2_id, "Overlap", (255, 255, 0))  # Yellow
            
            if verbose:
                print(f"Drew overlap connection between box {box1_id} and box {box2_id}")
    
    # Add street light issues
    if "streetlight_issues" in issues and issues["streetlight_issues"]:
        for item in issues["streetlight_issues"]:
            draw_problem_box(item["box_id"], "StrLight", (138, 43, 226))  # Purple for street light issues
            if verbose:
                print(f"Drew street light issue: box {item['box_id']}")
    
    # Add issue summary at the top
    issue_count = sum(len(issues[k]) for k in ["small_boxes", "large_boxes", "edge_boxes", "overlapping_boxes", "unusual_aspect_ratio"])
    if "streetlight_issues" in issues:
        issue_count += len(issues["streetlight_issues"])
    summary = f"Issues found: {issue_count}"
    
    draw.rectangle([0, 0, width, 30], fill=(0, 0, 0))
    draw.text((10, 5), summary, fill="white", font=font)
    
    # Create output directory if needed
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if verbose:
            print(f"Created directory: {os.path.dirname(output_path)}")
    except Exception as e:
        print(f"Error creating directory {os.path.dirname(output_path)}: {e}")
        return
    
    # Save image
    try:
        image.save(output_path)
        if verbose:
            print(f"Saved visualization to {output_path}")
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")


def generate_report(dataset_issues, class_metrics, output_dir, verbose=False, streetlight_validation=False):
    """
    Generate a summary report of issues found in the dataset.
    
    Args:
        dataset_issues: Dictionary with issues for all images
        class_metrics: Dictionary with aspect ratio statistics by class
        output_dir: Directory to save the report
        verbose: Enable verbose output for debugging
        streetlight_validation: Whether street light validation was performed
    """
    if verbose:
        print(f"\nGenerating report in directory: {output_dir}")
        print(f"Dataset issues count: {len(dataset_issues)}")
    
    # Collect statistics
    stats = {
        "total_images": len(dataset_issues),
        "images_with_issues": sum(1 for _, (issues, _) in dataset_issues.items() if issues["has_issues"]),
        "issue_counts": {
            "small_boxes": sum(len(issues["small_boxes"]) for issues, _ in dataset_issues.values()),
            "large_boxes": sum(len(issues["large_boxes"]) for issues, _ in dataset_issues.values()),
            "edge_boxes": sum(len(issues["edge_boxes"]) for issues, _ in dataset_issues.values()),
            "overlapping_boxes": sum(len(issues["overlapping_boxes"]) for issues, _ in dataset_issues.values()),
            "unusual_aspect_ratio": sum(len(issues["unusual_aspect_ratio"]) for issues, _ in dataset_issues.values())
        },
        "images_by_issue_type": {
            "small_boxes": sum(1 for _, (issues, _) in dataset_issues.items() if issues["small_boxes"]),
            "large_boxes": sum(1 for _, (issues, _) in dataset_issues.items() if issues["large_boxes"]),
            "edge_boxes": sum(1 for _, (issues, _) in dataset_issues.items() if issues["edge_boxes"]),
            "overlapping_boxes": sum(1 for _, (issues, _) in dataset_issues.items() if issues["overlapping_boxes"]),
            "unusual_aspect_ratio": sum(1 for _, (issues, _) in dataset_issues.items() if issues["unusual_aspect_ratio"])
        },
        "class_counts": {class_id: info["count"] for class_id, info in class_metrics.items()},
        "class_aspect_ratios": {
            class_id: {
                "mean": info["mean_aspect_ratio"], 
                "std": info["std_aspect_ratio"],
                "min": info["min_aspect_ratio"],
                "max": info["max_aspect_ratio"]
            } for class_id, info in class_metrics.items()
        }
    }
    
    # Add street light statistics if validation was performed
    if streetlight_validation:
        stats["issue_counts"]["streetlight_issues"] = sum(
            len(issues.get("streetlight_issues", [])) for issues, _ in dataset_issues.values()
        )
        stats["images_by_issue_type"]["streetlight_issues"] = sum(
            1 for _, (issues, _) in dataset_issues.items() 
            if "streetlight_issues" in issues and issues["streetlight_issues"]
        )
    
    if verbose:
        print("Statistics collected:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Images with issues: {stats['images_with_issues']}")
        for issue_type, count in stats["issue_counts"].items():
            print(f"  {issue_type}: {count} instances in {stats['images_by_issue_type'][issue_type]} images")
    
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created or verified output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        return
    
    # Create a human-readable summary
    summary_path = os.path.join(output_dir, "issue_summary.txt")
    if verbose:
        print(f"About to save summary report to: {summary_path}")
        
    try:
        with open(summary_path, "w") as f:
            f.write("========================================\n")
            f.write("       BOUNDING BOX ISSUE REPORT        \n")
            f.write("========================================\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write(f"  Total Images: {stats['total_images']}\n")
            f.write(f"  Images with Issues: {stats['images_with_issues']} ({stats['images_with_issues']/stats['total_images']*100:.1f}%)\n\n")
            
            f.write("Issue Summary:\n")
            total_issues = sum(stats["issue_counts"].values())
            for issue_type, count in stats["issue_counts"].items():
                f.write(f"  {issue_type.replace('_', ' ').title()}: {count} instances in {stats['images_by_issue_type'][issue_type]} images\n")
            f.write(f"  Total Issues: {total_issues}\n\n")
            
            f.write("Class Statistics:\n")
            for class_id, count in sorted(stats["class_counts"].items()):
                aspect_info = stats["class_aspect_ratios"][class_id]
                f.write(f"  Class {class_id}: {count} objects, Aspect Ratio: {aspect_info['mean']:.2f}±{aspect_info['std']:.2f} (min: {aspect_info['min']:.2f}, max: {aspect_info['max']:.2f})\n")
            
            f.write("\n")
            f.write("Images with Issues:\n")
            for img_path, (issues, _) in sorted(dataset_issues.items(), key=lambda x: sum(len(x[1][0][k]) for k in ["small_boxes", "large_boxes", "edge_boxes", "overlapping_boxes", "unusual_aspect_ratio"] + (["streetlight_issues"] if "streetlight_issues" in x[1][0] else [])), reverse=True):
                if not issues["has_issues"]:
                    continue
                    
                issue_types = []
                issue_count = 0
                if issues["small_boxes"]: 
                    issue_types.append(f"small({len(issues['small_boxes'])})")
                    issue_count += len(issues["small_boxes"])
                if issues["large_boxes"]: 
                    issue_types.append(f"large({len(issues['large_boxes'])})")
                    issue_count += len(issues["large_boxes"])
                if issues["edge_boxes"]: 
                    issue_types.append(f"edge({len(issues['edge_boxes'])})")
                    issue_count += len(issues["edge_boxes"])
                if issues["overlapping_boxes"]: 
                    issue_types.append(f"overlap({len(issues['overlapping_boxes'])})")
                    issue_count += len(issues["overlapping_boxes"])
                if issues["unusual_aspect_ratio"]: 
                    issue_types.append(f"aspect({len(issues['unusual_aspect_ratio'])})")
                    issue_count += len(issues["unusual_aspect_ratio"])
                if "streetlight_issues" in issues and issues["streetlight_issues"]:
                    issue_types.append(f"streetlight({len(issues['streetlight_issues'])})")
                    issue_count += len(issues["streetlight_issues"])
                
                f.write(f"  {os.path.basename(img_path)}: {issue_count} issues - {', '.join(issue_types)}\n")
            
        if verbose:
            print(f"Successfully wrote summary report to {summary_path}")
            
    except Exception as e:
        print(f"Error writing summary report to {summary_path}: {e}")
    
    # Save detailed JSON data
    json_path = os.path.join(output_dir, "detailed_issues.json")
    if verbose:
        print(f"About to save detailed JSON data to: {json_path}")
        
    try:
        with open(json_path, "w") as f:
            # Convert numpy values to Python native types for JSON serialization
            cleaned_class_metrics = {}
            for class_id, metrics in class_metrics.items():
                cleaned_class_metrics[int(class_id)] = {
                    "count": int(metrics["count"]),
                    "mean_aspect_ratio": float(metrics["mean_aspect_ratio"]),
                    "std_aspect_ratio": float(metrics["std_aspect_ratio"]),
                    "min_aspect_ratio": float(metrics["min_aspect_ratio"]),
                    "max_aspect_ratio": float(metrics["max_aspect_ratio"])
                }
            
            json_data = {
                "stats": {
                    "total_images": stats["total_images"],
                    "images_with_issues": stats["images_with_issues"],
                    "issue_counts": stats["issue_counts"],
                    "images_by_issue_type": stats["images_by_issue_type"]
                },
                "class_metrics": cleaned_class_metrics,
                "problematic_images": [
                    {
                        "image": os.path.basename(img_path),
                        "issues": {k: v for k, v in issues.items() if k != "has_issues"}
                    }
                    for img_path, (issues, _) in dataset_issues.items()
                    if issues["has_issues"]
                ]
            }
            json.dump(json_data, f, indent=2)
            
        if verbose:
            print(f"Successfully wrote detailed JSON data to {json_path}")
            
    except Exception as e:
        print(f"Error writing detailed JSON data to {json_path}: {e}")
    
    print(f"Report generated at {summary_path}")
    print(f"Detailed JSON data saved to {json_path}")


def find_image_extension(base_path):
    """Find the correct extension for an image file."""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        if os.path.exists(base_path + ext):
            return base_path + ext
    return None


def main():
    """Main function to run the validation tool."""
    args = parse_arguments()
    verbose = args.verbose
    
    if verbose:
        print("\n=== YOLO Bounding Box Validation Tool ===")
        print(f"Dataset directory: {args.dataset_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Validation thresholds:")
        print(f"  Minimum box size: {args.min_box_size}")
        print(f"  Maximum box size: {args.max_box_size}")
        print(f"  Overlap threshold: {args.overlap_threshold}")
        print(f"  Edge threshold: {args.edge_threshold}")
        print(f"  Aspect ratio threshold: {args.aspect_ratio_threshold}")
    
    # Validate input directories
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    preview_dir = args.preview_dir
    
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    if verbose:
        print(f"\nFound images directory: {images_dir}")
        print(f"Found labels directory: {labels_dir}")
    
    # Create output directories
    try:
        os.makedirs(output_dir, exist_ok=True)
        issues_viz_dir = os.path.join(output_dir, "issue_visualizations")
        os.makedirs(issues_viz_dir, exist_ok=True)
        
        if verbose:
            print(f"\nCreated output directory: {output_dir}")
            print(f"Created visualizations directory: {issues_viz_dir}")
    except Exception as e:
        print(f"Error creating directories: {e}")
        return
    
    # Get list of label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if verbose:
        print(f"\nFound {len(label_files)} label files")
    
    if len(label_files) == 0:
        print(f"No label files found in {labels_dir}")
        return
    
    # First pass: collect all issues
    print(f"Analyzing {len(label_files)} labels for potential issues...")
    dataset_issues = {}
    
    for label_path in tqdm(label_files):
        # Get the base filename without extension
        label_basename = os.path.basename(label_path)
        filename_no_ext = os.path.splitext(label_basename)[0]
        
        # Find corresponding image file
        image_base_path = os.path.join(images_dir, filename_no_ext)
        image_path = find_image_extension(image_base_path)
        
        if not image_path:
            print(f"Warning: No image file found for {label_basename}")
            continue
        
        # Check for issues
        issues, boxes = check_label_issues(image_path, label_path, args, verbose)
        dataset_issues[image_path] = (issues, boxes)
    
    if verbose:
        print(f"\nAnalyzed {len(dataset_issues)} images")
        images_with_issues = sum(1 for _, (issues, _) in dataset_issues.items() if issues["has_issues"])
        print(f"Found {images_with_issues} images with potential issues")
    
    # Collect class statistics and identify aspect ratio outliers
    print("Analyzing class statistics and identifying outliers...")
    class_metrics = collect_class_aspect_ratios(dataset_issues, verbose)
    dataset_issues = identify_aspect_ratio_outliers(dataset_issues, class_metrics, args.aspect_ratio_threshold, verbose)
    
    # Second pass: visualize issues
    print("Generating visualizations for images with issues...")
    images_with_issues = 0
    
    for image_path, (issues, _) in tqdm(dataset_issues.items()):
        if not issues["has_issues"]:
            continue
            
        images_with_issues += 1
        
        # Get the base filename without extension
        image_basename = os.path.basename(image_path)
        filename_no_ext = os.path.splitext(image_basename)[0]
        
        # Find corresponding label file
        label_path = os.path.join(labels_dir, f"{filename_no_ext}.txt")
        
        # Create visualization
        viz_path = os.path.join(issues_viz_dir, f"{filename_no_ext}_issues.jpg")
        visualize_issues(image_path, label_path, issues, viz_path, verbose)
    
    # After all other validation is complete, run street light validation if requested
    if args.validate_streetlights:
        print("Running street light validation...")
        
        # Integrate street light validation with the main validation results
        dataset_issues = integrate_with_validation_tool(
            dataset_issues,
            output_dir,
            streetlight_class_id=args.streetlight_class_id,
            confidence_threshold=args.streetlight_confidence,
            verbose=verbose
        )
        
        # Generate summary report with street light validation
        print(f"Found issues in {images_with_issues} out of {len(dataset_issues)} images")
        generate_report(dataset_issues, class_metrics, output_dir, verbose, streetlight_validation=True)
    else:
        # Generate standard summary report
        print(f"Found issues in {images_with_issues} out of {len(dataset_issues)} images")
        generate_report(dataset_issues, class_metrics, output_dir, verbose)


if __name__ == "__main__":
    main()