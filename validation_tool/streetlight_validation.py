"""
Street Light Validation Module

This module extends the validation tool to check if bounding boxes correctly enclose street lights.
It implements both simple heuristic approaches and more advanced computer vision techniques.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import cv2


def analyze_streetlight_content(image_path, bbox, class_id):
    """
    Analyze if a bounding box likely contains a street light.
    
    Args:
        image_path: Path to the image file
        bbox: Bounding box in format (x_center, y_center, width, height) - normalized
        class_id: Class ID of the object
        
    Returns:
        dict: Results of street light analysis with confidence score and details
    """
    # Load the image
    try:
        # Use PIL for initial loading
        pil_img = Image.open(image_path)
        img_width, img_height = pil_img.size
        
        # Convert to OpenCV format for analysis
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {
            "valid": False,
            "confidence": 0.0,
            "error": f"Error loading image: {str(e)}",
            "details": {}
        }
    
    # Convert normalized coordinates to pixel values
    x_center, y_center, width, height = bbox
    x_center_px = int(x_center * img_width)
    y_center_px = int(y_center * img_height)
    width_px = int(width * img_width)
    height_px = int(height * img_height)
    
    # Calculate corner coordinates
    x1 = max(0, x_center_px - width_px // 2)
    y1 = max(0, y_center_px - height_px // 2)
    x2 = min(img_width - 1, x_center_px + width_px // 2)
    y2 = min(img_height - 1, y_center_px + height_px // 2)
    
    # Extract the region of interest (ROI)
    roi = cv_img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return {
            "valid": False,
            "confidence": 0.0,
            "error": "Empty bounding box",
            "details": {}
        }
    
    # Perform analysis using multiple techniques
    results = {}
    
    # 1. Aspect ratio check - street lights are typically taller than wide
    aspect_ratio = height / width if width > 0 else float('inf')
    is_vertical = aspect_ratio > 1.5  # Street lights are usually taller than wide
    results["aspect_check"] = {
        "passed": is_vertical,
        "aspect_ratio": aspect_ratio,
        "score": min(1.0, aspect_ratio / 3.0) if aspect_ratio > 1.0 else 0.0
    }
    
    # 2. Color analysis - look for bright spots (street light bulbs)
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Check for bright regions (potential light sources)
    _, bright_mask = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
    bright_pixel_ratio = np.sum(bright_mask) / (bright_mask.shape[0] * bright_mask.shape[1] * 255)
    
    has_bright_spots = bright_pixel_ratio > 0.05  # At least 5% of pixels should be bright
    results["brightness_check"] = {
        "passed": has_bright_spots,
        "bright_pixel_ratio": bright_pixel_ratio,
        "score": min(1.0, bright_pixel_ratio * 10)  # Scale the score
    }
    
    # 3. Shape analysis - street lights often have a pole
    # Use edge detection to find vertical lines
    edges = cv2.Canny(gray_roi, 50, 150)
    
    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=height_px/3, maxLineGap=10)
    
    # Check if vertical lines are detected
    has_vertical_lines = False
    vertical_line_score = 0.0
    
    if lines is not None:
        vertical_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Check if the line is vertical (close to 90 degrees)
            if angle > 75 and angle < 105:
                vertical_count += 1
        
        has_vertical_lines = vertical_count > 0
        vertical_line_score = min(1.0, vertical_count / 5.0)  # Normalize score
    
    results["vertical_line_check"] = {
        "passed": has_vertical_lines,
        "score": vertical_line_score,
        "line_count": vertical_count if lines is not None else 0
    }
    
    # 4. Position check - street lights are often in the upper part of the image
    position_score = 0.0
    is_upper_half = y_center < 0.5
    position_score = 1.0 if is_upper_half else max(0, 1.0 - (y_center - 0.5) * 2)
    
    results["position_check"] = {
        "passed": is_upper_half,
        "y_center": y_center,
        "score": position_score
    }
    
    # Combine all scores to get final confidence
    weights = {
        "aspect_check": 0.3,
        "brightness_check": 0.3,
        "vertical_line_check": 0.3,
        "position_check": 0.1
    }
    
    confidence = sum(weights[key] * results[key]["score"] for key in weights)
    
    # Determine if this is likely a street light
    is_valid = confidence >= 0.5  # Confidence threshold
    
    return {
        "valid": is_valid,
        "confidence": confidence,
        "details": results
    }


def check_streetlight_boxes(image_path, boxes, threshold=0.5, expected_class_id=None):
    """
    Check all bounding boxes in an image to determine if they contain street lights.
    
    Args:
        image_path: Path to the image file
        boxes: List of (class_id, bbox) tuples
        threshold: Confidence threshold for street light detection
        expected_class_id: If provided, only check boxes with this class ID
        
    Returns:
        list: Details about each box's street light validation
    """
    results = []
    
    for box_id, (class_id, bbox) in enumerate(boxes):
        # Skip if we're only checking a specific class and this isn't it
        if expected_class_id is not None and class_id != expected_class_id:
            continue
        
        # Analyze box content
        analysis = analyze_streetlight_content(image_path, bbox, class_id)
        
        # Add box metadata
        result = {
            "box_id": box_id,
            "class_id": class_id,
            "bbox": bbox,
            "is_streetlight": analysis["valid"],
            "confidence": analysis["confidence"],
            "details": analysis["details"] if "details" in analysis else {}
        }
        
        results.append(result)
    
    return results


def visualize_streetlight_validation(image_path, boxes, validation_results, output_path):
    """
    Create a visualization showing street light validation results.
    
    Args:
        image_path: Path to the original image
        boxes: List of (class_id, bbox) tuples
        validation_results: Results from check_streetlight_boxes
        output_path: Path to save the visualization
    """
    try:
        # Open the image
        img = Image.open(image_path)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()
        
        # Create a mapping from box_id to validation result
        result_map = {r["box_id"]: r for r in validation_results}
        
        # Draw each box
        for box_id, (class_id, bbox) in enumerate(boxes):
            x_center, y_center, box_width, box_height = bbox
            
            # Convert to pixel coordinates
            x_center_px = int(x_center * width)
            y_center_px = int(y_center * height)
            w_px = int(box_width * width)
            h_px = int(box_height * height)
            
            # Calculate corner coordinates
            x1 = x_center_px - w_px // 2
            y1 = y_center_px - h_px // 2
            x2 = x_center_px + w_px // 2
            y2 = y_center_px + h_px // 2
            
            # Check if we have validation results for this box
            if box_id in result_map:
                result = result_map[box_id]
                
                # Choose color based on validation result
                if result["is_streetlight"]:
                    # Green for valid street light
                    color = (0, 255, 0)
                    label = f"SL {result['confidence']:.2f}"
                else:
                    # Red for invalid street light
                    color = (255, 0, 0)
                    label = f"Not SL {result['confidence']:.2f}"
            else:
                # Gray for boxes we didn't validate
                color = (128, 128, 128)
                label = f"Class {class_id}"
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:4]
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width, y1 - 5], fill=color)
            draw.text((x1, y1 - text_height - 5), label, fill="white", font=font)
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error visualizing street light validation: {e}")
        return False


# Integration with main validation tool
def integrate_with_validation_tool(dataset_issues, output_dir, streetlight_class_id=None, confidence_threshold=0.5, verbose=False):
    """
    Integrate street light validation with the main validation tool.
    
    Args:
        dataset_issues: Dictionary with issues for all images from the main validation tool
        output_dir: Directory to save reports and visualizations
        streetlight_class_id: Class ID representing street lights (if None, will try to detect them)
        confidence_threshold: Confidence threshold for street light detection
        verbose: Enable verbose output for debugging
        
    Returns:
        dict: Updated dataset_issues with street light validation results
    """
    if verbose:
        print("\nRunning street light validation...")
    
    # Create output directory for street light validations
    streetlight_viz_dir = os.path.join(output_dir, "streetlight_validation")
    os.makedirs(streetlight_viz_dir, exist_ok=True)
    
    # Count of images and street light validations
    total_images = 0
    images_with_streetlights = 0
    correct_streetlights = 0
    incorrect_streetlights = 0
    
    # Process each image
    for img_path, (issues, boxes_data) in dataset_issues.items():
        total_images += 1
        
        # Get original boxes for this image
        image_basename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(image_basename)[0]
        
        # Reconstruct box format for street light validation
        formatted_boxes = []
        for box in boxes_data:
            formatted_boxes.append((box["class_id"], box["coords"]))
        
        if len(formatted_boxes) == 0:
            continue
        
        # Run street light validation
        validation_results = check_streetlight_boxes(
            img_path, 
            formatted_boxes, 
            threshold=confidence_threshold,
            expected_class_id=streetlight_class_id
        )
        
        if len(validation_results) == 0:
            continue
        
        # Add street light issues to the issues dict
        streetlight_issues = []
        has_streetlight_issues = False
        
        for result in validation_results:
            # If this is expected to be a street light but isn't
            if streetlight_class_id is not None and result["class_id"] == streetlight_class_id and not result["is_streetlight"]:
                streetlight_issues.append({
                    "box_id": result["box_id"],
                    "class_id": result["class_id"],
                    "confidence": result["confidence"],
                    "coords": result["bbox"]
                })
                has_streetlight_issues = True
                incorrect_streetlights += 1
            
            # If this is a street light according to our detection
            if result["is_streetlight"]:
                images_with_streetlights += 1
                if streetlight_class_id is None or result["class_id"] == streetlight_class_id:
                    correct_streetlights += 1
        
        if has_streetlight_issues:
            issues["streetlight_issues"] = streetlight_issues
            issues["has_issues"] = True
        else:
            issues["streetlight_issues"] = []
        
        # Create visualization for this image
        viz_path = os.path.join(streetlight_viz_dir, f"{filename_no_ext}_streetlight.jpg")
        visualize_streetlight_validation(img_path, formatted_boxes, validation_results, viz_path)
        
        if verbose and (has_streetlight_issues or len(validation_results) > 0):
            print(f"Processed {image_basename}: {len(streetlight_issues)} street light issues")
    
    if verbose:
        print(f"Street light validation complete:")
        print(f"  Total images: {total_images}")
        print(f"  Images with street lights: {images_with_streetlights}")
        print(f"  Correct street light annotations: {correct_streetlights}")
        print(f"  Incorrect street light annotations: {incorrect_streetlights}")
    
    return dataset_issues