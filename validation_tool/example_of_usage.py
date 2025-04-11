#!/usr/bin/env python3
"""
Example script showing how to use the test_preview.py tool.

This script demonstrates how to:
1. Run the validation tool on your dataset
2. Process the JSON results programmatically
3. List the top 10 most problematic images for manual review
"""

import os
import json
import subprocess
import argparse
from collections import Counter


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Example usage of the bounding box validation tool."
    )
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        required=True, 
        help="Dataset directory with 'images' and 'labels' folders"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./validation_results",
        help="Output directory for validation results"
    )
    
    return parser.parse_args()


def run_validation(dataset_dir, output_dir):
    """Run the validation tool on the dataset."""
    cmd = [
        "python", "test_preview.py",
        "--dataset_dir", dataset_dir,
        "--output_dir", output_dir
    ]
    
    print(f"Running validation tool: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running validation tool:")
        print(result.stderr)
        return False
    
    print("Validation complete!")
    print(result.stdout)
    return True


def analyze_results(output_dir):
    """Analyze validation results from JSON file."""
    json_path = os.path.join(output_dir, "detailed_issues.json")
    
    if not os.path.exists(json_path):
        print(f"Error: Results file not found at {json_path}")
        return
    
    # Load results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Print overall statistics
    stats = results["stats"]
    print("\n===== VALIDATION SUMMARY =====")
    print(f"Total images: {stats['total_images']}")
    print(f"Images with issues: {stats['images_with_issues']} ({stats['images_with_issues']/stats['total_images']*100:.1f}%)")
    
    # Count issues by type
    print("\nIssue counts:")
    for issue_type, count in stats["issue_counts"].items():
        print(f"  {issue_type.replace('_', ' ').title()}: {count} instances in {stats['images_by_issue_type'][issue_type]} images")
    
    # Class statistics
    print("\nClass statistics:")
    for class_id, metrics in results["class_metrics"].items():
        print(f"  Class {class_id}: {metrics['count']} objects, Aspect ratio: {metrics['mean_aspect_ratio']:.2f}±{metrics['std_aspect_ratio']:.2f}")
    
    # Find top problematic images
    print("\nTop 10 most problematic images:")
    
    # Count total issues per image
    image_issues = []
    for img_data in results["problematic_images"]:
        image_name = img_data["image"]
        issue_count = sum(len(img_data["issues"][k]) for k in img_data["issues"])
        issue_types = Counter()
        
        # Count issue types
        for issue_type, issues in img_data["issues"].items():
            if len(issues) > 0:
                issue_types[issue_type] = len(issues)
        
        image_issues.append((image_name, issue_count, issue_types))
    
    # Sort by number of issues (descending)
    image_issues.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10
    for i, (image_name, issue_count, issue_types) in enumerate(image_issues[:10]):
        issue_summary = ", ".join([f"{k}:{v}" for k, v in issue_types.items()])
        print(f"  {i+1}. {image_name}: {issue_count} issues ({issue_summary})")
        print(f"     Preview: {os.path.join(output_dir, 'issue_visualizations', os.path.splitext(image_name)[0] + '_issues.jpg')}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Run validation tool
    success = run_validation(args.dataset_dir, args.output_dir)
    
    if success:
        # Analyze results
        analyze_results(args.output_dir)


if __name__ == "__main__":
    main()