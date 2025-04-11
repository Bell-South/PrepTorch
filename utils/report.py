"""
Report generation utilities for YOLO dataset preparation.
"""


def generate_report(output_path, train_count, val_count, class_counts, total_objects):
    """
    Generate a summary report of the processed dataset.
    
    Args:
        output_path: Path to save the report
        train_count: Number of training images
        val_count: Number of validation images
        class_counts: Dictionary mapping class IDs to counts
        total_objects: Total number of objects across all images
    """
    with open(output_path, 'w') as f:
        f.write("========================================\n")
        f.write("          DATASET SUMMARY REPORT        \n")
        f.write("========================================\n\n")
        
        # Dataset statistics
        f.write("Dataset Statistics:\n")
        f.write(f"  Total Images: {train_count + val_count}\n")
        f.write(f"  Training Images: {train_count} ({train_count/(train_count+val_count)*100:.1f}%)\n")
        f.write(f"  Validation Images: {val_count} ({val_count/(train_count+val_count)*100:.1f}%)\n\n")
        
        # Object statistics
        f.write("Object Statistics:\n")
        f.write(f"  Total Objects: {total_objects}\n")
        f.write(f"  Unique Classes: {len(class_counts)}\n")
        f.write(f"  Objects per Image: {total_objects/(train_count+val_count):.2f}\n\n")
        
        # Class distribution
        f.write("Class Distribution:\n")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[0])
        for class_id, count in sorted_classes:
            percentage = (count / total_objects) * 100
            f.write(f"  Class {class_id}: {count} objects ({percentage:.1f}%)\n")
        
        f.write("\n========================================\n")