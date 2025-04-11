# YOLO Dataset Preparation Tool

A Python tool for preparing image datasets in YOLO format for object detection training.

## Features

- **Resize Images**: Convert all images to a square format (640×640 or 1280×1280) while preserving aspect ratio
- **Adjust Bounding Boxes**: Automatically adjust YOLO-format bounding box coordinates
- **Expand Bounding Boxes**: Increase the size of bounding boxes by a specified percentage
- **Train/Val Split**: Split your dataset into training and validation sets
- **Preview Generation**: Create visual previews of images with their bounding boxes
- **Summary Reports**: Generate detailed dataset statistics
- **Debug Utilities**: Generate previews of original dataset and side-by-side comparisons

## Directory Structure

```
yolo_dataset_prep/
├── main.py                  # Main processing script
├── preview_original.py      # Utility to preview original dataset
├── compare_bbox.py        # Utility to compare original vs processed images
├── utils/
│   ├── __init__.py
│   ├── resize.py            # Image resizing utilities
│   ├── bbox.py              # Bounding box manipulation utilities
│   ├── visualization.py     # Drawing bounding boxes on images
│   └── report.py            # Summary report generation
├── README.md                # Documentation and usage examples
└── requirements.txt         # Dependencies
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Bell-South/PrepTorch.git
   cd PrepTorch
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Required Directory Structure

Your input directory should have the following structure:
```
input_dir/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

Each label file should follow the YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
With all values normalized between 0 and 1.

### Main Processing Tool

Process a dataset, resize images, and expand bounding boxes:

```bash
python main.py \
  --input_dir ./my_dataset \
  --output_dir ./processed_dataset \
  --size 640 \
  --expand_percentage 20.0 \
  --train_split 0.7 \
  --seed 42
```

#### Command Line Arguments for main.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dir` | Input directory containing 'images' and 'labels' folders | (Required) |
| `--output_dir` | Output directory for processed dataset | (Required) |
| `--size` | Target size for square images (640 or 1280) | 640 |
| `--expand_percentage` | Percentage to expand bounding boxes | 20.0 |
| `--train_split` | Proportion of images to use for training | 0.7 |
| `--seed` | Random seed for reproducible train/val splits | 42 |

### Debug and Visualization Tools

This project includes additional utilities to help visualize and debug your dataset.

#### Preview Original Dataset

Generate preview images for your original dataset with bounding boxes:

```bash
python preview_original.py \
  --input_dir ./my_dataset \
  --output_dir ./original_previews
```

This tool is useful for verifying that your original labels are correctly aligned with objects before processing.

#### Compare Original vs Processed Images

Create side-by-side comparisons between original and processed images:

```bash
# Compare a specific image:
python compare_bbox.py \
  --original_dir ./my_dataset \
  --processed_dir ./processed_dataset \
  --output_dir ./comparisons \
  --image_name image1

# Compare multiple random samples:
python compare_bbox.py \
  --original_dir ./my_dataset \
  --processed_dir ./processed_dataset \
  --output_dir ./comparisons \
  --num_samples 10
```

This helps you visually verify how the resizing and bounding box adjustments have affected your dataset.

## How It Works

### Image Resizing

The tool uses a **resize-then-pad** approach:

1. **Calculate Scaling**: Determine the scale that fits the image within the target dimensions while preserving aspect ratio
2. **Resize Image**: Scale the image according to the calculated factor
3. **Add Padding**: Create a square black canvas and center the resized image on it

This approach ensures no information is lost (unlike cropping) and maintains the original aspect ratio of objects.

### Bounding Box Processing

For each bounding box:

1. **Coordinate Adjustment**: Convert the normalized YOLO coordinates to account for the new image size and padding
2. **Box Expansion**: Expand each bounding box by the specified percentage while maintaining its center point
3. **Boundary Checking**: Ensure expanded boxes stay within image boundaries

### Train/Val Split

The tool randomly assigns images to training and validation sets according to the specified split ratio, which provides a proper dataset division for model training.

## Troubleshooting Common Issues

- **Misaligned Bounding Boxes**: Use the comparison tool to identify issues in original vs processed images
- **Missing Labels**: Check if some images don't have corresponding label files
- **Edge Cases**: Objects near image edges might require special attention
- **Expansion Problems**: Reduce the expansion percentage if objects overlap after processing

## License

[Apache License](LICENSE)