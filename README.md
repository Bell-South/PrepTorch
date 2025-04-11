# PrepTorch

## YOLO Dataset Preparation Tool

A Python tool for preparing image datasets in YOLO format for object detection training.

## Features

- **Resize Images**: Convert all images to a square format (640×640 or 1280×1280).
- **Adjust Bounding Boxes**: Automatically adjust YOLO-format bounding box coordinates.
- **Expand Bounding Boxes**: Increase the size of bounding boxes by a specified percentage.
- **Train/Val Split**: Split your dataset into training and validation sets.
- **Preview Generation**: Create visual previews of images with their bounding boxes.
- **Summary Reports**: Generate detailed dataset statistics.

## Directory Structure

```
PrepTorch/
├── main.py              # Entry point with CLI interface
├── utils/
│   ├── __init__.py
│   ├── resize.py        # Image resizing utilities
│   ├── bbox.py          # Bounding box manipulation utilities
│   ├── visualization.py # Drawing bounding boxes on images
│   └── report.py        # Summary report generation
├── README.md            # Documentation and usage examples
└── requirements.txt     # Dependencies
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

### Basic Usage

```bash
python main.py --input_dir /path/to/dataset --output_dir /path/to/output --size 640 --expand_percentage 20
```

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

### Output Structure

The tool will create the following output structure:
```
output_dir/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── preview/
│   ├── image1_preview.jpg
│   ├── image2_preview.jpg
│   └── ...
└── dataset_summary.txt
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dir` | Input directory containing 'images' and 'labels' folders | (Required) |
| `--output_dir` | Output directory for processed dataset | (Required) |
| `--size` | Target size for square images (640 or 1280) | 640 |
| `--expand_percentage` | Percentage to expand bounding boxes | 20.0 |
| `--train_split` | Proportion of images to use for training | 0.7 |
| `--seed` | Random seed for reproducible train/val splits | 42 |

## Example

Process a dataset, resize images to 1280×1280, and expand bounding boxes by 15%:

```bash
python main.py \
  --input_dir ./my_dataset \
  --output_dir ./processed_dataset \
  --size 1280 \
  --expand_percentage 15.0 \
  --train_split 0.8 \
  --seed 123
```