# Validation Tool Documentation

The validation tool is designed to identify problematic bounding box annotations in YOLO-formatted datasets. It helps ensure data quality by detecting common annotation issues before training.

## Features

The tool identifies six types of potential issues:

1. **Small Boxes**: Boxes that are too small (possible annotation errors or noise)
2. **Large Boxes**: Boxes that are too large (covering most of the image)
3. **Overlapping Boxes**: Boxes with high IoU (possible duplicates)
4. **Edge Boxes**: Boxes positioned at the edge (might be cut-off objects)
5. **Unusual Aspect Ratios**: Boxes with aspect ratios that differ from class averages
6. **Street Light Issues**: Boxes labeled as street lights that don't appear to contain street lights (optional feature)

## Usage

```bash
python validation_tool/validation_tool.py --dataset_dir ./my_dataset --output_dir ./validation_results
```

With street light validation:
```bash
python validation_tool/validation_tool.py --dataset_dir ./my_dataset --output_dir ./validation_results --validate_streetlights --streetlight_class_id 5
```

### Arguments

| Argument                  | Description                                        | Default |
|---------------------------|----------------------------------------------------|---------|
| `--dataset_dir`           | Dataset directory with 'images' and 'labels' folders | (Required) |
| `--output_dir`            | Directory to save reports and visualizations       | (Required) |
| `--preview_dir`           | Directory containing preview images (optional)     | None    |
| `--min_box_size`          | Minimum relative box size (area) to consider valid | 0.01    |
| `--max_box_size`          | Maximum relative box size (area) to consider valid | 0.9     |
| `--overlap_threshold`     | IoU threshold for flagging overlapping boxes       | 0.7     |
| `--edge_threshold`        | Distance from edge to flag edge boxes              | 0.01    |
| `--aspect_ratio_threshold`| Threshold for flagging unusual aspect ratios       | 2.0     |
| `--verbose`               | Enable verbose output for debugging                | False   |
| `--validate_streetlights` | Enable validation of street light annotations      | False   |
| `--streetlight_class_id`  | Class ID for street lights (if omitted, will auto-detect) | None |
| `--streetlight_confidence`| Confidence threshold for street light detection    | 0.5     |

## Choosing Appropriate Thresholds

The default thresholds work well for many datasets, but you may need to adjust them:

- **Small box threshold**: Decrease for datasets with intentionally small objects (e.g., distant people)
- **Large box threshold**: Increase for datasets with objects that fill most of the frame
- **Overlap threshold**: Decrease to find more overlapping boxes, increase to reduce false positives
- **Edge threshold**: Increase to find more objects near edges, decrease to focus on objects extending outside the frame
- **Aspect ratio threshold**: Adjust based on the variability of object shapes in your dataset
- **Street light confidence**: Increase for higher precision (fewer false positives), decrease for higher recall (fewer false negatives)

## Output

### Visualizations

For each image with issues, the tool generates a visualization showing:

- **Red**: Small boxes (too small, possible annotation errors)
- **Blue**: Large boxes (too large, covering most of the image)
- **Orange**: Edge boxes (positioned at the edge of the image)
- **Purple**: Unusual aspect ratios (compared to class average)
- **Yellow**: Overlapping boxes (connected with a line)
- **Purple** (different shade): Street light issues (when street light validation is enabled)

If street light validation is enabled, additional visualizations will be saved in the `streetlight_validation` subdirectory.

### Reports

The tool generates two report files:

1. **issue_summary.txt**: Human-readable report containing:
   - Dataset statistics (total images, images with issues)
   - Issue counts by type (including street light issues when enabled)
   - Class statistics and aspect ratio information
   - List of problematic images sorted by issue count

2. **detailed_issues.json**: Detailed data in JSON format for further analysis or programmatic processing

## Example Workflow

1. **Run the validation tool on your dataset**:
   ```bash
   python validation_tool/validation_tool.py --dataset_dir ./my_dataset --output_dir ./validation_results
   ```

2. **Review the issue summary**:
   ```bash
   cat ./validation_results/issue_summary.txt
   ```

3. **Examine visualizations of problematic images**:
   The visualizations are saved in `./validation_results/issue_visualizations/`

4. **Fix identified issues** in your dataset (manually or with other tools)

5. **Re-run validation** to confirm fixes:
   ```bash
   python validation_tool/validation_tool.py --dataset_dir ./fixed_dataset --output_dir ./validation_results_after
   ```

## Street Light Validation Feature

The street light validation feature uses computer vision techniques to verify whether boxes labeled as street lights actually contain street lights. This can help improve dataset quality for street light detection models.

### How It Works

The validation analyzes each bounding box using several criteria:

1. **Aspect ratio**: Street lights are typically taller than wide
2. **Brightness**: Street lights often have bright spots (the light source)
3. **Shape analysis**: Street lights typically have a pole (vertical line)
4. **Position**: Street lights are often in the upper part of the image

These factors are combined to generate a confidence score. Boxes labeled as street lights with low confidence scores are flagged as potential issues.

### Using Street Light Validation

To enable street light validation, use the `--validate_streetlights` flag and specify the class ID for street lights in your dataset:

```bash
python validation_tool/validation_tool.py --dataset_dir ./my_dataset --output_dir ./validation_results --validate_streetlights --streetlight_class_id 5
```

If you don't specify a class ID, the tool will attempt to detect street lights automatically based on visual characteristics.

## Troubleshooting

If no issues are found but you suspect problems:
- Try lowering the thresholds (especially `--min_box_size`)
- Enable verbose mode with `--verbose` to see more details about the process
- Check that your dataset follows the YOLO format correctly

If the tool reports too many issues:
- Adjust thresholds to better match your specific dataset characteristics
- Focus on the most critical issues first (like zero-size boxes or extreme overlaps)

For street light validation issues:
- If too many valid street lights are flagged as issues, try lowering the `--streetlight_confidence` threshold
- If too many invalid street lights are missed, try increasing the threshold

## Understanding YOLO Format Issues

### Common Annotation Problems

- **Zero-width or zero-height boxes**: Sometimes occur due to annotation software bugs
- **Duplicate annotations**: Multiple people annotating the same object or accidental double-clicks
- **Inverted coordinates**: When x_min > x_max or y_min > y_max, resulting in negative width/height
- **Out-of-bounds coordinates**: Values outside the 0-1 range
- **Class ID mismatches**: Using class IDs that don't match your classes.txt file
- **Misclassified objects**: Objects labeled with incorrect class IDs (e.g., non-street lights labeled as street lights)

### YOLO Format Quick Reference

YOLO format uses normalized coordinates:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer starting from 0
- `x_center`, `y_center`: Center coordinates (0-1) relative to image width/height
- `width`, `height`: Box dimensions (0-1) relative to image width/height