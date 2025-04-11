# YOLO Bounding Box Issues - Cheat Sheet

## Common Issues and How to Fix Them

### 1. Small Boxes (Red)

**Symptoms:**
- Tiny boxes that are hard to see
- May represent annotation errors or very small objects

**Potential causes:**
- Accidental clicks during annotation
- Trying to annotate objects that are too small to be useful for training
- Decimal point errors in label files

**Solutions:**
- Delete the box if it's an error
- If it's a valid small object, consider if it's useful for your model
- Check for decimal point errors in label coordinates

### 2. Large Boxes (Blue)

**Symptoms:**
- Boxes covering most or all of the image
- Training may focus too much on background

**Potential causes:**
- Annotating the entire scene instead of specific objects
- Mistakes in coordinate conversion
- Accidentally swapping width/height values

**Solutions:**
- Resize the box to tightly fit the object
- Split into multiple more specific boxes if needed
- Check for coordinate conversion errors

### 3. Edge Boxes (Orange)

**Symptoms:**
- Boxes that touch or extend beyond image edges
- May represent partially visible objects

**Potential causes:**
- Objects partially outside the frame
- Incorrect coordinate normalization

**Solutions:**
- Decide if partially visible objects are useful for your model
- Consider cropping boxes to stay within image
- Fix normalization if coordinates are incorrect

### 4. Overlapping Boxes (Yellow)

**Symptoms:**
- Multiple boxes with high overlap (connected by yellow line)
- May represent duplicate annotations

**Potential causes:**
- Multiple people annotating the same object
- Accidentally adding the same box twice
- Intentional overlap for hierarchical objects (car vs. license plate)

**Solutions:**
- Remove duplicate boxes
- Decide on annotation policy for overlapping objects
- Adjust overlap threshold if needed for specific cases

### 5. Unusual Aspect Ratios (Purple)

**Symptoms:**
- Boxes with aspect ratios very different from class average
- May look unnaturally stretched or compressed

**Potential causes:**
- Incorrect width/height values
- Annotating partial objects
- Genuine outliers in object appearance

**Solutions:**
- Check if the aspect ratio makes sense for the object
- Fix width/height if they were swapped
- Consider if this is a valid special case

## YOLO Format Quick Reference

YOLO format uses normalized coordinates:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer starting from 0
- `x_center`, `y_center`: Center coordinates (0-1) relative to image width/height
- `width`, `height`: Box dimensions (0-1) relative to image width/height

Example:
```
0 0.5 0.5 0.2 0.3
```
Means: Class 0, centered at image center, width is 20% of image width, height is 30% of image height.

## Common Conversion Formulas

### Pixel to Normalized Coordinates

```python
# From pixel coordinates to normalized YOLO format
def pixel_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / (2 * img_width)
    y_center = (y_min + y_max) / (2 * img_height)
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height
```

### Normalized to Pixel Coordinates

```python
# From normalized YOLO format to pixel coordinates
def yolo_to_pixel(x_center, y_center, width, height, img_width, img_height):
    x_min = int((x_center - width/2) * img_width)
    y_min = int((y_center - height/2) * img_height)
    x_max = int((x_center + width/2) * img_width)
    y_max = int((y_center + height/2) * img_height)
    return x_min, y_min, x_max, y_max
```

## Quick Checks for Label Files

1. **Check value ranges**: All x, y, width, height values should be between 0 and 1
2. **Check class IDs**: Should match your classes.txt file and start from 0
3. **Check file names**: Label file names should match image file names
4. **Check file format**: Each line should have exactly 5 values
5. **Check for duplicates**: No duplicate boxes for the same object