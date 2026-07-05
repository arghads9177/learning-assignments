# Module 8: Using Custom Images with Morphological Operations

## Overview

The updated Module 8 now supports analyzing morphological operations on **custom images** instead of only synthetic masks. You can supply any image (document, photo, diagram, text, etc.) and the tool will perform comprehensive morphological analysis on it.

## How to Use Custom Images

### Basic Command-Line Usage

#### 1. **With Generated Synthetic Mask (Default)**
```bash
python morphological_visual_comparison.py
```
- Generates a synthetic corrupted binary mask
- Outputs: 3 dashboards (rect, ellipse, cross kernels)
- Output directory: `./output_morphology`

#### 2. **With Custom Image (Simple)**
```bash
python morphological_visual_comparison.py --custom /path/to/image.jpg
```
- Loads your custom image
- Automatically converts to binary using default threshold (127)
- Creates 3 comprehensive dashboards
- **Output**: `/output_morphology/morphological_dashboard_*.png`

#### 3. **With Custom Image & Custom Threshold**
```bash
python morphological_visual_comparison.py --custom document.png --threshold 150
```
- Loads custom image
- Binarizes with threshold value 150
- **Threshold range**: 0-255
  - Lower (0-100): Lighter areas become white
  - Medium (120-150): Balanced binarization
  - Higher (150-255): Darker areas become black

#### 4. **With Custom Kernel Size**
```bash
python morphological_visual_comparison.py --custom image.jpg --kernel-size 7
```
- Uses 7×7 structuring elements instead of default 5×5
- Larger kernels = stronger morphological effects
- Useful kernel sizes: 3, 5, 7, 9, 11

#### 5. **With Custom Output Directory**
```bash
python morphological_visual_comparison.py --custom image.jpg --output-dir ./results
```
- Saves all visualizations to `./results` directory
- Useful for organizing multiple analyses

#### 6. **All Options Combined**
```bash
python morphological_visual_comparison.py \
    --custom /path/to/document.jpg \
    --threshold 140 \
    --kernel-size 7 \
    --output-dir ./morphology_results
```

## Image Requirements

### Supported Formats
- **JPEG**: `.jpg`, `.jpeg`
- **PNG**: `.png`
- **BMP**: `.bmp`
- **TIFF**: `.tiff`, `.tif`
- **Any format OpenCV supports**

### Image Characteristics
| Aspect | Recommendation | Details |
|--------|---|---|
| Size | 300-600 pixels | Larger files take longer but show more detail |
| Color | Color or Grayscale | Both are converted to binary |
| Content | Text, shapes, documents | Works with any binary-able image |
| Quality | Clear and contrasted | Best results with clean images |

## Threshold Selection Guide

### What is Threshold?
Binarization threshold converts grayscale images to pure black (0) and white (255).
- Pixels darker than threshold → BLACK
- Pixels lighter than threshold → WHITE

### Choosing the Right Threshold

#### **For Document Scans / Text (Use: 140-160)**
```python
# Dark text on light paper
--threshold 150
```
- Good for: Printed documents, typed text
- Result: Clean black text on white background

#### **For Light Images (Use: 100-120)**
```python
# Faint or light-colored content
--threshold 110
```
- Good for: Faded documents, pencil writing
- Result: Preserves lighter content

#### **For Dark Images (Use: 160-200)**
```python
# Mostly dark image with small details
--threshold 180
```
- Good for: Photographs, complex diagrams
- Result: Preserves detailed structures

#### **For Standard/Balanced (Use: 120-140)**
```python
# Default, works for most images
--threshold 127  # Default
```
- Good for: General purpose analysis
- Result: Balanced black/white distribution

## Practical Examples

### Example 1: Analyze a Document Scan
```bash
python morphological_visual_comparison.py \
    --custom ~/Documents/scanned_invoice.pdf \
    --threshold 145
```
**Expected Output:**
- Shows how opening removes noise from document
- Closing fills small holes in characters
- Gradient shows text boundaries

### Example 2: Analyze a Handwritten Note
```bash
python morphological_visual_comparison.py \
    --custom handwritten_note.jpg \
    --threshold 130 \
    --kernel-size 5
```
**Expected Output:**
- Opening preserves handwriting while removing speckles
- Erosion thins thick strokes
- Dilation connects broken strokes

### Example 3: Analyze a Diagram
```bash
python morphological_visual_comparison.py \
    --custom circuit_diagram.png \
    --threshold 160 \
    --kernel-size 7
```
**Expected Output:**
- Opening preserves line structure
- Gradient extracts line boundaries
- Top Hat shows fine line details

### Example 4: Quality Control - Surface Inspection
```bash
python morphological_visual_comparison.py \
    --custom product_surface.jpg \
    --threshold 128 \
    --output-dir ./inspection_results
```
**Expected Output:**
- Black Hat highlights surface defects
- Gradient shows defect boundaries
- Useful for detecting scratches, dust, etc.

## Programmatic Usage (Python Code)

### Basic Usage with Default Settings
```python
from morphological_visual_comparison import MorphologicalVisualComparison

# Create comparison tool with custom image
comparison = MorphologicalVisualComparison(
    output_dir='./my_results',
    custom_image_path='path/to/image.jpg'
)

# Run analysis with default threshold (127)
comparison.run_complete_analysis(
    kernel_size=5,
    use_custom=True,
    threshold=127
)
```

### Advanced Usage with Custom Parameters
```python
from morphological_visual_comparison import MorphologicalVisualComparison

# Create with custom image
comparison = MorphologicalVisualComparison(
    output_dir='./analysis_output',
    custom_image_path='document.png'
)

# Run with custom parameters
comparison.run_complete_analysis(
    kernel_size=7,        # Larger kernel for stronger effects
    use_custom=True,
    threshold=150         # Optimized for document scanning
)

# Results saved to:
# - ./analysis_output/clean_mask.png
# - ./analysis_output/morphological_dashboard_rect_7x7.png
# - ./analysis_output/morphological_dashboard_ellipse_7x7.png
# - ./analysis_output/morphological_dashboard_cross_7x7.png
```

### Load Custom Image Directly
```python
from morphological_visual_comparison import MorphologicalVisualComparison

comparison = MorphologicalVisualComparison(output_dir='./results')

# Load and preprocess custom image
clean_mask, corrupted_mask = comparison.load_custom_image(
    image_path='my_image.jpg',
    threshold_value=140
)

# Now use the binary image for analysis
kernels = comparison.create_structuring_elements(kernel_size=5)
results = comparison.apply_morphological_operations(corrupted_mask, kernels['rect'])
comparison.create_8panel_dashboard(corrupted_mask, results, 'rect', 5)
```

## Output Files Explanation

When analyzing a custom image, the following files are created:

### Binary Masks
```
output_morphology/
├── clean_mask.png           # Binarized version of your image
└── corrupted_mask.png       # Same (no synthetic corruption for custom images)
```

### 8-Panel Dashboards (One per kernel type)
```
output_morphology/
├── morphological_dashboard_rect_5x5.png      # Rectangle kernel analysis
├── morphological_dashboard_ellipse_5x5.png   # Ellipse kernel analysis
└── morphological_dashboard_cross_5x5.png     # Cross kernel analysis
```

### Each Dashboard Contains 8 Panels:
1. **Original Binary Image** - Your binarized input
2. **Erosion** - Shrunk white regions
3. **Dilation** - Expanded white regions
4. **Opening** - Denoised version
5. **Closing** - Filled holes version
6. **Morphological Gradient** - Edge extraction
7. **Top Hat** - Noise/detail extraction
8. **Black Hat** - Defect/spot detection

## Analyzing the Results

### What to Look For

#### **Text Documents**
- **Opening** should clean up noise while preserving letters
- **Closing** should fill small holes in letters
- **Gradient** shows text boundaries clearly

#### **Photographs**
- **Erosion** removes small white noise
- **Dilation** connects fragmented areas
- **Black Hat** shows defects/blemishes

#### **Handwriting**
- **Opening** preserves pen strokes while removing specks
- **Top Hat** isolates fine details
- **Gradient** shows stroke boundaries

#### **Quality Control**
- **Black Hat** highlights surface defects
- **Morphological Gradient** shows defect edges
- **Top Hat** reveals fine imperfections

## Troubleshooting

### Issue: Image looks all white or all black
**Solution**: Adjust threshold value
```bash
# Try middle range first
--threshold 127

# If still wrong, try:
--threshold 100   # For very dark images
--threshold 180   # For very light images
```

### Issue: Text/shapes are too thin after processing
**Solution**: Reduce kernel size or use dilation
```bash
# Smaller kernel = less aggressive
--kernel-size 3

# Or use dilation to thicken
```

### Issue: Too much noise in result
**Solution**: Use opening operation or increase threshold
```bash
# Opening removes small white noise
# Increase threshold to make more areas white
--threshold 140
```

### Issue: Output directory doesn't exist
**Solution**: Specify output directory or create it first
```bash
python morphological_visual_comparison.py \
    --custom image.jpg \
    --output-dir ./my_results
```

## Performance Notes

- **Image size impact**: Larger images (>1000px) take longer to process
- **Kernel size impact**: Larger kernels (>7) increase computation time
- **Number of kernels**: Always processes 3 kernel types (rect, ellipse, cross)
- **Typical processing time**: 5-30 seconds depending on image size and kernel size

## Tips for Best Results

1. **Start with default threshold (127)** and adjust if needed
2. **Use kernel size 5** for most purposes
3. **For documents**: Increase threshold to 150-160
4. **For photos**: Use threshold around 100-120
5. **For detailed analysis**: Use kernel size 7
6. **Save results to different directories** for easy comparison

## File Structure
```
Chapter1/module8/
├── morphological_visual_comparison.py    # Main script (updated)
├── CUSTOM_IMAGE_USAGE.md                # This file
├── test_document.jpg                    # Example custom image
└── output_morphology/                   # Results directory
    ├── clean_mask.png
    ├── morphological_dashboard_rect_5x5.png
    ├── morphological_dashboard_ellipse_5x5.png
    └── morphological_dashboard_cross_5x5.png
```

## Getting Help

```bash
# Display help message with all options
python morphological_visual_comparison.py --help
```

This will show all available command-line arguments and usage examples.
