# Module 9: Automated Coin Counter

## Overview

The Automated Coin Counter is a comprehensive computer vision solution for detecting, isolating, and counting coins in images. It leverages Otsu's thresholding, morphological operations, and contour analysis to reliably identify coins and calculate their spatial coordinates.

## Pipeline Architecture

```
1. Load Image
        ↓
2. Convert to Grayscale
        ↓
3. Apply Gaussian Blur (noise reduction)
        ↓
4. Otsu's Thresholding (cv2.THRESH_BINARY_INV)
        ↓
5. Morphological Closing (fill interior holes)
        ↓
6. Contour Detection (cv2.findContours)
        ↓
7. Filter by Area Threshold (remove small noise)
        ↓
8. Calculate Centroids (cv2.moments)
        ↓
9. Visualize & Report
```

## Key Features

✅ **Automatic Coin Detection** - Detects coins of varying sizes
✅ **Spatial Coordinates** - Provides exact (x, y) centroid positions
✅ **Area Analysis** - Calculates coin area and radius statistics
✅ **Synthetic Generation** - Creates test coin images automatically
✅ **Custom Images** - Works with any photograph of coins
✅ **Comprehensive Reporting** - Detailed analysis with statistics
✅ **Visual Dashboard** - 4-panel pipeline visualization
✅ **Configurable Parameters** - Adjust thresholds and kernel sizes

## How to Use

### Quick Start - Generate & Analyze Synthetic Coins

```bash
python automated_coin_counter.py
```

**Output:**
- Detected coins with coordinates printed to console
- 4-panel analysis dashboard
- Detailed report in text file
- All intermediate images saved

### Using Custom Coin Images

#### **Simple Usage (Default Settings)**
```bash
python automated_coin_counter.py --custom coins.jpg
```

#### **With Optimized Threshold**
```bash
python automated_coin_counter.py --custom coins.jpg --min-area 400
```

#### **Complete Example with All Options**
```bash
python automated_coin_counter.py \
    --custom /path/to/coin_photo.jpg \
    --min-area 350 \
    --kernel-size 7 \
    --output-dir ./coin_results
```

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--custom` | path | None | Path to custom coin image (JPG, PNG, BMP, TIFF) |
| `--min-area` | int | 500 | Minimum coin area threshold in pixels (filter noise) |
| `--kernel-size` | int | 9 | Size of structuring element for morphology (must be odd) |
| `--output-dir` | path | ./output_coins | Directory to save all output files |

## Parameter Guide

### Minimum Area Threshold
Filters out noise and small artifacts. Adjust based on coin and noise size:

| Scenario | Recommended | Details |
|----------|------------|---------|
| Large coins (>50px radius) | 400-600 | Standard coins |
| Small coins (<30px radius) | 200-400 | Coins far from camera |
| High noise image | 600-1000 | Noisy backgrounds |
| Clean image | 300-500 | Good lighting/contrast |

```bash
# Detect smaller coins (more sensitive)
--min-area 300

# Detect only larger coins (less noise)
--min-area 800
```

### Kernel Size
Controls morphological operation strength. Must be odd number:

| Kernel | Behavior | Use Case |
|--------|----------|----------|
| 5x5 | Gentle smoothing | Small noise only |
| 7x7 | Moderate smoothing | Standard images |
| 9x9 | Strong smoothing | Heavy shadows/reflections |
| 11x11 | Very strong smoothing | Extreme conditions |

```bash
# Gentle morphology (preserve details)
--kernel-size 5

# Standard morphology (balanced)
--kernel-size 9

# Strong morphology (heavy processing)
--kernel-size 11
```

## Output Files

### Generated in Output Directory:

```
output_coins/
├── coin_analysis_dashboard.png        # 4-panel pipeline visualization
├── coin_analysis_report.txt           # Detailed statistics and coordinates
├── coin_detection.png                 # Final image with detected coins
├── original_image.png                 # Input image
├── binary_threshold.png               # Otsu threshold result
├── morphological_closing.png          # After morphology
└── generated_coins.png                # (If synthetic image was generated)
```

### File Descriptions:

**coin_analysis_dashboard.png**
- Panel 1: Original image
- Panel 2: Binary threshold result
- Panel 3: After morphological closing
- Panel 4: Final detection with circles and centroids

**coin_analysis_report.txt**
```
- Total coin count
- Area statistics (mean, min, max)
- Radius statistics
- Complete list of coordinates
- Processing time
```

**coin_detection.png**
- Green circles around detected coins
- Red center points at centroids
- Text labels with (x, y) coordinates
- Coin count displayed on image

## Practical Examples

### Example 1: Real Coin Photography

**Setup:** Photo of assorted coins on white background

```bash
python automated_coin_counter.py \
    --custom coins_photo.jpg \
    --min-area 400 \
    --kernel-size 7 \
    --output-dir ./real_coins_analysis
```

**Expected Results:**
- Detects all coins accurately
- Shadows don't interfere (due to morphological closing)
- Metallic shine handled by Otsu's threshold
- Accurate centroid calculation

### Example 2: Document Coins (Overhead Shot)

**Setup:** Coins photographed from directly above on dark surface

```bash
python automated_coin_counter.py \
    --custom overhead_coins.png \
    --min-area 300 \
    --kernel-size 9 \
    --output-dir ./overhead_analysis
```

**Expected Results:**
- Clear circular shapes detected
- Strong kernel handles shadows
- Accurate radius calculation
- Good for counting/cataloging

### Example 3: Coin Collection Quality Check

**Setup:** Batch processing multiple coin images

```bash
for img in coins/*.jpg; do
    echo "Processing $img..."
    python automated_coin_counter.py \
        --custom "$img" \
        --min-area 350 \
        --output-dir "./results/$(basename $img .jpg)"
done
```

**Results:**
- Individual report for each image
- Consistent analysis parameters
- Easy comparison across images

### Example 4: Calibration with Known Coins

**Setup:** Reference coins of known size for scale

```bash
# First: Analyze reference coin of known diameter
python automated_coin_counter.py \
    --custom reference_quarter.jpg \
    --output-dir ./calibration

# Extract the radius and note pixel dimension
# Then analyze test coins with known scale
python automated_coin_counter.py \
    --custom test_coins.jpg \
    --output-dir ./scaled_analysis
```

## Programmatic Usage (Python API)

### Basic Analysis

```python
from automated_coin_counter import AutomatedCoinCounter

# Create counter
counter = AutomatedCoinCounter(output_dir='./results')

# Run analysis (generates synthetic image by default)
coins, result = counter.run_analysis()

print(f"Detected {len(coins)} coins")
for coin in coins:
    cx, cy = coin['centroid']
    radius = coin['radius']
    area = coin['area']
    print(f"Position: ({cx}, {cy}), Radius: {radius}px, Area: {area}px²")
```

### Custom Image Analysis

```python
from automated_coin_counter import AutomatedCoinCounter

counter = AutomatedCoinCounter(output_dir='./results')

# Analyze custom image
coins, result = counter.run_analysis(
    image_path='my_coins.jpg',
    min_area=400,
    kernel_size=7
)

# Process detected coins
for i, coin in enumerate(coins, 1):
    print(f"Coin {i}: {coin['centroid']}, Area: {coin['area']}px²")
```

### Advanced: Manual Pipeline Control

```python
from automated_coin_counter import AutomatedCoinCounter
import cv2

counter = AutomatedCoinCounter(output_dir='./results')

# Load and preprocess
img = cv2.imread('coins.jpg')
gray, blurred = counter.preprocess_image(img)

# Apply threshold
binary = counter.threshold_image(blurred)

# Apply morphology
closed = counter.apply_morphology(binary, kernel_size=7)

# Detect coins
coins = counter.find_coins(closed, min_area=400)

# Visualize
result = counter.visualize_coins(img, coins)
cv2.imshow('Detected Coins', result)
cv2.waitKey(0)
```

## Understanding the Output Report

### Sample Report Output:

```
======================================================================
AUTOMATED COIN COUNTER - DETAILED ANALYSIS REPORT
======================================================================
Image Resolution: 800x600 pixels
Processing Time: 0.15 seconds
Total Coins Detected: 12

COIN STATISTICS:
  Average Area: 5234.50 px²
  Std Dev Area: 1245.30 px²
  Min Area: 3200.00 px²
  Max Area: 8100.00 px²
  
  Average Radius: 40.80 px
  Std Dev Radius: 12.20 px
  Min Radius: 32.00 px
  Max Radius: 51.00 px

DETECTED COIN LOCATIONS:
  Coin  1: Position=(150, 120) | Area=5100 px² | Radius=40 px
  Coin  2: Position=(250, 180) | Area=4900 px² | Radius=39 px
  ...

COORDINATE BOUNDS:
  X Range: 85 - 745 px
  Y Range: 65 - 565 px
```

### Interpreting Statistics:

- **Average Area**: Shows typical coin size in image
- **Std Dev Area**: Indicates size variation (0 = same size, high = varied)
- **Average Radius**: Average coin radius in pixels
- **Coordinate Bounds**: Spatial distribution of coins

## Troubleshooting

### Issue: No coins detected

**Possible Causes:**
1. Min area threshold too high
2. Image quality too poor
3. Coins not contrasting with background

**Solutions:**
```bash
# Lower the minimum area threshold
--min-area 200

# Or try higher kernel size to enhance morphology
--kernel-size 11
```

### Issue: Detecting too much noise

**Possible Causes:**
1. Min area threshold too low
2. High image noise
3. Complex background

**Solutions:**
```bash
# Increase minimum area threshold
--min-area 800

# Use stronger morphological smoothing
--kernel-size 9
```

### Issue: Incomplete coin detection (cutting off edges)

**Possible Causes:**
1. Coins at image borders
2. Partially visible coins
3. Blur affecting edges

**Solutions:**
- Use `--min-area 300` to detect partial coins
- Ensure coins are fully visible in image
- Try `--kernel-size 5` for finer detail preservation

### Issue: Inaccurate centroid positions

**Possible Causes:**
1. Binary threshold not optimal
2. Coins overlapping (detected as one)
3. Metallic reflections affecting edges

**Solutions:**
- Check `binary_threshold.png` output
- Adjust lighting when photographing coins
- Use `--kernel-size 7` to refine boundaries

## Performance Notes

| Metric | Typical | Range |
|--------|---------|-------|
| **Processing Time** | 0.1-0.5s | Depends on image size |
| **Coins Detected/sec** | 50-200 | Based on image resolution |
| **Memory Usage** | <100MB | Standard for 1000x1000px |

### Optimization Tips:

1. **Resize large images** - 800x600 is ideal
2. **Use appropriate kernel size** - 7-9 for most cases
3. **Batch processing** - Process similar images with same parameters
4. **Multi-processing** - Process multiple images in parallel

## Technical Details

### Otsu's Thresholding

- Automatically determines optimal binary threshold
- Minimizes within-class variance
- Works well with bimodal histograms (coin vs. background)
- Handles varying lighting conditions

### Morphological Closing

- Combines dilation then erosion
- Fills interior holes and shadows
- Preserves overall coin shape
- Removes interior metallic engraving details

### Centroid Calculation

- Uses image moments: M10/M00 (x), M01/M00 (y)
- Sub-pixel precision (rounded to integer)
- Accurate for well-separated coins
- May be inaccurate for overlapping coins

### Contour Filtering

- `cv2.RETR_EXTERNAL` finds external boundaries only
- Ignores internal holes (coins with engravings)
- `cv2.contourArea()` used for area filtering
- Efficient O(n) filtering where n = number of contours

## Tips for Best Results

1. **Lighting**: Use even, diffuse lighting to minimize shadows
2. **Background**: Use contrasting background (white for dark coins, dark for shiny)
3. **Focus**: Ensure image is sharp and in focus
4. **Distance**: Coins should occupy 10-30% of image area
5. **Angle**: Photograph coins from directly above when possible
6. **Cleanliness**: Remove dust and debris before photographing

## Common Parameters by Scenario

### Professional Coin Counting
```bash
--min-area 350 --kernel-size 7
```

### Casual Photography
```bash
--min-area 400 --kernel-size 9
```

### High-Precision Analysis
```bash
--min-area 300 --kernel-size 5
```

### Noisy/Low-Quality Images
```bash
--min-area 600 --kernel-size 11
```

## Next Steps

1. Run with `--help` to see all options
2. Test with generated synthetic coins first
3. Try with real coin images
4. Adjust parameters based on your specific use case
5. Use API for integration into larger applications

## Help & Support

```bash
# Display help with all options
python automated_coin_counter.py --help

# Run tests to verify installation
python test_coin_counter.py
```
