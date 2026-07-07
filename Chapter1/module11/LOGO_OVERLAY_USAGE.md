# Module 11: Logo Overlay Utility

## Overview

The Logo Overlay Utility is a sophisticated image processing solution for seamlessly blending irregular, non-rectangular logos onto background images. It uses binary masking, inverse operations, and bitwise image addition to create natural-looking overlays without any rectangular framing artifacts.

## The Logo Overlay Pipeline

```
1. Load/Generate Images
        ↓
2. Resize Logo (if >30% of background)
        ↓
3. Isolate ROI (background region matching logo size)
        ↓
4. Create Binary Mask (high-contrast logo isolation)
        ↓
5. Create Inverse Mask (invert binary mask)
        ↓
6. Clear Logo from ROI (black out logo area in background)
        ↓
7. Isolate Logo Colors (extract logo, background becomes black)
        ↓
8. Blend Images (sum cleared ROI and isolated logo)
        ↓
9. Inject Composite (place blended patch back in background)
        ↓
10. Visualize & Report
```

## Key Concepts

### Binary Masking
- Converts logo to grayscale
- Uses Otsu's thresholding for high-contrast separation
- Result: 255 where logo exists, 0 elsewhere
- Provides precise logo geometry for all subsequent operations

### Inverse Masking
- Bitwise NOT of binary mask
- Result: 0 where logo exists, 255 elsewhere
- Used to clear logo area from background safely

### Bitwise Operations
- **cv2.bitwise_and(img, img, mask=mask)**: Extract pixels where mask=255
- **cv2.bitwise_not(mask)**: Invert binary mask
- **cv2.add(img1, img2)**: Saturated addition for blending

### No Rectangular Artifacts
- Direct array slicing would show rectangular bounds
- Masking extracts only the logo shape
- Addition preserves background where logo doesn't exist
- Result: seamless irregular logo overlay

## Key Features

✅ **Seamless Logo Blending** - No rectangular framing artifacts  
✅ **Binary Masking** - Precise logo geometry isolation  
✅ **Automatic Resizing** - Scales logos >30% of background  
✅ **ROI Extraction** - Isolates background regions  
✅ **Custom Positioning** - Place logos anywhere on background  
✅ **Synthetic Generation** - Creates test images automatically  
✅ **Custom Images** - Works with any image file  
✅ **4×2 Pipeline Visualization** - Shows all processing steps  
✅ **Comprehensive Reporting** - Detailed analysis output  
✅ **Command-Line Interface** - Easy parameter control  

## How to Use

### Quick Start - Generate & Analyze Synthetic Images

```bash
python logo_overlay_utility.py
```

**Output:**
- Synthetic background and logo images
- All 9 intermediate processing images
- 4×2 visualization dashboard
- Detailed analysis report
- Console statistics

### Using Custom Images

#### **Simple Usage (Default Settings)**
```bash
python logo_overlay_utility.py --background background.jpg --logo logo.png
```

#### **With Custom Position**
```bash
python logo_overlay_utility.py \
    --background background.jpg \
    --logo logo.png \
    --position 200 150
```

#### **Complete Example with All Options**
```bash
python logo_overlay_utility.py \
    --background /path/to/background.jpg \
    --logo /path/to/logo.png \
    --position 300 250 \
    --output-dir ./overlay_results
```

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--background` | path | None | Path to background image (JPG, PNG, BMP, TIFF) |
| `--logo` | path | None | Path to logo image (JPG, PNG, BMP, TIFF) |
| `--position` | x y | 100 100 | Logo placement coordinates (top-left corner) |
| `--output-dir` | path | ./output_logo_overlay | Directory for all output files |

## Output Files

### Generated in Output Directory:

```
output_logo_overlay/
├── background_original.png        # Input background image
├── logo_original.png              # Input logo image
├── logo_resized.png               # Resized logo (if needed)
├── roi_extracted.png              # Background ROI patch
├── binary_mask.png                # Binary logo mask
├── inverse_mask.png               # Inverse binary mask
├── cleared_roi.png                # ROI with logo area cleared
├── isolated_logo.png              # Logo with isolated colors
├── blended_composite.png          # Blended composite patch
├── final_overlay.png              # Final result (logo on background)
├── logo_overlay_dashboard.png     # 4×2 pipeline visualization
└── logo_overlay_report.txt        # Detailed analysis report
```

### File Descriptions:

**binary_mask.png**
- White (255) where logo pixels exist
- Black (0) where logo background exists
- High contrast for precise masking

**inverse_mask.png**
- Black (0) where logo pixels exist
- White (255) where logo background exists
- Used to clear logo area from background

**cleared_roi.png**
- Background ROI with logo area blacked out (0,0,0)
- Ready for logo blending
- Shows exact region where logo will be placed

**isolated_logo.png**
- Original logo colors preserved
- Background turned black (0,0,0)
- Ready to be added to cleared ROI

**blended_composite.png**
- Result of cv2.add(cleared_roi, isolated_logo)
- Combines background and logo perfectly
- Exact dimensions of logo

**final_overlay.png**
- Complete image with logo overlaid at specified position
- Shows final result ready for use

## Dashboard Layout

The 4×2 visualization dashboard shows:

```
Row 1: Background Image | Logo Image
Row 2: Binary Mask (Logo=255) | Inverse Mask (Logo=0)
Row 3: Cleared ROI (Logo Area Blacked Out) | Isolated Logo (Colors Only)
Row 4: Blended Composite | Final Overlay Result
```

Each visualization helps understand each pipeline step.

## Programmatic Usage (Python API)

### Basic Logo Overlay

```python
from logo_overlay_utility import LogoOverlayUtility

# Create overlay utility
overlay = LogoOverlayUtility(output_dir='./results')

# Run with synthetic images
result = overlay.run_analysis()

# Display result
cv2.imshow('Final Overlay', result)
cv2.waitKey(0)
```

### Custom Images with Specific Position

```python
from logo_overlay_utility import LogoOverlayUtility
import cv2

overlay = LogoOverlayUtility(output_dir='./results')

# Overlay with custom position
result = overlay.run_analysis(
    background_path='my_background.jpg',
    logo_path='my_logo.png',
    position=(200, 150)  # x=200, y=150
)

cv2.imwrite('final_result.png', result)
```

### Manual Pipeline Control

```python
from logo_overlay_utility import LogoOverlayUtility
import cv2

overlay = LogoOverlayUtility()

# Load images
background = cv2.imread('background.jpg')
logo = cv2.imread('logo.png')

# Resize if needed
logo = overlay.resize_logo_if_needed(logo, background)

# Isolate ROI
roi, position = overlay.isolate_roi(background, logo, position=(100, 100))

# Create masks
binary_mask = overlay.create_binary_mask(logo)
inverse_mask = overlay.create_inverse_mask(binary_mask)

# Clear logo from ROI
cleared = overlay.clear_logo_from_roi(roi, inverse_mask)

# Isolate logo colors
isolated = overlay.isolate_logo_colors(logo, binary_mask)

# Blend
composite = overlay.blend_images(cleared, isolated)

# Inject into background
final = overlay.inject_composite(background, composite, position)

cv2.imshow('Result', final)
cv2.waitKey(0)
```

## Practical Examples

### Example 1: Logo on Website Banner

**Scenario:** Overlay company logo on promotional banner

```python
overlay = LogoOverlayUtility('./banner_results')

# Load banner and logo
background = cv2.imread('promotional_banner.jpg')
logo = cv2.imread('company_logo.png')

# Position logo in top-right corner
result = overlay.run_analysis(
    background_path='promotional_banner.jpg',
    logo_path='company_logo.png',
    position=(650, 50)
)

# Save result
cv2.imwrite('banner_with_logo.jpg', result)
```

### Example 2: Watermark Application

**Scenario:** Add watermark to photograph

```python
overlay = LogoOverlayUtility('./watermark_results')

result = overlay.run_analysis(
    background_path='photo.jpg',
    logo_path='watermark.png',
    position=(700, 550)  # Bottom-right corner
)
```

### Example 3: Batch Logo Overlay

**Scenario:** Apply logo to multiple images

```python
import os
from logo_overlay_utility import LogoOverlayUtility

logo_path = 'company_logo.png'
images = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']

for img_path in images:
    overlay = LogoOverlayUtility(f'./results/{os.path.basename(img_path)}')
    result = overlay.run_analysis(
        background_path=img_path,
        logo_path=logo_path,
        position=(100, 100)
    )
    output = f'./results/{os.path.basename(img_path)}'
    cv2.imwrite(f'{output}_with_logo.jpg', result)
```

### Example 4: Dynamic Position Calculation

**Scenario:** Position logo in corner based on background size

```python
import cv2
from logo_overlay_utility import LogoOverlayUtility

overlay = LogoOverlayUtility('./smart_position_results')

# Load images
background = cv2.imread('background.jpg')
logo = cv2.imread('logo.png')

# Calculate bottom-right position
bg_h, bg_w = background.shape[:2]
logo_h, logo_w = logo.shape[:2]
margin = 20

position = (bg_w - logo_w - margin, bg_h - logo_h - margin)

# Overlay
result = overlay.run_analysis(
    background_path='background.jpg',
    logo_path='logo.png',
    position=position
)
```

## Understanding the Pipeline Steps

### Step 1: Image Loading & Resizing

```python
# Load images
background = cv2.imread('background.jpg')
logo = cv2.imread('logo.png')

# Resize if >30% of background
logo = overlay.resize_logo_if_needed(logo, background)
```

**Why resize?** Large logos can dominate the image. 30% threshold ensures logo is prominent but not overwhelming.

### Step 2: ROI Isolation

```python
roi, position = overlay.isolate_roi(background, logo, position=(100, 100))
```

**What it does:** Extracts the background region where logo will be placed
**Why needed:** Processing only the affected region is more efficient

### Step 3: Binary Masking

```python
binary_mask = overlay.create_binary_mask(logo)
# Result: 255 where logo is, 0 where background is
```

**Why Otsu's threshold?**
- Automatically finds optimal threshold
- Handles varying lighting and contrast
- Creates sharp, clean mask

### Step 4: Inverse Mask

```python
inverse_mask = overlay.create_inverse_mask(binary_mask)
# Result: 0 where logo is, 255 where background is
```

**Why invert?** To black out logo area in background without affecting non-logo regions

### Step 5: Clear Logo Region

```python
cleared = overlay.clear_logo_from_roi(roi, inverse_mask)
# Uses: cv2.bitwise_and(roi, roi, mask=inverse_mask)
```

**Effect:** Replaces logo area with black (0,0,0)
**Result:** Background ready for logo blending

### Step 6: Isolate Logo Colors

```python
isolated = overlay.isolate_logo_colors(logo, binary_mask)
# Uses: cv2.bitwise_and(logo, logo, mask=binary_mask)
```

**Effect:** Keeps logo colors, makes background black
**Result:** Logo ready to be added

### Step 7: Blending

```python
composite = overlay.blend_images(cleared, isolated)
# Uses: cv2.add(cleared, isolated)
```

**Why cv2.add?**
- Saturated addition handles overflow
- Adding to black background (0,0,0) is non-destructive
- Result is valid uint8 values [0, 255]

**Why no rectangular artifacts?**
- Not using simple array assignment (would show rectangle)
- Using masking ensures only logo shape is blended
- Addition only affects non-black pixels

### Step 8: Inject Composite

```python
final = overlay.inject_composite(background, composite, position)
```

**Effect:** Places blended composite back into original background
**Result:** Final image with seamlessly integrated logo

## Understanding Binary Masks

### Creating a Binary Mask

```python
# Logo image has white background and colored shape
logo = cv2.imread('logo.png')

# Convert to grayscale
gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

# Apply Otsu's threshold
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Result:
# - 255 (white) where logo pixels are
# - 0 (black) where background is
```

### Using Masks

```python
# Extract regions where mask=255
extracted = cv2.bitwise_and(image, image, mask=mask)

# Extract regions where mask=0
inverted = cv2.bitwise_not(mask)
extracted_inverse = cv2.bitwise_and(image, image, mask=inverted)
```

## Troubleshooting

### Issue: Logo appears with rectangular box

**Possible Causes:**
1. Direct array assignment used instead of masking
2. Background not properly cleared before addition
3. Mask has low contrast

**Solutions:**
```python
# ✗ WRONG - shows rectangular box
background[y:y+h, x:x+w] = logo

# ✓ CORRECT - uses masking and addition
masked_logo = cv2.bitwise_and(logo, logo, mask=binary_mask)
cleared_roi = cv2.bitwise_and(roi, roi, mask=inverse_mask)
composite = cv2.add(cleared_roi, masked_logo)
background[y:y+h, x:x+w] = composite
```

### Issue: Logo blends with wrong background color

**Possible Causes:**
1. Isolated logo background not fully black
2. Binary mask not high contrast enough
3. Background ROI not properly cleared

**Solutions:**
```python
# Ensure isolated logo is truly black
isolated = cv2.bitwise_and(logo, logo, mask=binary_mask)
assert isolated.min() == 0, "Background should be pure black"

# Verify binary mask is high contrast
unique_values = set(binary_mask.flatten())
assert unique_values == {0, 255}, f"Expected binary, got {unique_values}"
```

### Issue: Logo extends outside image bounds

**Possible Causes:**
1. Position calculations not clamped
2. Logo larger than remaining background space

**Solutions:**
```python
# Position is automatically clamped in isolate_roi()
# For manual positioning:
bg_h, bg_w = background.shape[:2]
logo_h, logo_w = logo.shape[:2]
x = max(0, min(position[0], bg_w - logo_w))
y = max(0, min(position[1], bg_h - logo_h))
```

### Issue: Resized logo quality degraded

**Possible Causes:**
1. Downsampling without proper interpolation
2. Resizing too small

**Solutions:**
```python
# Use INTER_AREA for downsampling quality
resized = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Use INTER_CUBIC for upsampling quality (if needed)
resized = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
```

## Performance Characteristics

| Metric | Typical | Notes |
|--------|---------|-------|
| **Processing Time** | 0.2-0.5s | Depends on image size |
| **Memory Usage** | <100MB | Manageable for standard images |
| **Mask Generation** | <5ms | Very fast operation |
| **Addition Operation** | <10ms | Efficient bitwise operation |

### Optimization Tips:

1. **Use appropriate image sizes** - 800×600 ideal for standard use
2. **Pre-resize large images** - Reduces processing time significantly
3. **Batch processing** - Process multiple logos on same background efficiently
4. **Reuse masks** - If overlaying same logo multiple times, create mask once

## Advanced Topics

### Multiple Logos

Overlay multiple logos on same background:

```python
overlay = LogoOverlayUtility()

background = cv2.imread('background.jpg')
result = background.copy()

logos = [
    ('logo1.png', (100, 100)),
    ('logo2.png', (500, 100)),
    ('logo3.png', (300, 400))
]

for logo_path, position in logos:
    final = overlay.run_analysis(
        background_path=cv2.imwrite('temp_bg.jpg', result),
        logo_path=logo_path,
        position=position
    )
    result = final

cv2.imwrite('multi_logo_overlay.jpg', result)
```

### Logo with Transparency

Handle PNG logos with alpha channel:

```python
# Load PNG with alpha
logo_bgra = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)

# If has alpha channel, use it as mask
if logo_bgra.shape[2] == 4:
    logo_bgr = logo_bgra[:,:,:3]
    alpha = logo_bgra[:,:,3]
    # Use alpha as binary mask
    binary_mask = (alpha > 127).astype(np.uint8) * 255
else:
    logo_bgr = logo_bgra
    # Generate mask normally
    binary_mask = overlay.create_binary_mask(logo_bgr)
```

### Gradient Blending

Create smooth edges using gradient mask:

```python
# Instead of binary mask, create gradient mask
center_x, center_y = logo_w // 2, logo_h // 2
Y, X = np.meshgrid(np.arange(logo_h), np.arange(logo_w), indexing='ij')
dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
gradient_mask = np.clip(255 * (1 - dist / max(logo_w, logo_h)), 0, 255).astype(np.uint8)

# Use for blending
isolated = cv2.bitwise_and(logo, logo, mask=gradient_mask)
```

## Next Steps

1. Run with `--help` to see all options
2. Test with generated synthetic images first
3. Try with real logo and background images
4. Experiment with different positions
5. Integrate into larger applications
6. Try advanced techniques (multiple logos, gradient blending)

## Help & Support

```bash
# Display help with all options
python logo_overlay_utility.py --help

# Run tests to verify installation
python test_logo_overlay.py
```

## References

- **OpenCV Documentation:** https://docs.opencv.org/
- **Image Masking:** Binary masks and bitwise operations
- **Image Arithmetic:** Addition, AND, OR, NOT operations
- **Thresholding Techniques:** Otsu's method for automatic threshold selection
