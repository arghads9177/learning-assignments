# Module 10: ROI Extraction Exercises

## Overview

This module demonstrates three fundamental ROI (Region of Interest) extraction techniques used extensively in computer vision applications. Each technique is optimized for different use cases and provides different levels of precision.

## Three ROI Extraction Methods

### 1. Face ROI (Bounding Box Slicing)

**Technique:** Coordinate-based array slicing

Extract rectangular regions using direct numpy array slicing with bounding box coordinates.

```python
face = img[y:y+h, x:x+w]
```

**Characteristics:**
- **Speed:** Extremely fast - O(w×h)
- **Accuracy:** Lower - includes background in corners
- **Flexibility:** Low - only rectangular regions
- **Edge Type:** Sharp boundaries

**Code Example:**
```python
from roi_extraction_utility import ROIExtractionUtility

extractor = ROIExtractionUtility()
img = cv2.imread('photo.jpg')

# Extract face ROI at coordinates (150, 100) with size 200×250
face_roi = extractor.extract_face_roi(img, x=150, y=100, w=200, h=250)
```

**Use Cases:**
- Face detection and recognition
- Object bounding box tracking
- License plate detection
- Document scanning (rectangular documents)

**Advantages:**
- ✓ Simple and intuitive
- ✓ No mask computation overhead
- ✓ Fastest method
- ✓ Preserves original colors

**Limitations:**
- ✗ Only rectangular regions
- ✗ Includes background/corners
- ✗ Not suitable for circular objects
- ✗ Not flexible for irregular shapes

### 2. Circular ROI (Vignette Isolation)

**Technique:** Circle mask with bitwise AND operation

Extract circular regions using a white circle mask on black canvas combined with bitwise AND.

```python
mask = cv2.circle(black_canvas, center, radius, 255, -1)
roi = cv2.bitwise_and(img, img, mask=mask)
```

**Characteristics:**
- **Speed:** Fast - O(w×h)
- **Accuracy:** High - precise circular boundaries
- **Flexibility:** Medium - supports radius variation and vignette fades
- **Edge Type:** Smooth (can be sharp or gradient-based)

**Code Example:**
```python
from roi_extraction_utility import ROIExtractionUtility

extractor = ROIExtractionUtility()
img = cv2.imread('photo.jpg')

# Extract circular ROI at center (300, 200) with radius 100
circular_roi, mask = extractor.extract_circular_roi(
    img, center_x=300, center_y=200, radius=100, fade=False
)

# With vignette fade effect
circular_roi_fade, mask_fade = extractor.extract_circular_roi(
    img, center_x=300, center_y=200, radius=100, fade=True
)
```

**How It Works:**

```
Step 1: Create black canvas
    mask = zeros(img.shape[:2])

Step 2: Draw white circle
    cv2.circle(mask, center, radius, 255, -1)

Step 3: Bitwise AND with original
    roi = cv2.bitwise_and(img, img, mask=mask)
```

**Use Cases:**
- Iris/pupil detection and extraction
- Circular object detection (buttons, coins, wheels)
- Attention mechanism visualization
- Spotlight and vignette effects
- Focus region highlighting

**Advantages:**
- ✓ Circular precision
- ✓ Support for soft vignette edges
- ✓ Fast mask-based operation
- ✓ Efficient for spotlight effects
- ✓ Easy gradient mask generation

**Variations:**
- **Sharp Circle:** Binary mask (0 or 255)
- **Vignette:** Gradient mask with smooth fade
- **Ellipse:** Use `cv2.ellipse()` for eye-like regions
- **Multi-level:** Multiple concentric circles

### 3. Object ROI (Polygon Segment)

**Technique:** Polygon mask with bitwise AND operation

Extract irregular polygon-shaped regions using `cv2.fillPoly()` and bitwise AND.

```python
vertices = [[x1,y1], [x2,y2], [x3,y3], ...]
mask = zeros(img.shape[:2])
cv2.fillPoly(mask, [vertices], 255)
roi = cv2.bitwise_and(img, img, mask=mask)
```

**Characteristics:**
- **Speed:** Fast - O(w×h)
- **Accuracy:** Highest - arbitrary precision
- **Flexibility:** Highest - any polygon shape
- **Edge Type:** Precise polygon boundaries

**Code Example:**
```python
from roi_extraction_utility import ROIExtractionUtility

extractor = ROIExtractionUtility()
img = cv2.imread('photo.jpg')

# Define polygon vertices (triangle)
triangle_vertices = [[100, 50], [200, 100], [150, 200]]
polygon_roi, mask = extractor.extract_polygon_roi(img, triangle_vertices)

# Quadrilateral (4 vertices)
quad_vertices = [[50, 50], [200, 50], [200, 200], [50, 200]]
quad_roi, quad_mask = extractor.extract_polygon_roi(img, quad_vertices)

# Pentagon (5 vertices)
pentagon_vertices = [[150, 50], [250, 75], [200, 200], [100, 200], [50, 75]]
pentagon_roi, pent_mask = extractor.extract_polygon_roi(img, pentagon_vertices)
```

**How It Works:**

```
Step 1: Create black canvas
    mask = zeros(img.shape[:2])

Step 2: Fill polygon with white
    cv2.fillPoly(mask, [vertices], 255)

Step 3: Bitwise AND with original
    roi = cv2.bitwise_and(img, img, mask=mask)
```

**Use Cases:**
- Road lane detection (trapezoidal regions)
- Building/roof detection (polygonal contours)
- Semantic segmentation
- Agricultural field detection
- Custom region definition
- Vehicle detection with keypoints

**Vertex Ordering:**
- Vertices define polygon perimeter in sequence
- Clockwise or counter-clockwise both work
- Last vertex connects to first (closed polygon)
- Order affects fill direction

**Advantages:**
- ✓ Arbitrary polygon shapes
- ✓ Maximum flexibility
- ✓ Precise boundary control
- ✓ Supports N-vertex polygons
- ✓ Efficient bitwise operation

**Advanced Polygon Techniques:**
- **Convex Hull:** Find convex boundary of points
- **Contour Approximation:** Simplify complex shapes
- **Multiple Polygons:** Extract multiple regions in one mask
- **Nested Polygons:** Combine with bitwise operations

## Quick Start

### Basic Usage

Generate synthetic test image and extract all ROI types:
```bash
python roi_extraction_utility.py
```

### With Custom Image

Analyze your own image:
```bash
python roi_extraction_utility.py --custom photo.jpg
```

### Custom Output Directory

Save results to specific location:
```bash
python roi_extraction_utility.py --custom photo.jpg --output-dir ./results
```

## Output Files

```
output_roi/
├── original_image.png              # Input image
├── face_roi.png                    # Extracted face ROI
├── face_mask.png                   # Face bounding box mask
├── circular_roi.png                # Extracted circular ROI
├── circular_mask.png               # Circular extraction mask
├── polygon_roi.png                 # Extracted polygon ROI
├── polygon_mask.png                # Polygon extraction mask
├── roi_extraction_dashboard.png    # 3×3 visualization
├── roi_extraction_report.txt       # Detailed analysis
└── generated_test_image.png        # (if synthetic)
```

## Dashboard Layout

The 3×3 visualization dashboard shows:

```
Original          | Face ROI          | Face Mask
Original          | Circular ROI      | Circular Mask
Original          | Polygon ROI       | Polygon Mask
```

Each row demonstrates one extraction method with:
- Column 1: Original image for reference
- Column 2: Extracted region
- Column 3: Binary mask used

## Programmatic Usage

### Simple Face ROI

```python
import cv2
from roi_extraction_utility import ROIExtractionUtility

img = cv2.imread('photo.jpg')
extractor = ROIExtractionUtility()

# Extract rectangular region
face = extractor.extract_face_roi(img, x=100, y=50, w=150, h=200)
cv2.imshow('Face ROI', face)
cv2.waitKey(0)
```

### All Three Methods

```python
img = cv2.imread('photo.jpg')
extractor = ROIExtractionUtility('./results')

# Extract all three types
face = extractor.extract_face_roi(img, 100, 50, 150, 200)
circle, c_mask = extractor.extract_circular_roi(img, 300, 200, 100)
polygon, p_mask = extractor.extract_polygon_roi(img, [[50,50],[200,50],[200,200],[50,200]])

# Display results
cv2.imshow('Face', face)
cv2.imshow('Circle', circle)
cv2.imshow('Polygon', polygon)
```

### Complete Analysis

```python
from roi_extraction_utility import ROIExtractionUtility

extractor = ROIExtractionUtility('./analysis')
face_roi, circular_roi, polygon_roi = extractor.run_analysis('custom_image.jpg')

# All output files automatically created
# Console statistics printed
# HTML dashboard generated
```

## Practical Examples

### Example 1: Face Detection Pipeline

Extract and analyze faces from photo:

```python
face_roi = extractor.extract_face_roi(img, x, y, w, h)

# Process face region
face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
face_hist = cv2.calcHist([face_gray], [0], None, [256], [0,256])

# Apply face-specific processing
face_equalized = cv2.equalizeHist(face_gray)
```

### Example 2: Iris Detection

Extract circular iris region with vignette:

```python
iris, iris_mask = extractor.extract_circular_roi(
    img, center_x=eye_cx, center_y=eye_cy, radius=50, fade=True
)

# Analyze iris features
iris_gray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(iris_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### Example 3: Road Lane Detection

Extract lane region with polygon:

```python
lane_vertices = [
    [50, 480],    # Bottom-left
    [200, 300],   # Top-left
    [400, 300],   # Top-right
    [550, 480]    # Bottom-right
]

lane_roi, lane_mask = extractor.extract_polygon_roi(img, lane_vertices)

# Detect lane lines
lane_gray = cv2.cvtColor(lane_roi, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(lane_gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
```

### Example 4: Multiple Regions

Extract multiple ROIs for parallel processing:

```python
# Extract multiple faces
faces = []
for (x, y, w, h) in detected_faces:
    face = extractor.extract_face_roi(img, x, y, w, h)
    faces.append(face)

# Process all faces in parallel
for i, face in enumerate(faces):
    cv2.imwrite(f'face_{i}.jpg', face)
```

## Performance Characteristics

| Method | Speed | Preprocessing | Use Case |
|--------|-------|---------------|----------|
| Face (Box) | ⚡⚡⚡ Fastest | None | Rectangles |
| Circular | ⚡⚡ Fast | Mask generation | Circles |
| Polygon | ⚡⚡ Fast | Vertices → Mask | Any shape |

**Speed Notes:**
- All methods are O(w×h) image size
- Face method: Direct slicing, no computation
- Circular/Polygon: Mask generation + bitwise AND
- Mask generation is fast (~1-5ms per image)

## Mask Generation Tips

### Creating Better Masks

**Binary Masks (Sharp edges):**
```python
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(mask, center, radius, 255, -1)  # 255 = white, 0 = black
```

**Gradient Masks (Soft vignette):**
```python
X, Y = np.meshgrid(np.arange(w), np.arange(h))
dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
mask = np.clip(255 * (1 - dist/radius), 0, 255).astype(np.uint8)
```

**Multi-level Masks (Concentric regions):**
```python
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(mask, center, radius1, 100, -1)  # Inner region
cv2.circle(mask, center, radius2, 200, -1)  # Middle region
cv2.circle(mask, center, radius3, 255, -1)  # Outer region
```

## Troubleshooting

### Issue: Polygon not filling correctly

**Solution:** Check vertex order is consistent (all clockwise or all counter-clockwise)

```python
# Correct: Clockwise order
vertices = [[50,50], [200,50], [200,200], [50,200]]

# Also correct: Counter-clockwise order
vertices = [[50,50], [50,200], [200,200], [200,50]]
```

### Issue: Circular mask appears as ellipse

**Solution:** Ensure center coordinates are integers and radius is positive

```python
center_x = int(center_x)  # Convert to integer
center_y = int(center_y)
radius = abs(radius)      # Ensure positive
```

### Issue: Black regions where ROI should be

**Solution:** Verify mask has correct dtype (uint8) and values are 0 or 255

```python
assert mask.dtype == np.uint8, "Mask must be uint8"
assert set(mask.flatten()) == {0, 255}, "Mask must be binary"
```

## Advanced Topics

### Combining Multiple ROIs

Extract and combine regions:

```python
# Create composite mask
combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(combined_mask, (200, 200), 100, 255, -1)
cv2.circle(combined_mask, (400, 200), 100, 255, -1)

# Extract with combined mask
combined_roi = cv2.bitwise_and(img, img, mask=combined_mask)
```

### Inverse Masking

Extract everything EXCEPT a region:

```python
# Create circle mask
circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(circle_mask, center, radius, 255, -1)

# Invert mask
inverse_mask = cv2.bitwise_not(circle_mask)

# Extract inverse region
outside_circle = cv2.bitwise_and(img, img, mask=inverse_mask)
```

### Morphological Operations on Masks

Refine masks with morphology:

```python
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [vertices], 255)

# Clean up mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise

roi = cv2.bitwise_and(img, img, mask=mask)
```

## Next Steps

1. Run the module with synthetic test image
2. Understand each extraction method
3. Apply to your own images
4. Combine techniques for complex ROI extraction
5. Integrate into larger computer vision pipelines

## Help & Support

```bash
# Display command help
python roi_extraction_utility.py --help

# Run test suite
python test_roi_extraction.py

# Verify installation
python -c "from roi_extraction_utility import ROIExtractionUtility; print('✓ Module loaded successfully')"
```

## References

- **OpenCV Tutorials:** https://docs.opencv.org/
- **Region of Interest:** https://en.wikipedia.org/wiki/Region_of_interest
- **Image Masking:** OpenCV bitwise operations documentation
- **Geometric Shapes:** OpenCV drawing functions (circle, fillPoly, etc.)

