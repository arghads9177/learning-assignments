# Computer Vision Learning Assignments

A comprehensive collection of computer vision mini-lab assignments and hands-on exercises covering essential image and video processing techniques using OpenCV and Python.

## 📋 Overview

This repository contains structured learning modules that progressively build computer vision skills, from fundamental image processing techniques to advanced video analysis and dataset creation. Each module includes complete implementations, comprehensive test suites, detailed documentation, and practical examples.

**Author:** Argha Dey Sarkar  
**Email:** email2argha@gmail.com  
**Platform:** Linux (Ubuntu 22.04+)  
**Python Version:** 3.8+

---

## 📁 Repository Structure

```
learning-assignments/
├── README.md                          # This file
├── Chapter1/
│   ├── module8/                       # Morphological Visual Comparison
│   ├── module9/                       # Automated Coin Counter
│   ├── module10/                      # ROI Extraction Exercises
│   ├── module11/                      # Logo Overlay Utility
│   ├── module12/                      # (Planned)
│   └── module13/                      # Video Player & Frame Extractor
└── [Future chapters...]
```

---

## 🎓 Implemented Modules

### Module 8: Morphological Visual Comparison
**Objective:** Visualize and compare various morphological operations on images

**Key Features:**
- ✅ Multiple morphological operations (Erosion, Dilation, Opening, Closing)
- ✅ Side-by-side comparison dashboard with 3×3 grid layout
- ✅ Configurable kernel sizes and iteration counts
- ✅ Synthetic test image generation with shapes
- ✅ Custom image support
- ✅ Comprehensive statistical analysis
- ✅ High-quality visualization (150 DPI)

**Usage:**
```bash
cd Chapter1/module8
python morphological_visual_comparison.py
python morphological_visual_comparison.py --custom image.jpg --kernel-size 7 --iterations 3
```

**Key Concepts:**
- Erosion: Reduces white regions, enlarges black regions
- Dilation: Expands white regions, shrinks black regions
- Opening: Erosion followed by dilation (removes small objects)
- Closing: Dilation followed by erosion (fills small holes)

**Output:** 9-panel dashboard, statistics, comparison report

---

### Module 9: Automated Coin Counter
**Objective:** Detect, isolate, and count coins in images using advanced thresholding and morphology

**Key Features:**
- ✅ Otsu's binary thresholding for coin isolation
- ✅ Morphological closing for interior hole filling
- ✅ Contour detection with area-based filtering
- ✅ Centroid calculation using image moments
- ✅ Minimum enclosing circle for radius estimation
- ✅ Synthetic coin image generation (metallic rendering)
- ✅ 4-panel analysis dashboard
- ✅ Detailed statistical reporting

**Key Algorithms:**
- Otsu's Threshold (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
- Morphological Closing (cv2.morphologyEx with elliptical kernel)
- Contour Detection (cv2.findContours with area filtering)
- Image Moments (cv2.moments for centroid calculation)

**Usage:**
```bash
cd Chapter1/module9
python automated_coin_counter.py
python automated_coin_counter.py --custom coins.jpg --min-area 500 --kernel-size 9
```

**Output:** Detected coins with coordinates, statistics, 4-panel dashboard

---

### Module 10: ROI Extraction Exercises
**Objective:** Master three fundamental ROI extraction techniques for diverse applications

**Three Extraction Methods:**

1. **Face ROI (Bounding Box Slicing)**
   - Direct array slicing: `img[y:y+h, x:x+w]`
   - Fastest method (O(w×h))
   - Use cases: Face detection, object tracking

2. **Circular ROI (Vignette Isolation)**
   - Circle mask with bitwise AND
   - Optional gradient vignette fade
   - Use cases: Iris detection, circular objects

3. **Object ROI (Polygon Segment)**
   - Polygon mask using cv2.fillPoly
   - Supports arbitrary N-vertex shapes
   - Use cases: Road lanes, building detection, segmentation

**Key Features:**
- ✅ All three extraction methods with comprehensive docstrings
- ✅ Mask visualization for understanding
- ✅ 3×3 pipeline visualization dashboard
- ✅ Binary mask validation
- ✅ Synthetic test image generation
- ✅ Detailed analysis with coverage statistics

**Usage:**
```bash
cd Chapter1/module10
python roi_extraction_utility.py
python roi_extraction_utility.py --custom photo.jpg --position 200 150
```

**Output:** 3 extracted ROIs, 3 masks, dashboard, analysis report

---

### Module 11: Logo Overlay Utility
**Objective:** Seamlessly blend irregular, non-rectangular logos onto background images

**Key Concepts:**
- ✅ Binary masking for precise logo geometry isolation
- ✅ Inverse masking for safe background clearing
- ✅ Bitwise operations (AND, NOT, ADD)
- ✅ No rectangular framing artifacts
- ✅ Dynamic resize threshold (configurable 1-99%)
- ✅ Position clamping with boundary validation

**Pipeline (9 Steps):**
1. Load/generate images
2. Resize logo if exceeds threshold %
3. Isolate background ROI
4. Create binary mask (Otsu's threshold)
5. Create inverse mask
6. Clear logo area from background
7. Isolate logo colors
8. Blend using cv2.add
9. Inject composite back into background

**Usage:**
```bash
cd Chapter1/module11
python logo_overlay_utility.py
python logo_overlay_utility.py --background bg.jpg --logo logo.png --resize-threshold 30
python logo_overlay_utility.py --background bg.jpg --logo logo.png --resize-threshold 50
```

**Output:** 12 files including original, masks, blended composite, dashboard, report

---

### Module 13: Video Player & Frame Extractor
**Objective:** Play videos at native timing and extract frames for dataset creation

**Dual Functionality:**
- ✅ Interactive video playback with frame overlay
- ✅ Periodic frame extraction (configurable intervals)
- ✅ Live webcam capture support
- ✅ Video metadata extraction (FPS, resolution, duration)
- ✅ Program continuation after user interaction
- ✅ Automatic frame numbering (frame_XXXXXX.png)
- ✅ Dataset size calculation and reporting

**Interactive Controls:**
- **q** - Quit playback (program continues to generate reports)
- **p** - Pause/Resume playback
- After pressing 'q' → Dashboard generation → Report creation → Exit

**Usage:**
```bash
cd Chapter1/module13

# Synthetic video
python video_player_extractor.py

# Custom video file
python video_player_extractor.py --video video.mp4 --frame-interval 30

# Webcam capture
python video_player_extractor.py --webcam --webcam-duration 10

# No playback, just extraction
python video_player_extractor.py --video video.mp4 --no-play --max-frames 100
```

**Output:** Extracted frames (PNG), 4×4 dashboard, analysis report

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Linux/Ubuntu operating system (recommended)
- Webcam (optional, for Module 13 webcam features)

### Quick Installation

```bash
# Navigate to repository
cd /home/argha-ds/datascience/computer\ vision/learning-assignments

# Install dependencies
pip install opencv-python numpy matplotlib

# Verify installation
python -c "import cv2; print(f'OpenCV {cv2.__version__} installed')"
```

---

## 🚀 Quick Start Guide

### Try the Quickest Module (Video Player)
```bash
cd Chapter1/module13
python video_player_extractor.py
# Generates synthetic video and extracts frames automatically

# Check outputs
ls output_video_frames/
# You'll see: frame_*.png, extraction_dashboard.png, video_extraction_report.txt
```

### Or Start with Coin Detection
```bash
cd Chapter1/module9
python automated_coin_counter.py
# Generates synthetic coins and detects them automatically

# Check outputs
ls output_coins/
```

### Process Your Own Data
```bash
# Analyze coins in your image
cd Chapter1/module9
python automated_coin_counter.py --custom my_coins.jpg --min-area 400

# Extract frames from your video
cd Chapter1/module13
python video_player_extractor.py --video my_video.mp4 --frame-interval 30
```

---

## 📊 Features Summary

| Feature | M8 | M9 | M10 | M11 | M13 |
|---------|----|----|-----|-----|-----|
| Synthetic Data Generation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Custom Image Support | ✅ | ✅ | ✅ | ✅ | ✅ |
| Comprehensive Docstrings | ✅ | ✅ | ✅ | ✅ | ✅ |
| Test Suite (5-10 tests) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dashboard Visualization | ✅ | ✅ | ✅ | ✅ | ✅ |
| Analysis Reports | ✅ | ✅ | ✅ | ✅ | ✅ |
| Command-Line Interface | ✅ | ✅ | ✅ | ✅ | ✅ |
| Python API | ✅ | ✅ | ✅ | ✅ | ✅ |
| Documentation (700+ lines) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Configurable Parameters | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 💻 Usage Examples

### Example 1: Programmatic Usage (Python)
```python
# Module 9: Coin Counter API
from Chapter1.module9.automated_coin_counter import AutomatedCoinCounter

counter = AutomatedCoinCounter(output_dir='./coin_results')
coins, result = counter.run_analysis('coins.jpg', min_area=400)
print(f"Detected {len(coins)} coins")

# Module 13: Video Extraction API
from Chapter1.module13.video_player_extractor import VideoPlayerExtractor

extractor = VideoPlayerExtractor(output_dir='./frames')
frames, props = extractor.run_analysis(
    video_path='video.mp4',
    frame_interval=30,
    play_video=False
)
print(f"Extracted {len(frames)} frames")
```

### Example 2: Batch Processing
```bash
# Process multiple videos with Module 13
for video in videos/*.mp4; do
    echo "Extracting from $video..."
    python Chapter1/module13/video_player_extractor.py \
        --video "$video" \
        --no-play \
        --output-dir "./dataset/$(basename $video .mp4)"
done
```

### Example 3: Command-Line Workflows
```bash
# Extract frames → Analyze for coins → Generate reports
cd Chapter1/module13
python video_player_extractor.py --video coins.mp4 --no-play --max-frames 10

cd Chapter1/module9
python automated_coin_counter.py --custom output_video_frames/frame_000000.png
```

---

## 🧪 Testing

Each module includes comprehensive test suites. All tests pass successfully:

```bash
# Test individual modules
cd Chapter1/module8 && python test_morphological_comparison.py
cd Chapter1/module9 && python test_coin_counter.py
cd Chapter1/module10 && python test_roi_extraction.py
cd Chapter1/module11 && python test_logo_overlay.py
cd Chapter1/module13 && python test_video_basic.py
```

**Test Coverage:**
- ✅ Module 8: 5 comprehensive tests
- ✅ Module 9: 5 comprehensive tests
- ✅ Module 10: 5 comprehensive tests
- ✅ Module 11: 10 comprehensive tests (with dynamic threshold)
- ✅ Module 13: 8 comprehensive tests

**Total:** 33+ passing test cases

---

## 📖 Documentation

Each module includes extensive documentation:

```bash
# View comprehensive usage guides
cat Chapter1/module8/MORPHOLOGICAL_COMPARISON_USAGE.md
cat Chapter1/module9/COIN_COUNTER_USAGE.md
cat Chapter1/module10/ROI_EXTRACTION_USAGE.md
cat Chapter1/module11/LOGO_OVERLAY_USAGE.md
cat Chapter1/module13/VIDEO_PLAYER_USAGE.md
```

**Documentation includes:**
- Complete usage guide (700+ lines per module)
- NumPy/Google style docstrings
- Practical examples and use cases
- Troubleshooting guides
- Performance characteristics
- API reference and parameters

---

## 🎯 Learning Objectives

This repository helps you master:

1. **Image Processing Fundamentals**
   - Thresholding (Otsu's method)
   - Morphological operations
   - Contour detection and analysis

2. **Computer Vision Algorithms**
   - Image moments for centroid calculation
   - Binary masking and bitwise operations
   - ROI extraction techniques
   - Shape detection and measurement

3. **Video Processing**
   - Video playback and timing
   - Frame extraction and sampling
   - Webcam capture and live processing
   - Dataset compilation from video

4. **Software Engineering Best Practices**
   - Comprehensive docstrings
   - Test-driven development
   - Command-line interfaces
   - API design
   - Error handling
   - Performance optimization

5. **Data Visualization**
   - Dashboard creation with matplotlib
   - Multi-panel layouts
   - High-DPI output
   - Statistical visualization

---

## 🔧 Advanced Features

### Dynamic Parameters
```bash
# Module 9: Adjust detection sensitivity
python Chapter1/module9/automated_coin_counter.py --min-area 300 --kernel-size 7

# Module 11: Control logo sizing
python Chapter1/module11/logo_overlay_utility.py --resize-threshold 50

# Module 13: Customize extraction
python Chapter1/module13/video_player_extractor.py --frame-interval 15 --max-frames 100
```

### Custom Output Directories
```bash
python Chapter1/module9/automated_coin_counter.py --output-dir ./my_results
python Chapter1/module13/video_player_extractor.py --output-dir ./my_frames
```

---

## 📊 Output Examples

### Module 9 (Coin Detection) Output
```
output_coins/
├── coin_analysis_dashboard.png (4-panel visualization)
├── coin_analysis_report.txt (statistics and analysis)
├── coin_detection.png (detected coins with circles)
├── binary_threshold.png (thresholding result)
├── morphological_closing.png (morphology output)
└── original_image.png
```

### Module 13 (Video Extraction) Output
```
output_video_frames/
├── frame_000000.png (extracted frame 0)
├── frame_000001.png (extracted frame 1)
├── ... (more extracted frames)
├── extraction_dashboard.png (4×4 grid visualization)
└── video_extraction_report.txt
```

---

## 🐛 Troubleshooting

### Issue: Module not found
```bash
# Ensure you're in the correct directory
cd Chapter1/module9
python automated_coin_counter.py
```

### Issue: Dependencies not installed
```bash
# Install required packages
pip install opencv-python numpy matplotlib
```

### Issue: Webcam not detected (Module 13)
```bash
# Check available cameras
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera found at index {i}')
        cap.release()
"
```

### Issue: Video file cannot be opened
```bash
# Verify the file exists and is readable
ls -lh video.mp4

# Try converting to standard MP4 format
ffmpeg -i video.mkv -c:v libx264 -c:a aac video.mp4
```

---

## 📚 References

- **OpenCV Documentation:** https://docs.opencv.org/
- **NumPy Guide:** https://numpy.org/doc/
- **Matplotlib:** https://matplotlib.org/
- **Computer Vision Fundamentals:** Image processing and analysis techniques
- **Video Processing:** Codecs, frame extraction, real-time processing

---

## 📝 Module Development Status

| Module | Status | Tests | Documentation | Lines of Code |
|--------|--------|-------|---|---|
| Module 8 | ✅ Complete | ✅ 5/5 | ✅ 600+ lines | 350+ |
| Module 9 | ✅ Complete | ✅ 5/5 | ✅ 600+ lines | 650+ |
| Module 10 | ✅ Complete | ✅ 5/5 | ✅ 700+ lines | 420+ |
| Module 11 | ✅ Complete | ✅ 10/10 | ✅ 800+ lines | 580+ |
| Module 12 | 🟡 Planned | - | - | - |
| Module 13 | ✅ Complete | ✅ 8/8 | ✅ 700+ lines | 600+ |
| **TOTAL** | **5/6** | **33/33** | **3,400+ lines** | **2,600+** |

---

## 🎓 Recommended Learning Path

1. **Start with Module 8** - Understand morphological operations
2. **Progress to Module 9** - Apply operations to real problem (coin detection)
3. **Learn Module 10** - Master ROI extraction techniques
4. **Master Module 11** - Advanced image blending with masking
5. **Advance to Module 13** - Video processing and dataset creation

---

## 👨‍💼 Author & Contact

**Argha Dey Sarkar**  
📧 email2argha@gmail.com  
🏠 Location: Computer Vision Learning Lab

---

## 📄 License

These learning assignments are provided as educational material for computer vision learning and research purposes.

---

## 🚀 Getting Started Now

```bash
# Navigate to repository
cd /home/argha-ds/datascience/computer\ vision/learning-assignments

# Try the most accessible module
cd Chapter1/module13
python video_player_extractor.py

# Verify the output was created
ls output_video_frames/
# Output: frame_000000.png, frame_000001.png, ..., extraction_dashboard.png, video_extraction_report.txt

# Read the comprehensive guide
cat VIDEO_PLAYER_USAGE.md

# Run the test suite
python test_video_basic.py
```

---

**Last Updated:** July 2026  
**Repository Size:** 2,600+ lines of code  
**Total Test Cases:** 33+ comprehensive tests  
**Total Documentation:** 3,400+ lines  
**Status:** ✅ Production Ready
