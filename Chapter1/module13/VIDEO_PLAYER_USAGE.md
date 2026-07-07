# Module 13: Video Player & Frame Extractor

## Overview

The Video Player & Frame Extractor utility provides dual functionality for video processing:

1. **Video Playback** - Play videos at native timing with real-time frame information overlay
2. **Frame Extraction** - Extract frames at periodic intervals (e.g., every 30th frame) to create training datasets

This tool is essential for video dataset compilation, allowing you to sample frames efficiently from long video files while maintaining native playback timing for analysis.

## Key Features

✅ **Dual Functionality** - Both playback and frame extraction in one tool  
✅ **Native Timing** - Videos play at original FPS for realistic analysis  
✅ **Periodic Sampling** - Extract frames at configurable intervals  
✅ **Video Metadata** - Extract FPS, resolution, duration, frame count  
✅ **Synthetic Video Generation** - Create test videos automatically  
✅ **Custom Positioning** - Save frames with automatic numbering  
✅ **Visualization Dashboard** - 4×4 grid of extracted frames  
✅ **Comprehensive Reporting** - Detailed analysis and dataset info  
✅ **Dataset Compilation** - Efficient frame sampling for training data  

## Pipeline Architecture

```
Load Video
    ↓
Extract Video Properties
    ↓
[Optional] Play Video at Native Timing
    ↓
[Optional] Extract Frames at Intervals
    ↓
Save Frames with Numbering
    ↓
Create Dashboard Visualization
    ↓
Generate Analysis Report
```

## How to Use

### Quick Start - Generate & Analyze Synthetic Video

```bash
python video_player_extractor.py
```

**Output:**
- Synthetic MP4 video (10 seconds, 30 FPS)
- 30 extracted frames (every 10th frame)
- Visualization dashboard (4×4 grid)
- Detailed analysis report

### Using Custom Video Files

#### **Simple Frame Extraction**
```bash
python video_player_extractor.py --video my_video.mp4
```

#### **Custom Extraction Interval**
```bash
# Extract every 15th frame (fewer frames for large videos)
python video_player_extractor.py --video my_video.mp4 --frame-interval 15
```

#### **Limit Total Frames Extracted**
```bash
# Extract up to 100 frames maximum
python video_player_extractor.py --video my_video.mp4 --max-frames 100
```

#### **Complete Example with All Options**
```bash
python video_player_extractor.py \
    --video /path/to/video.mp4 \
    --frame-interval 30 \
    --no-play \
    --max-frames 200 \
    --output-dir ./dataset_frames
```

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--video` | path | None | Path to video file (MP4, AVI, MOV, MKV, FLV) |
| `--frame-interval` | int | 30 | Extract every nth frame (use 1 for all frames) |
| `--play` | flag | True | Play video during processing (interactive) |
| `--no-play` | flag | False | Skip video playback |
| `--extract` | flag | True | Extract and save frames |
| `--no-extract` | flag | False | Skip frame extraction |
| `--max-frames` | int | None | Maximum frames to extract (if None, extracts all) |
| `--output-dir` | path | ./output_video_frames | Directory for extracted frames |

## Output Files

### Generated in Output Directory:

```
output_video_frames/
├── frame_000000.png           # First extracted frame
├── frame_000001.png           # Second extracted frame
├── frame_XXXXXX.png           # Last extracted frame
├── extraction_dashboard.png   # 4×4 grid visualization (150 DPI)
└── video_extraction_report.txt # Detailed analysis report
```

### File Descriptions:

**frame_XXXXXX.png**
- Extracted frames with 6-digit zero-padded numbering
- PNG format for lossless compression
- Full resolution (matches video resolution)
- Sequential numbering starting from 000000

**extraction_dashboard.png**
- 4×4 grid showing sample of extracted frames
- Up to 16 frames displayed with timestamps
- High-DPI output (150 DPI) for quality visualization
- Shows frame numbers and timestamps for each

**video_extraction_report.txt**
- Video properties (resolution, FPS, duration, frame count)
- Extraction parameters and results
- Dataset information (file sizes, frame count)
- Frame numbering scheme documentation

## Frame Extraction Strategy

### Frame Intervals

| Interval | Use Case | Example |
|----------|----------|---------|
| **1** | All frames | Dataset requires every frame (memory intensive) |
| **5** | Dense sampling | 10-20% of frames for fine details |
| **15** | Medium sampling | 5-10% of frames for balanced size |
| **30** | Sparse sampling | 3-5% of frames for large datasets (default) |
| **60** | Very sparse | 1-2% for quick overviews |

### Calculating Frames Extracted

```
Frames Extracted = Total Frames / Frame Interval

Example:
- Video: 300 frames (10 seconds @ 30 FPS)
- Interval: 30
- Result: 300 / 30 = 10 frames extracted
```

### Dataset Size Calculation

```
Dataset Size = Extracted Frames × Frame Size

Example:
- Extracted: 100 frames
- Resolution: 640×480 (307,200 pixels)
- Channels: 3 (BGR)
- Per Frame: ~307K pixels × 3 × 1 byte = ~900 KB (PNG ~200 KB)
- Total: ~100 frames × 0.2 MB = ~20 MB
```

## Programmatic Usage (Python API)

### Basic Video Information

```python
from video_player_extractor import VideoPlayerExtractor
import cv2

extractor = VideoPlayerExtractor(output_dir='./results')

# Load and get properties
cap, video_path = extractor.load_video('my_video.mp4')
props = extractor.get_video_properties(cap)
cap.release()

print(f"Resolution: {props['width']}×{props['height']}")
print(f"FPS: {props['fps']}")
print(f"Total Frames: {props['frame_count']}")
print(f"Duration: {props['duration_seconds']:.2f}s")
```

### Extract Frames with Custom Interval

```python
from video_player_extractor import VideoPlayerExtractor

extractor = VideoPlayerExtractor(output_dir='./dataset')

# Extract every 15th frame (up to 200 frames max)
frames = extractor.extract_frames(
    video_path='video.mp4',
    frame_interval=15,
    max_frames=200
)

print(f"Extracted {len(frames)} frames")
for idx, frame in enumerate(frames):
    print(f"Frame {idx}: shape={frame.shape}, dtype={frame.dtype}")
```

### Complete Analysis Pipeline

```python
from video_player_extractor import VideoPlayerExtractor

extractor = VideoPlayerExtractor(output_dir='./analysis')

# Run full pipeline (no playback, extract frames)
frames, props = extractor.run_analysis(
    video_path='video.mp4',
    frame_interval=30,
    play_video=False,  # Skip interactive playback
    extract_frames_flag=True,
    max_frames=None  # Extract all
)

print(f"Extracted {len(frames)} frames from {props['frame_count']} total")
```

### Play Video with Frame Overlay

```python
from video_player_extractor import VideoPlayerExtractor

extractor = VideoPlayerExtractor()

# Play video with real-time frame information
# Press 'q' to quit, 'p' to pause
extractor.play_video(video_path='video.mp4')
```

## Practical Examples

### Example 1: Create Image Classification Dataset

**Scenario:** Extract frames from video clips for training an image classifier

```bash
# Extract every 30th frame for balanced dataset size
python video_player_extractor.py \
    --video action_video_1.mp4 \
    --frame-interval 30 \
    --no-play \
    --output-dir ./action_dataset
```

### Example 2: Object Detection Dataset from Multiple Videos

**Scenario:** Extract frames from multiple video sources

```bash
for video in videos/*.mp4; do
    echo "Processing $video..."
    python video_player_extractor.py \
        --video "$video" \
        --frame-interval 30 \
        --no-play \
        --output-dir "./detection_dataset/$(basename $video .mp4)"
done
```

### Example 3: Temporal Sequence Extraction

**Scenario:** Extract frames for video understanding tasks (every frame or minimal sampling)

```bash
# Extract every frame for temporal models
python video_player_extractor.py \
    --video surveillance.mp4 \
    --frame-interval 1 \
    --no-play \
    --output-dir ./surveillance_frames
```

### Example 4: Quick Video Preview

**Scenario:** Extract subset for quick overview without full processing

```bash
# Extract only 25 frames for preview
python video_player_extractor.py \
    --video long_video.mp4 \
    --frame-interval 1 \
    --max-frames 25 \
    --no-play \
    --output-dir ./preview
```

### Example 5: Batch Processing with Script

**Scenario:** Process entire video directory with consistent parameters

```python
import os
from video_player_extractor import VideoPlayerExtractor

video_dir = './videos'
output_dir = './extracted_frames'

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_file)
        dataset_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(output_dir, dataset_name)

        extractor = VideoPlayerExtractor(output_dir=output_path)
        frames, props = extractor.run_analysis(
            video_path=video_path,
            frame_interval=30,
            play_video=False,
            max_frames=500  # Limit to 500 frames per video
        )

        print(f"✓ {video_file}: {len(frames)} frames extracted")
```

## Understanding Video Properties

### FPS (Frames Per Second)

```
Common FPS values:
- 24 fps: Film/cinema standard
- 25 fps: PAL TV standard (Europe)
- 29.97 fps: NTSC TV standard (US, Japan)
- 30 fps: Video standard
- 60 fps: High-speed video

Frame Extraction Impact:
- Higher FPS videos have more frames per second
- Same interval extracts fewer frames from 24fps vs 60fps
```

### Resolution

```
Common video resolutions:
- 480p: 640×480 pixels (0.3 MP)
- 720p: 1280×720 pixels (0.9 MP)
- 1080p: 1920×1080 pixels (2 MP)
- 4K: 3840×2160 pixels (8 MP)

Frame Size Impact:
- Higher resolution = larger frame files
- 1080p frame: ~2 MB raw, ~200-400 KB PNG
- 4K frame: ~8 MB raw, ~1-2 MB PNG
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Video Loading** | <100ms | Varies by codec and file size |
| **Frame Extraction** | ~1-5ms per frame | Depends on resolution |
| **Dashboard Creation** | ~500ms | Matplotlib rendering |
| **Memory Usage** | ~50-200 MB | Depends on video resolution |

### Optimization Tips:

1. **Choose appropriate intervals** - 30 is good default
2. **Limit max_frames for large videos** - Prevents excessive memory use
3. **Use no-play flag** - Faster processing without visualization
4. **Process offline** - Extract once, use many times
5. **Batch similar videos** - Consistent parameters across dataset

## Troubleshooting

### Issue: Video file cannot be opened

**Possible Causes:**
1. Unsupported codec
2. Corrupted video file
3. Invalid file path

**Solutions:**
```bash
# Check file exists and is readable
ls -lh video.mp4

# Verify with FFmpeg (if installed)
ffprobe video.mp4

# Try converting to MP4 H.264
ffmpeg -i video.mkv -c:v libx264 -c:a aac video.mp4
```

### Issue: No frames extracted

**Possible Causes:**
1. frame_interval too large (larger than total frames)
2. max_frames set too small
3. Video has issues

**Solutions:**
```bash
# Check video properties first
python -c "
from video_player_extractor import VideoPlayerExtractor
ext = VideoPlayerExtractor()
cap, _ = ext.load_video('video.mp4')
props = ext.get_video_properties(cap)
print(f'Total frames: {props[\"frame_count\"]}')
print(f'With interval 30: {props[\"frame_count\"] // 30} frames')
"
```

### Issue: Slow frame extraction

**Possible Causes:**
1. High resolution video
2. Slow disk I/O
3. Filesystem limitations

**Solutions:**
```bash
# Extract to SSD (faster I/O)
python video_player_extractor.py --video video.mp4 --output-dir /mnt/ssd/frames

# Use larger intervals for preview
python video_player_extractor.py --video video.mp4 --frame-interval 100 --max-frames 100

# Reduce resolution if post-processing is OK
```

## Video Format Support

| Format | Codec | Support | Notes |
|--------|-------|---------|-------|
| MP4 | H.264, H.265 | ✓ Excellent | Most common, widely supported |
| AVI | MPEG-4, DivX | ✓ Good | Legacy format, usually works |
| MOV | H.264, ProRes | ✓ Good | Apple format, generally works |
| MKV | Various | ✓ Good | Container format, codec dependent |
| FLV | H.264 | ✓ Fair | Less common, may have issues |
| WebM | VP8, VP9 | ✓ Fair | Web format, slower processing |

## Frame Numbering Scheme

### Naming Convention

```
frame_XXXXXX.png

Where:
- 6-digit zero-padded index
- Starts at 000000 for first extracted frame
- Increments by 1 for each extracted frame

Examples:
- frame_000000.png (1st extracted frame)
- frame_000001.png (2nd extracted frame)
- frame_000099.png (100th extracted frame)
```

### Mapping to Video Frames

```
Extracted Frame Index → Video Frame Number

With interval=30:
- Extracted frame 0 → Video frame 0
- Extracted frame 1 → Video frame 30
- Extracted frame 2 → Video frame 60
- Extracted frame N → Video frame N × 30
```

## Next Steps

1. Run with synthetic video to understand functionality
2. Try with your own video file
3. Experiment with different frame intervals
4. Create datasets for your machine learning tasks
5. Integrate extracted frames into training pipelines

## Help & Support

```bash
# Display command help
python video_player_extractor.py --help

# Run tests to verify installation
python test_video_basic.py
```

## References

- **OpenCV VideoCapture:** https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html
- **Common Video Codecs:** https://en.wikipedia.org/wiki/Video_codec
- **FFmpeg Documentation:** https://ffmpeg.org/documentation.html
- **Video Frame Extraction:** Best practices for dataset creation
