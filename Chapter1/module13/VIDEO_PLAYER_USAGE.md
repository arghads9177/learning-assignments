# Module 13: Video Player & Frame Extractor

## Overview

The Video Player & Frame Extractor utility provides dual functionality for video processing:

1. **Video Playback** - Play videos at native timing with real-time frame information overlay
2. **Frame Extraction** - Extract frames at periodic intervals (e.g., every 30th frame) to create training datasets
3. **Webcam Capture** - Capture frames directly from webcam for live dataset collection

This tool is essential for video dataset compilation, allowing you to sample frames efficiently from long video files, local files, or webcam sources while maintaining native playback timing for analysis.

## Key Features

✅ **Dual Functionality** - Both playback and frame extraction in one tool  
✅ **Webcam Support** - Capture directly from webcam  
✅ **Native Timing** - Videos play at original FPS for realistic analysis  
✅ **Periodic Sampling** - Extract frames at configurable intervals  
✅ **Automatic Continuation** - Program continues after 'q' press to generate reports  
✅ **Video Metadata** - Extract FPS, resolution, duration, frame count  
✅ **Synthetic Video Generation** - Create test videos automatically  
✅ **Custom Positioning** - Save frames with automatic numbering  
✅ **Visualization Dashboard** - 4×4 grid of extracted frames  
✅ **Comprehensive Reporting** - Detailed analysis and dataset info  

## Pipeline Architecture

```
Load Video / Webcam / Generate Synthetic
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
    ↓
Complete Execution & Exit
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

### Using Webcam for Live Capture

#### **Simple Webcam Capture**
```bash
# Capture from webcam for 5 seconds (default)
python video_player_extractor.py --webcam
```

#### **Custom Duration Webcam Capture**
```bash
# Capture for 10 seconds
python video_player_extractor.py --webcam --webcam-duration 10
```

#### **Webcam with Custom Extraction**
```bash
# Capture for 15 seconds, extract every 3rd frame (max 50 frames)
python video_player_extractor.py \
    --webcam \
    --webcam-duration 15 \
    --frame-interval 3 \
    --max-frames 50
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
| `--webcam` | flag | False | Use webcam instead of video file |
| `--webcam-duration` | float | 5 | Duration for webcam capture in seconds |
| `--frame-interval` | int | 30 | Extract every nth frame (use 1 for all frames) |
| `--play` | flag | True | Play video during processing (interactive) |
| `--no-play` | flag | False | Skip video playback |
| `--extract` | flag | True | Extract and save frames |
| `--no-extract` | flag | False | Skip frame extraction |
| `--max-frames` | int | None | Maximum frames to extract (if None, extracts all) |
| `--output-dir` | path | ./output_video_frames | Directory for extracted frames |

## Interactive Playback Controls

When playing a video or capturing from webcam:

| Key | Action |
|-----|--------|
| **q** | Quit playback - program continues to generate reports |
| **p** | Pause/Resume playback |
| **ESC** | Close window (alternative to q) |

After pressing 'q', the program will:
1. Close the playback window
2. Continue with frame extraction (if enabled)
3. Generate visualization dashboard
4. Create analysis report
5. Exit normally

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

| Interval | Use Case | Coverage |
|----------|----------|----------|
| **1** | All frames | 100% (full video) |
| **5** | Dense sampling | 20% of frames |
| **15** | Medium sampling | 7% of frames |
| **30** | Sparse sampling | 3% of frames (default) |
| **60** | Very sparse | 1.5% of frames |

## Programmatic Usage (Python API)

### Basic Video Information

```python
from video_player_extractor import VideoPlayerExtractor

extractor = VideoPlayerExtractor(output_dir='./results')

# Load and get properties
cap, video_path = extractor.load_video('my_video.mp4')
props = extractor.get_video_properties(cap)
cap.release()

print(f"Resolution: {props['width']}×{props['height']}")
print(f"FPS: {props['fps']}")
print(f"Total Frames: {props['frame_count']}")
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
```

### Webcam Capture

```python
from video_player_extractor import VideoPlayerExtractor

extractor = VideoPlayerExtractor(output_dir='./webcam_data')

# Capture from webcam for 10 seconds
frames = extractor.extract_frames(
    video_path=None,
    use_webcam=True,
    frame_interval=5,
    duration_seconds=10,
    max_frames=None
)

print(f"Captured {len(frames)} frames from webcam")
```

### Complete Analysis Pipeline

```python
from video_player_extractor import VideoPlayerExtractor

extractor = VideoPlayerExtractor(output_dir='./analysis')

# Run full pipeline
frames, props = extractor.run_analysis(
    video_path='video.mp4',
    frame_interval=30,
    play_video=False,
    extract_frames_flag=True,
    max_frames=None
)

print(f"Extracted {len(frames)} frames")
```

## Practical Examples

### Example 1: Create Image Classification Dataset

```bash
# Extract every 30th frame for balanced dataset size
python video_player_extractor.py \
    --video action_video_1.mp4 \
    --frame-interval 30 \
    --no-play \
    --output-dir ./action_dataset
```

### Example 2: Live Webcam Dataset Collection

```bash
# Capture 30 seconds of video, extract every 5th frame
python video_player_extractor.py \
    --webcam \
    --webcam-duration 30 \
    --frame-interval 5 \
    --output-dir ./live_dataset
```

### Example 3: Play Video with Automatic Report

```bash
# Watch video (press 'q' to stop), then automatic report generation
python video_player_extractor.py \
    --video my_video.mp4 \
    --play \
    --max-frames 50 \
    --output-dir ./preview
```

### Example 4: Batch Process Multiple Videos

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
            max_frames=500
        )

        print(f"✓ {video_file}: {len(frames)} frames extracted")
```

## Video Format Support

| Format | Support | Notes |
|--------|---------|-------|
| MP4 | ✓ Excellent | Most common, widely supported |
| AVI | ✓ Good | Legacy format, usually works |
| MOV | ✓ Good | Apple format, generally works |
| MKV | ✓ Good | Container format, codec dependent |
| FLV | ✓ Fair | Less common, may have issues |
| WebM | ✓ Fair | Web format, slower processing |

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
