import cv2
import numpy as np
import os
import shutil
from video_player_extractor import VideoPlayerExtractor


def test_synthetic_video_generation():
    """Test synthetic test video generation."""
    print("\n" + "="*70)
    print("TEST 1: Synthetic Video Generation")
    print("="*70)

    extractor = VideoPlayerExtractor('./test_output_video_gen')
    video_path = extractor.generate_test_video(duration_seconds=3, fps=30, width=640, height=480)

    assert os.path.exists(video_path), "Generated video file not found"
    assert os.path.getsize(video_path) > 0, "Generated video is empty"

    # Verify video can be read
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Generated video cannot be opened"
    cap.release()

    print("\n✓ Test 1 PASSED: Synthetic video generated successfully")
    print(f"  - Video path: {video_path}")
    print(f"  - File size: {os.path.getsize(video_path) / (1024**2):.2f} MB")


def test_video_properties_extraction():
    """Test extraction of video properties and metadata."""
    print("\n" + "="*70)
    print("TEST 2: Video Properties Extraction")
    print("="*70)

    extractor = VideoPlayerExtractor('./test_output_video_props')
    cap, _ = extractor.load_video()
    props = extractor.get_video_properties(cap)
    cap.release()

    assert props is not None, "Properties dict is None"
    assert 'fps' in props, "Missing fps property"
    assert 'frame_count' in props, "Missing frame_count property"
    assert 'width' in props, "Missing width property"
    assert 'height' in props, "Missing height property"
    assert 'duration_seconds' in props, "Missing duration_seconds property"

    assert props['fps'] > 0, "FPS should be positive"
    assert props['frame_count'] > 0, "Frame count should be positive"
    assert props['width'] > 0 and props['height'] > 0, "Resolution invalid"
    assert props['duration_seconds'] > 0, "Duration should be positive"

    print("\n✓ Test 2 PASSED: Video properties extracted successfully")
    print(f"  - Resolution: {props['width']}×{props['height']}")
    print(f"  - FPS: {props['fps']:.2f}")
    print(f"  - Total Frames: {props['frame_count']}")
    print(f"  - Duration: {props['duration_seconds']:.2f} seconds")


def test_frame_extraction():
    """Test frame extraction at specified intervals."""
    print("\n" + "="*70)
    print("TEST 3: Frame Extraction")
    print("="*70)

    extractor = VideoPlayerExtractor('./test_output_video_extract')
    frames = extractor.extract_frames(frame_interval=10, max_frames=None)

    assert frames is not None, "Extracted frames is None"
    assert len(frames) > 0, "No frames extracted"

    # Verify frame properties
    for idx, frame in enumerate(frames):
        assert isinstance(frame, np.ndarray), f"Frame {idx} is not numpy array"
        assert frame.shape[2] == 3, f"Frame {idx} should be BGR (3 channels)"
        assert frame.dtype == np.uint8, f"Frame {idx} should be uint8"

    # Check that files were saved
    extracted_files = [f for f in os.listdir(extractor.output_dir) if f.startswith('frame_')]
    assert len(extracted_files) == len(frames), "Mismatch in saved frames"

    print("\n✓ Test 3 PASSED: Frame extraction successful")
    print(f"  - Frames extracted: {len(frames)}")
    print(f"  - Frame shape: {frames[0].shape}")
    print(f"  - Files saved: {len(extracted_files)}")


def test_frame_interval_variations():
    """Test extraction with different frame intervals."""
    print("\n" + "="*70)
    print("TEST 4: Frame Interval Variations")
    print("="*70)

    extractor = VideoPlayerExtractor('./test_output_video_intervals')

    # Test different intervals
    intervals = [1, 5, 15]
    results = {}

    for interval in intervals:
        # Clear output directory
        for f in os.listdir(extractor.output_dir):
            if f.startswith('frame_'):
                os.remove(os.path.join(extractor.output_dir, f))

        frames = extractor.extract_frames(frame_interval=interval, max_frames=None)
        results[interval] = len(frames)

    # Verify relationships: larger interval = fewer frames
    for i in range(len(intervals) - 1):
        assert results[intervals[i]] > results[intervals[i+1]], \
            f"Interval {intervals[i]} should extract more frames than {intervals[i+1]}"

    print("\n✓ Test 4 PASSED: Frame interval variations work correctly")
    for interval, count in results.items():
        print(f"  - Interval {interval:2d}: {count:3d} frames extracted")


def test_max_frames_limit():
    """Test max_frames parameter for limiting extraction."""
    print("\n" + "="*70)
    print("TEST 5: Max Frames Limit")
    print("="*70)

    extractor = VideoPlayerExtractor('./test_output_video_max')

    # Test with max_frames limit
    max_limit = 5
    frames = extractor.extract_frames(frame_interval=1, max_frames=max_limit)

    assert len(frames) <= max_limit, f"Extracted {len(frames)} frames, expected <= {max_limit}"
    assert len(frames) == max_limit, f"Expected exactly {max_limit} frames, got {len(frames)}"

    # Verify files
    extracted_files = [f for f in os.listdir(extractor.output_dir) if f.startswith('frame_')]
    assert len(extracted_files) == max_limit, f"Expected {max_limit} files, got {len(extracted_files)}"

    print("\n✓ Test 5 PASSED: Max frames limit works correctly")
    print(f"  - Max limit set: {max_limit}")
    print(f"  - Frames extracted: {len(frames)}")


def test_extraction_dashboard_creation():
    """Test creation of extraction visualization dashboard."""
    print("\n" + "="*70)
    print("TEST 6: Extraction Dashboard Creation")
    print("="*70)

    extractor = VideoPlayerExtractor('./test_output_video_dashboard')
    frames = extractor.extract_frames(frame_interval=10)

    # Create dashboard
    cap, _ = extractor.load_video()
    properties = extractor.get_video_properties(cap)
    cap.release()

    extractor.create_extraction_dashboard(frames, frame_interval=10, fps=properties['fps'])

    # Check dashboard file exists
    dashboard_path = os.path.join(extractor.output_dir, 'extraction_dashboard.png')
    assert os.path.exists(dashboard_path), "Dashboard file not created"
    assert os.path.getsize(dashboard_path) > 0, "Dashboard file is empty"

    print("\n✓ Test 6 PASSED: Dashboard created successfully")
    print(f"  - Frames in dashboard: {len(frames)}")
    print(f"  - Dashboard size: {os.path.getsize(dashboard_path) / (1024**2):.2f} MB")


def test_report_generation():
    """Test detailed analysis report generation."""
    print("\n" + "="*70)
    print("TEST 7: Report Generation")
    print("="*70)

    extractor = VideoPlayerExtractor('./test_output_video_report')

    # Extract frames and generate report
    frames = extractor.extract_frames(frame_interval=10)
    cap, _ = extractor.load_video()
    properties = extractor.get_video_properties(cap)
    cap.release()

    extractor.save_report(properties, frame_interval=10, extracted_count=len(frames))

    # Verify report file
    report_path = os.path.join(extractor.output_dir, 'video_extraction_report.txt')
    assert os.path.exists(report_path), "Report file not created"

    with open(report_path, 'r') as f:
        report_content = f.read()

    assert 'VIDEO PLAYER & FRAME EXTRACTOR' in report_content, "Report header missing"
    assert 'EXTRACTION PARAMETERS' in report_content, "Parameters section missing"
    assert 'DATASET INFORMATION' in report_content, "Dataset info section missing"

    print("\n✓ Test 7 PASSED: Report generated successfully")
    print(f"  - Report size: {os.path.getsize(report_path)} bytes")


def test_frame_numbering_scheme():
    """Test that extracted frames use correct numbering scheme."""
    print("\n" + "="*70)
    print("TEST 8: Frame Numbering Scheme")
    print("="*70)

    extractor = VideoPlayerExtractor('./test_output_video_numbering')
    frames = extractor.extract_frames(frame_interval=10, max_frames=None)

    # Get extracted frame files
    frame_files = sorted([f for f in os.listdir(extractor.output_dir) if f.startswith('frame_')])

    assert len(frame_files) == len(frames), "Mismatch in frame count"

    # Verify naming scheme
    for idx, filename in enumerate(frame_files):
        expected_name = f'frame_{idx:06d}.png'
        assert filename == expected_name, f"Expected {expected_name}, got {filename}"

    print("\n✓ Test 8 PASSED: Frame numbering scheme is correct")
    print(f"  - Total frames: {len(frame_files)}")
    print(f"  - First frame: {frame_files[0]}")
    print(f"  - Last frame: {frame_files[-1]}")


def cleanup_test_outputs():
    """Remove test output directories."""
    test_dirs = [
        './test_output_video_gen',
        './test_output_video_props',
        './test_output_video_extract',
        './test_output_video_intervals',
        './test_output_video_max',
        './test_output_video_dashboard',
        './test_output_video_report',
        './test_output_video_numbering'
    ]

    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("VIDEO PLAYER & FRAME EXTRACTOR - TEST SUITE")
    print("="*70)

    try:
        test_synthetic_video_generation()
        test_video_properties_extraction()
        test_frame_extraction()
        test_frame_interval_variations()
        test_max_frames_limit()
        test_extraction_dashboard_creation()
        test_report_generation()
        test_frame_numbering_scheme()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")

    finally:
        cleanup_test_outputs()
