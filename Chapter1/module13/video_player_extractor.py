import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime


class VideoPlayerExtractor:
    """
    Video player and frame extractor utility for video processing and dataset creation.

    This utility provides dual functionality: play videos at native timing and extract
    frames at periodic intervals for dataset compilation. Supports comprehensive video
    analysis, visualization, and batch frame extraction.
    """

    def __init__(self, output_dir='./output_video_frames'):
        """
        Initialize the VideoPlayerExtractor with output directory configuration.

        Args:
            output_dir (str): Directory to save extracted frames and analysis outputs.
                Default is './output_video_frames'.

        Returns:
            None

        Notes:
            - Creates output directory if it doesn't exist
            - All output files are saved relative to script location, not working directory
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, os.path.basename(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_test_video(self, duration_seconds=10, fps=30, width=640, height=480):
        """
        Generate a synthetic test video with moving shapes and text overlays.

        Args:
            duration_seconds (int): Duration of video in seconds. Default is 10.
            fps (int): Frames per second for video. Default is 30.
            width (int): Video frame width in pixels. Default is 640.
            height (int): Video frame height in pixels. Default is 480.

        Returns:
            str: Path to generated test video file.

        Notes:
            - Creates synthetic video with animated shapes and frame counter
            - Video saved as MP4 with H.264 codec
            - Each frame shows unique content for testing extraction
            - Useful for demonstrating frame extraction without external video files
        """
        video_path = os.path.join(self.output_dir, 'test_video.mp4')
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        total_frames = duration_seconds * fps

        for frame_idx in range(total_frames):
            # Create frame with gradient background
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Create gradient
            for i in range(height):
                frame[i, :] = (
                    int(50 + (i / height) * 100),
                    int(100 + (i / height) * 80),
                    int(150 + (i / height) * 50)
                )

            # Draw animated circle
            circle_x = int(100 + 200 * np.sin(frame_idx * 2 * np.pi / total_frames))
            circle_y = int(height // 2 + 100 * np.cos(frame_idx * 2 * np.pi / total_frames))
            cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 0), -1)

            # Draw animated rectangle
            rect_x = int(100 + frame_idx * (width - 200) / total_frames)
            cv2.rectangle(frame, (rect_x, 100), (rect_x + 80, 200), (255, 100, 0), -1)

            # Add frame counter
            cv2.putText(frame, f'Frame: {frame_idx + 1}/{total_frames}',
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Add timestamp
            timestamp = frame_idx / fps
            cv2.putText(frame, f'Time: {timestamp:.2f}s',
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        return video_path

    def load_video(self, video_path=None, use_webcam=False):
        """
        Load video from file, webcam, or generate synthetic test video.

        Args:
            video_path (str): Path to video file. If None, generates synthetic video.
            use_webcam (bool): If True, use webcam instead of video file. Default is False.

        Returns:
            tuple: (cap, video_path) where:
                - cap: cv2.VideoCapture object for video
                - video_path: Path to loaded video file or 'webcam'

        Notes:
            - Supports MP4, AVI, MOV, MKV, FLV and other formats
            - Returns VideoCapture object ready for frame reading
            - Automatically generates synthetic video if file not found
            - Webcam mode uses default camera (device 0)
        """
        if use_webcam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Warning: Webcam not available. Using synthetic video.")
                video_path = self.generate_test_video()
                cap = cv2.VideoCapture(video_path)
            else:
                video_path = 'webcam'
        elif video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.generate_test_video())
        else:
            video_path = self.generate_test_video()
            cap = cv2.VideoCapture(video_path)

        return cap, video_path

    def get_video_properties(self, cap):
        """
        Extract video properties and metadata from VideoCapture object.

        Args:
            cap (cv2.VideoCapture): Video capture object.

        Returns:
            dict: Dictionary containing video properties:
                - fps: Frames per second
                - frame_count: Total number of frames
                - width: Frame width in pixels
                - height: Frame height in pixels
                - duration_seconds: Video duration in seconds

        Notes:
            - Extracts properties using cv2.CAP_PROP_* constants
            - Frame count may be 0 for some codecs (reads until end required)
            - FPS must be > 0 for meaningful frame interval calculations
        """
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_seconds = frame_count / fps if fps > 0 else 0

        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_seconds': duration_seconds
        }

    def play_video(self, video_path=None, max_duration=None, use_webcam=False):
        """
        Play video at native timing with frame information overlay.

        Args:
            video_path (str): Path to video file. If None, generates synthetic video.
            max_duration (float): Maximum playback duration in seconds. If None, plays full video.
            use_webcam (bool): If True, capture from webcam instead of file. Default is False.

        Returns:
            dict: Video properties from playback session.

        Notes:
            - Plays video at native FPS for realistic timing
            - Displays frame number, timestamp, and total duration
            - Press 'q' to quit, 'p' to pause/resume
            - Each frame shows native timing and frame counter
            - Window title shows current FPS
            - For webcam, max_duration controls recording length
        """
        cap, actual_video_path = self.load_video(video_path, use_webcam=use_webcam)
        props = self.get_video_properties(cap)

        fps = props['fps']
        if fps == 0:
            fps = 30  # Default for webcam or unknown sources
        frame_delay = int(1000 / fps) if fps > 0 else 30
        frame_idx = 0
        paused = False
        start_time = time.time()
        pause_time = 0
        pause_start = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate elapsed time
                elapsed_time = time.time() - start_time - pause_time
                frame_time = frame_idx / fps if fps > 0 else 0

                # Check max duration
                if max_duration and elapsed_time > max_duration:
                    break

                # Add overlay information
                display_frame = frame.copy()
                frame_text = f'Frame: {frame_idx + 1}'
                if props['frame_count'] > 0:
                    frame_text += f'/{props["frame_count"]}'

                cv2.putText(display_frame, frame_text,
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(display_frame, f'Time: {frame_time:.2f}s',
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(display_frame, f'FPS: {fps:.1f}',
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(display_frame, 'Press q to quit, p to pause',
                           (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                window_title = 'Webcam' if use_webcam else f'Video Player - {os.path.basename(actual_video_path)}'
                cv2.imshow(window_title, display_frame)

                frame_idx += 1

            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                if paused:
                    pause_start = time.time()
                else:
                    pause_time += time.time() - pause_start

        cv2.destroyAllWindows()
        cap.release()

        return props

    def extract_frames(self, video_path=None, frame_interval=30, max_frames=None, use_webcam=False,
                       duration_seconds=None):
        """
        Extract frames from video at specified periodic intervals.

        Args:
            video_path (str): Path to video file. If None, generates synthetic video.
            frame_interval (int): Extract every nth frame. Default is 30 (every 30th frame).
            max_frames (int): Maximum number of frames to extract. If None, extracts all.
            use_webcam (bool): If True, capture from webcam. Default is False.
            duration_seconds (float): For webcam, capture duration in seconds. If None, uses max_frames.

        Returns:
            list: List of extracted frame image arrays.

        Notes:
            - frame_interval=1 extracts every frame (all frames)
            - frame_interval=30 extracts every 30th frame
            - Saves extracted frames as numbered PNG files
            - Returns list of numpy arrays for programmatic access
            - For webcam: duration_seconds or max_frames controls capture length
            - Useful for creating training datasets with sparse sampling
        """
        cap, _ = self.load_video(video_path, use_webcam=use_webcam)

        extracted_frames = []
        frame_count = 0
        extracted_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check duration limit for webcam
            if use_webcam and duration_seconds:
                elapsed = time.time() - start_time
                if elapsed > duration_seconds:
                    break

            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                if max_frames and extracted_count >= max_frames:
                    break

                # Save frame
                frame_filename = f'frame_{extracted_count:06d}.png'
                frame_path = os.path.join(self.output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                # Store in list
                extracted_frames.append(frame)
                extracted_count += 1

            frame_count += 1

        cap.release()

        return extracted_frames

    def save_frame(self, frame, filename):
        """
        Save a single frame to output directory.

        Args:
            frame (np.ndarray): Frame image in BGR format.
            filename (str): Filename for saved frame (e.g., 'frame_001.png').

        Returns:
            str: Full path to saved frame file.

        Notes:
            - Saves as PNG format for lossless compression
            - Filename should include extension (e.g., '.png', '.jpg')
            - Output path is in configured output directory
        """
        frame_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(frame_path, frame)
        return frame_path

    def create_extraction_dashboard(self, extracted_frames, frame_interval, fps):
        """
        Create 4×4 grid visualization of extracted frames.

        Args:
            extracted_frames (list): List of extracted frame arrays.
            frame_interval (int): Interval at which frames were extracted.
            fps (float): Video frames per second.

        Returns:
            None

        Notes:
            - Creates 4×4 grid showing sample of extracted frames
            - Displays frame number and timestamp for each
            - Saves as high-DPI PNG (150 DPI)
            - Shows up to 16 frames in grid layout
        """
        num_frames = min(len(extracted_frames), 16)
        grid_size = int(np.ceil(np.sqrt(num_frames)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
        if grid_size == 1:
            axes = np.array([[axes]])
        else:
            axes = axes.flatten() if grid_size > 1 else np.array([axes])

        fig.suptitle(f'Extracted Frames (every {frame_interval}th frame)',
                    fontsize=16, fontweight='bold')

        for idx, frame in enumerate(extracted_frames[:num_frames]):
            ax = axes[idx] if isinstance(axes, np.ndarray) else axes
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_number = idx * frame_interval
            timestamp = frame_number / fps if fps > 0 else 0
            ax.set_title(f'Frame {frame_number} ({timestamp:.2f}s)')
            ax.axis('off')

        # Hide remaining subplots
        for idx in range(num_frames, len(axes.flatten())):
            axes.flatten()[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'extraction_dashboard.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def print_statistics(self, props, frame_interval, extracted_count):
        """
        Print video analysis statistics to console.

        Args:
            props (dict): Video properties dictionary from get_video_properties().
            frame_interval (int): Frame extraction interval.
            extracted_count (int): Number of frames extracted.

        Returns:
            None

        Notes:
            - Displays video resolution, duration, and FPS
            - Shows extraction parameters and results
            - Prints dataset size information
        """
        print("\n" + "=" * 70)
        print("VIDEO PLAYER & FRAME EXTRACTOR - STATISTICS")
        print("=" * 70)
        print(f"Video Resolution: {props['width']}×{props['height']} pixels")
        print(f"Total Frames: {props['frame_count']}")
        print(f"Frames Per Second: {props['fps']:.2f}")
        print(f"Duration: {props['duration_seconds']:.2f} seconds")
        print(f"\nExtraction Parameters:")
        print(f"  Frame Interval: {frame_interval}")
        print(f"  Extracted Frames: {extracted_count}")
        print(f"  Dataset Size: {extracted_count * props['width'] * props['height'] * 3 / (1024**2):.2f} MB")
        print("=" * 70 + "\n")

    def save_report(self, props, frame_interval, extracted_count):
        """
        Save detailed analysis report to text file.

        Args:
            props (dict): Video properties dictionary.
            frame_interval (int): Frame extraction interval.
            extracted_count (int): Number of frames extracted.

        Returns:
            None

        Notes:
            - Creates structured text report in output directory
            - Includes timestamp, video properties, and extraction details
            - Documents frame numbering scheme
            - Suitable for archival and dataset documentation
        """
        report_path = os.path.join(self.output_dir, 'video_extraction_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("VIDEO PLAYER & FRAME EXTRACTOR - ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("VIDEO PROPERTIES:\n")
            f.write(f"  Resolution: {props['width']}×{props['height']} pixels\n")
            f.write(f"  Total Frames: {props['frame_count']}\n")
            f.write(f"  Frames Per Second: {props['fps']:.2f}\n")
            f.write(f"  Duration: {props['duration_seconds']:.2f} seconds\n\n")

            f.write("EXTRACTION PARAMETERS:\n")
            f.write(f"  Frame Interval: {frame_interval} (every {frame_interval}th frame)\n")
            f.write(f"  Total Extracted: {extracted_count} frames\n")
            f.write(f"  Extraction Ratio: 1/{frame_interval}\n\n")

            f.write("DATASET INFORMATION:\n")
            f.write(f"  Frame Size: {props['width']}×{props['height']}×3 pixels\n")
            f.write(f"  Frame File Size: ~{props['width'] * props['height'] * 3 / (1024**2):.2f} MB (PNG)\n")
            f.write(f"  Total Dataset Size: ~{extracted_count * props['width'] * props['height'] * 3 / (1024**2):.2f} MB\n\n")

            f.write("OUTPUT FILES:\n")
            f.write("  - frame_000000.png to frame_xxxxxx.png: Extracted frames\n")
            f.write("  - extraction_dashboard.png: 4×4 grid visualization\n")
            f.write("  - video_extraction_report.txt: This report\n\n")

            f.write("FRAME NUMBERING SCHEME:\n")
            f.write("  - Files named: frame_XXXXXX.png (6-digit zero-padded)\n")
            f.write("  - Frame 0 corresponds to video frame: 0\n")
            f.write(f"  - Frame 1 corresponds to video frame: {frame_interval}\n")
            f.write(f"  - Frame N corresponds to video frame: N × {frame_interval}\n\n")

            f.write("KEY CONCEPTS:\n")
            f.write("  - Native Timing: Video plays at original FPS\n")
            f.write("  - Frame Extraction: Periodic sampling for dataset creation\n")
            f.write("  - Dataset Compilation: Selected frames form training dataset\n")
            f.write("=" * 70 + "\n")

    def run_analysis(self, video_path=None, frame_interval=30, play_video=True,
                     extract_frames_flag=True, max_frames=None, use_webcam=False,
                     webcam_duration=5):
        """
        Execute complete video analysis pipeline with playback and frame extraction.

        Args:
            video_path (str): Path to video file. If None, generates synthetic video.
            frame_interval (int): Extract every nth frame. Default is 30.
            play_video (bool): Play video during analysis. Default is True.
            extract_frames_flag (bool): Extract and save frames. Default is True.
            max_frames (int): Maximum frames to extract. If None, extracts all.
            use_webcam (bool): If True, use webcam instead of video file. Default is False.
            webcam_duration (float): Duration for webcam capture in seconds. Default is 5.

        Returns:
            tuple: (extracted_frames, props) where:
                - extracted_frames: List of extracted frame arrays
                - props: Dictionary of video properties

        Notes:
            - Loads or generates test video (or captures from webcam)
            - Optionally plays video with frame overlay
            - Optionally extracts frames at specified intervals
            - Saves all extracted frames as PNG files
            - Creates visualization dashboard
            - Generates detailed analysis report
            - For webcam: automatically extracts frames during playback
        """
        # Load video
        cap, actual_video_path = self.load_video(video_path, use_webcam=use_webcam)
        properties = self.get_video_properties(cap)
        cap.release()

        # Play video if requested
        if play_video:
            if use_webcam:
                print(f"\nCapturing from webcam for {webcam_duration} seconds...")
                print("(press 'q' to quit, 'p' to pause)")
            else:
                print("\nPlaying video... (press 'q' to quit, 'p' to pause)")
            self.play_video(actual_video_path, max_duration=webcam_duration if use_webcam else None,
                          use_webcam=use_webcam)

        # Extract frames if requested
        extracted_frames = []
        if extract_frames_flag:
            extracted_frames = self.extract_frames(
                actual_video_path, frame_interval, max_frames,
                use_webcam=use_webcam,
                duration_seconds=webcam_duration if use_webcam else None
            )

        # Create visualization
        if extracted_frames:
            self.create_extraction_dashboard(extracted_frames, frame_interval, properties['fps'])

        # Print statistics
        self.print_statistics(properties, frame_interval, len(extracted_frames))

        # Save report
        self.save_report(properties, frame_interval, len(extracted_frames))

        return extracted_frames, properties


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Video Player & Frame Extractor - Play videos and extract frames for datasets'
    )
    parser.add_argument(
        '--video', type=str, default=None,
        help='Path to video file (MP4, AVI, MOV, MKV, FLV)'
    )
    parser.add_argument(
        '--webcam', action='store_true', default=False,
        help='Use webcam instead of video file'
    )
    parser.add_argument(
        '--webcam-duration', type=float, default=5,
        help='Duration for webcam capture in seconds (default: 5)'
    )
    parser.add_argument(
        '--frame-interval', type=int, default=30,
        help='Extract every nth frame (default: 30). Use 1 for all frames'
    )
    parser.add_argument(
        '--play', action='store_true', default=True,
        help='Play video during processing (default: True)'
    )
    parser.add_argument(
        '--no-play', dest='play', action='store_false',
        help='Skip video playback'
    )
    parser.add_argument(
        '--extract', action='store_true', default=True,
        help='Extract frames (default: True)'
    )
    parser.add_argument(
        '--no-extract', dest='extract', action='store_false',
        help='Skip frame extraction'
    )
    parser.add_argument(
        '--max-frames', type=int, default=None,
        help='Maximum frames to extract (if None, extracts all)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for extracted frames'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.webcam and args.video:
        print("Error: Cannot use both --webcam and --video. Choose one.")
        return

    if args.output_dir:
        extractor = VideoPlayerExtractor(args.output_dir)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        extractor = VideoPlayerExtractor(os.path.join(current_dir, 'output_video_frames'))

    extractor.run_analysis(
        video_path=args.video,
        frame_interval=args.frame_interval,
        play_video=args.play,
        extract_frames_flag=args.extract,
        max_frames=args.max_frames,
        use_webcam=args.webcam,
        webcam_duration=args.webcam_duration
    )


if __name__ == '__main__':
    main()
