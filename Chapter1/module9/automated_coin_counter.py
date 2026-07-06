import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt


class AutomatedCoinCounter:
    """
    Detect and count coins in images using Otsu's thresholding and contour analysis.

    This class implements a complete coin detection pipeline that processes images
    through preprocessing, thresholding, morphological operations, and contour
    analysis. It detects individual coins, calculates their spatial coordinates
    (centroids), and generates comprehensive visualizations and reports.

    The detection pipeline steps:
    1. Load image or generate synthetic test data
    2. Convert to grayscale and apply Gaussian blur
    3. Apply Otsu's binary thresholding (inverted)
    4. Morphological closing to fill interior holes
    5. Contour detection and area-based filtering
    6. Centroid calculation using image moments
    7. Visualization with circles and coordinate labels
    """

    def __init__(self, output_dir='./output_coins'):
        """
        Initialize counter with output directory for results.

        Args:
            output_dir (str): Directory where all output files will be saved.
                Default is './output_coins'. Directory is created if it doesn't exist.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_or_generate_image(self, image_path=None, width=800, height=600):
        """
        Load image from file or generate synthetic coin image for testing.

        If a valid image path is provided, loads that image. Otherwise generates
        a synthetic image with realistic coins including metallic highlights and
        shadows. Useful for testing and demonstration purposes.

        Args:
            image_path (str, optional): Path to custom coin image file. Supports
                JPG, PNG, BMP, TIFF formats. Defaults to None (generates synthetic).
            width (int): Width of synthetic image in pixels. Default is 800.
            height (int): Height of synthetic image in pixels. Default is 600.

        Returns:
            ndarray: BGR color image in OpenCV format (height, width, 3).

        Raises:
            ValueError: If image_path is provided but file cannot be read.
        """
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unable to load image from {image_path}")
            return img
        return self.generate_coin_image(width, height)

    def generate_coin_image(self, width=800, height=600):
        """
        Generate synthetic coin image with realistic metallic effects and shadows.

        Creates a test image containing randomly positioned coins with:
        - Realistic metallic colors (gold, silver, bronze)
        - Bright highlights simulating coin shine
        - Dark shadows underneath each coin
        - Variable sizes between 25-50 pixel radius

        Useful for testing detection algorithm without requiring real coin photos.
        Generated image is saved to output directory as 'generated_coins.png'.

        Args:
            width (int): Image width in pixels. Default is 800.
            height (int): Image height in pixels. Default is 600.

        Returns:
            ndarray: BGR color image containing synthetic coins.

        Notes:
            - Uses fixed random seed (42) for reproducibility
            - Generates 5-12 coins per image
            - Coins placed with 80px margin from edges
        """
        img = np.ones((height, width, 3), dtype=np.uint8) * 180

        np.random.seed(42)
        num_coins = np.random.randint(5, 12)

        for _ in range(num_coins):
            x = np.random.randint(80, width - 80)
            y = np.random.randint(80, height - 80)
            radius = np.random.randint(25, 50)

            color = (np.random.randint(50, 100), np.random.randint(100, 150),
                    np.random.randint(100, 180))
            cv2.circle(img, (x, y), radius, color, -1)

            highlight_x = x - radius // 3
            highlight_y = y - radius // 3
            cv2.circle(img, (highlight_x, highlight_y), radius // 4,
                      (255, 255, 200), -1)

            shadow_offset = int(radius * 0.7)
            cv2.ellipse(img, (x + 5, y + 10), (shadow_offset, shadow_offset // 3),
                       0, 0, 180, (0, 0, 0), -1)

        cv2.imwrite(os.path.join(self.output_dir, 'generated_coins.png'), img)
        return img

    def preprocess_image(self, img):
        """
        Convert to grayscale and apply Gaussian blur for noise reduction.

        Preprocessing step that prepares the image for thresholding by:
        1. Converting BGR color image to single-channel grayscale
        2. Applying 7×7 Gaussian blur to reduce noise and smooth metallic textures

        The blur helps reduce high-frequency noise from metallic coin surfaces
        and lighting artifacts while preserving coin boundaries.

        Args:
            img (ndarray): BGR color image from OpenCV.

        Returns:
            tuple: Two-element tuple containing:
                - gray (ndarray): Grayscale image
                - blurred (ndarray): Gaussian-blurred grayscale image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        return gray, blurred

    def threshold_image(self, blurred):
        """
        Apply Otsu's binary inversion thresholding to isolate coins as white silhouettes.

        Uses Otsu's method to automatically determine the optimal binary threshold,
        then inverts the result so coins (dark objects) become white (255) and
        background becomes black (0). This creates a clean binary mask where
        coins are easily detected as contours.

        Otsu's method minimizes within-class variance and works well for bimodal
        histograms (two distinct peaks), which is ideal for coin vs. background.

        Args:
            blurred (ndarray): Blurred grayscale image.

        Returns:
            ndarray: Binary image (values 0 or 255) with coins as white silhouettes.

        Notes:
            - Uses cv2.THRESH_BINARY_INV for inversion
            - Uses cv2.THRESH_OTSU for automatic threshold calculation
            - Ignores manual threshold parameter (0) when OTSU is specified
        """
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def apply_morphology(self, binary, kernel_size=9):
        """
        Apply morphological closing to fill interior holes and shadow voids.

        Morphological closing operation (dilation followed by erosion) fills small
        holes inside coins (from engravings or shadows) while preserving overall
        coin shape. Uses an elliptical structuring element which provides smooth
        circular morphology suitable for coin-shaped objects.

        This step is important for coins that have interior details (engravings,
        date, etc.) that create holes in the binary image after thresholding.

        Args:
            binary (ndarray): Binary image from Otsu's thresholding.
            kernel_size (int): Size of structuring element kernel. Must be odd number.
                Default is 9 (9×9 kernel). Larger kernels fill larger holes.

        Returns:
            ndarray: Binary image after morphological closing.

        Notes:
            - Uses cv2.MORPH_ELLIPSE for smooth circular operations
            - kernel_size should be odd (automatically handled by user parameters)
            - Larger kernel sizes create stronger smoothing effects
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return closed

    def find_coins(self, binary, min_area=500):
        """
        Detect coin contours, filter by area threshold, and calculate centroids using moments.

        Finds all contours in the binary image using external contour detection
        (ignores internal holes). Filters out small contours (noise) using minimum
        area threshold. For each valid coin contour, calculates:
        - Centroid position using image moments (M10/M00, M01/M00)
        - Bounding circle using minimum enclosing circle algorithm
        - Area and radius metrics for size analysis

        Args:
            binary (ndarray): Binary image from morphological closing.
            min_area (int): Minimum area threshold in pixels². Contours with area
                less than this are filtered out as noise. Default is 500px².

        Returns:
            list: List of coin dictionaries, each containing:
                - 'contour': Numpy array of contour points
                - 'area': Contour area in pixels²
                - 'centroid': (cx, cy) tuple of centroid coordinates
                - 'radius': Radius of minimum enclosing circle in pixels
                - 'circle_center': (x, y) center of minimum enclosing circle

        Notes:
            - Uses cv2.RETR_EXTERNAL to find only external contours
            - Centroid calculation uses first and zeroth image moments
            - Minimum enclosing circle provides more robust radius than contour bounds
            - Coins must have area > min_area to be detected
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        coins = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    radius = int(radius)

                    coins.append({
                        'contour': contour,
                        'area': area,
                        'centroid': (cx, cy),
                        'radius': radius,
                        'circle_center': (int(x), int(y))
                    })

        return coins

    def visualize_coins(self, img, coins, title_suffix=''):
        """
        Draw circles around detected coins with centroids and coordinate labels.

        Creates visualization by overlaying detection results on original image:
        - Green circles show coin boundaries (minimum enclosing circle)
        - Red filled circles mark centroid positions
        - Blue text labels show (x, y) coordinates for each coin
        - Green text shows total coin count at top-left

        This annotated image provides immediate visual feedback on detection
        accuracy and coin positions. Saved to output directory.

        Args:
            img (ndarray): Original BGR image.
            coins (list): List of coin dictionaries from find_coins().
            title_suffix (str): Suffix to append to output filename. Default is ''.

        Returns:
            ndarray: Annotated BGR image with coin circles and labels.

        Notes:
            - Saves result as 'coin_detection{title_suffix}.png' to output_dir
            - Green (0, 255, 0): Coin boundary circles
            - Red (0, 0, 255): Centroid positions
            - Blue (255, 0, 0): Coordinate text labels
        """
        result = img.copy()

        for coin in coins:
            cx, cy = coin['centroid']
            radius = coin['radius']

            cv2.circle(result, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(result, f"({cx}, {cy})", (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        coin_count = len(coins)
        cv2.putText(result, f"Coins Detected: {coin_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        output_path = os.path.join(self.output_dir, f'coin_detection{title_suffix}.png')
        cv2.imwrite(output_path, result)

        return result

    def create_analysis_dashboard(self, img, binary, closed, result, coins):
        """
        Generate 4-panel visualization showing the complete detection pipeline.

        Creates a comprehensive matplotlib figure displaying all major pipeline stages:
        - Panel 1 (top-left): Original BGR image
        - Panel 2 (top-right): Binary threshold result from Otsu's method
        - Panel 3 (bottom-left): After morphological closing
        - Panel 4 (bottom-right): Final detection with circles and labels

        This dashboard provides visual verification that each pipeline stage is
        working correctly. Useful for debugging and understanding the detection
        process. Saved at 150 DPI for publication-quality output.

        Args:
            img (ndarray): Original BGR image.
            binary (ndarray): Binary threshold image.
            closed (ndarray): After morphological closing.
            result (ndarray): Final detection result with visualizations.
            coins (list): List of detected coin dictionaries.

        Notes:
            - Saves as 'coin_analysis_dashboard.png' to output_dir
            - 2×2 subplot layout with 14×10 inch figure size
            - 150 DPI resolution for high quality
            - Uses grayscale colormap for binary images
            - Displays coin count in bottom-right panel title
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Coin Counter: Complete Analysis Pipeline', fontsize=16, fontweight='bold')

        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title(f'Otsu Threshold (Binary Inverted)')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(closed, cmap='gray')
        axes[1, 0].set_title('After Morphological Closing')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Detected Coins: {len(coins)}')
        axes[1, 1].axis('off')

        plt.tight_layout()
        dashboard_path = os.path.join(self.output_dir, 'coin_analysis_dashboard.png')
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()

    def print_statistics(self, coins, img_shape):
        """
        Print area and radius statistics with coin centroid locations to console.

        Outputs a formatted report to console displaying:
        - Image resolution
        - Total number of coins detected
        - Area statistics (mean, min, max)
        - Radius statistics (mean, min, max)
        - Complete list of all coins with coordinates and metrics

        Useful for quick verification of detection results without needing to
        open output files. Statistics help identify if detection parameters need
        adjustment (e.g., if only very small or very large coins detected).

        Args:
            coins (list): List of coin dictionaries from find_coins().
            img_shape (tuple): Image shape from numpy array (height, width, channels).

        Prints:
            - Formatted header with coin count
            - Area statistics in pixels²
            - Radius statistics in pixels
            - Per-coin centroid coordinates and metrics
            - Output directory path
        """
        print("\n" + "="*70)
        print("AUTOMATED COIN COUNTER - ANALYSIS REPORT")
        print("="*70)
        print(f"Image Resolution: {img_shape[1]}x{img_shape[0]} pixels")
        print(f"Total Coins Detected: {len(coins)}")
        print("-"*70)

        if coins:
            areas = [c['area'] for c in coins]
            radii = [c['radius'] for c in coins]

            print(f"\nCOIN STATISTICS:")
            print(f"  Average Area: {np.mean(areas):.2f} px²")
            print(f"  Min Area: {np.min(areas):.2f} px²")
            print(f"  Max Area: {np.max(areas):.2f} px²")
            print(f"\n  Average Radius: {np.mean(radii):.2f} px")
            print(f"  Min Radius: {np.min(radii):.2f} px")
            print(f"  Max Radius: {np.max(radii):.2f} px")

            print(f"\nCOIN CENTROIDS (Spatial Coordinates):")
            for i, coin in enumerate(coins, 1):
                cx, cy = coin['centroid']
                area = coin['area']
                radius = coin['radius']
                print(f"  Coin {i:2d}: ({cx:4d}, {cy:4d}) | "
                      f"Area: {area:6.0f} px² | Radius: {radius:3d} px")

        print("-"*70)
        print(f"Output Directory: {os.path.abspath(self.output_dir)}")
        print("="*70 + "\n")

    def save_report(self, coins, img_shape, processing_time):
        """
        Save detailed analysis report with statistics and coordinates to text file.

        Generates a comprehensive text report saved to 'coin_analysis_report.txt'
        containing all detection results and analysis metrics:
        - Timestamp of analysis
        - Image resolution and processing time
        - Area and radius statistics with standard deviation
        - Complete list of detected coins with coordinates and metrics
        - Coordinate bounds (min/max X and Y positions)

        Report is human-readable and suitable for documentation, logging, or
        batch processing pipelines. Can be imported into spreadsheets or databases.

        Args:
            coins (list): List of coin dictionaries from find_coins().
            img_shape (tuple): Image shape (height, width, channels).
            processing_time (float): Total processing time in seconds.

        Notes:
            - Saves to '{output_dir}/coin_analysis_report.txt'
            - Includes timestamp for traceability
            - Calculates standard deviation in addition to mean/min/max
            - Lists coordinate bounds for spatial analysis
            - Fixed-width formatting for easy parsing
        """
        report_path = os.path.join(self.output_dir, 'coin_analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("AUTOMATED COIN COUNTER - DETAILED ANALYSIS REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image Resolution: {img_shape[1]}x{img_shape[0]} pixels\n")
            f.write(f"Processing Time: {processing_time:.2f} seconds\n")
            f.write(f"Total Coins Detected: {len(coins)}\n")
            f.write("-"*70 + "\n")

            if coins:
                areas = [c['area'] for c in coins]
                radii = [c['radius'] for c in coins]

                f.write("\nCOIN STATISTICS:\n")
                f.write(f"  Average Area: {np.mean(areas):.2f} px²\n")
                f.write(f"  Std Dev Area: {np.std(areas):.2f} px²\n")
                f.write(f"  Min Area: {np.min(areas):.2f} px²\n")
                f.write(f"  Max Area: {np.max(areas):.2f} px²\n")
                f.write(f"\n  Average Radius: {np.mean(radii):.2f} px\n")
                f.write(f"  Std Dev Radius: {np.std(radii):.2f} px\n")
                f.write(f"  Min Radius: {np.min(radii):.2f} px\n")
                f.write(f"  Max Radius: {np.max(radii):.2f} px\n")

                f.write(f"\nDETECTED COIN LOCATIONS:\n")
                for i, coin in enumerate(coins, 1):
                    cx, cy = coin['centroid']
                    area = coin['area']
                    radius = coin['radius']
                    f.write(f"  Coin {i:2d}: Position=({cx:4d}, {cy:4d}) | "
                           f"Area={area:6.0f} px² | Radius={radius:3d} px\n")

                f.write(f"\nCOORDINATE BOUNDS:\n")
                x_coords = [c['centroid'][0] for c in coins]
                y_coords = [c['centroid'][1] for c in coins]
                f.write(f"  X Range: {min(x_coords)} - {max(x_coords)} px\n")
                f.write(f"  Y Range: {min(y_coords)} - {max(y_coords)} px\n")

            f.write("-"*70 + "\n")

    def run_analysis(self, image_path=None, min_area=500, kernel_size=9):
        """
        Execute complete coin detection pipeline and generate visualizations and reports.

        Orchestrates the entire detection workflow from image loading through
        reporting. Executes the following steps in sequence:
        1. Load image or generate synthetic test data
        2. Preprocess: convert to grayscale and blur
        3. Apply Otsu's binary thresholding
        4. Apply morphological closing
        5. Detect coins and calculate centroids
        6. Visualize results with annotations
        7. Create analysis dashboard
        8. Print console statistics
        9. Save detailed text report
        10. Save all intermediate images

        This is the main entry point for the coin detection pipeline.

        Args:
            image_path (str, optional): Path to custom image file. If None,
                generates synthetic coin image. Default is None.
            min_area (int): Minimum coin area threshold in pixels². Default is 500.
            kernel_size (int): Morphological kernel size (must be odd). Default is 9.

        Returns:
            tuple: Two-element tuple containing:
                - coins (list): List of detected coin dictionaries
                - result (ndarray): Annotated visualization image

        Notes:
            - Saves 7 output files to output_dir:
              * original_image.png
              * binary_threshold.png
              * morphological_closing.png
              * coin_detection.png
              * coin_analysis_dashboard.png
              * coin_analysis_report.txt
              * generated_coins.png (if synthetic)
            - Prints statistics to console
            - Processing time is calculated and included in report
        """
        import time
        start_time = time.time()

        img = self.load_or_generate_image(image_path)
        _, blurred = self.preprocess_image(img)
        binary = self.threshold_image(blurred)
        closed = self.apply_morphology(binary, kernel_size)
        coins = self.find_coins(closed, min_area)
        result = self.visualize_coins(img, coins)

        self.create_analysis_dashboard(img, binary, closed, result, coins)
        self.print_statistics(coins, img.shape)

        processing_time = time.time() - start_time
        self.save_report(coins, img.shape, processing_time)

        cv2.imwrite(os.path.join(self.output_dir, 'original_image.png'), img)
        cv2.imwrite(os.path.join(self.output_dir, 'binary_threshold.png'), binary)
        cv2.imwrite(os.path.join(self.output_dir, 'morphological_closing.png'), closed)

        return coins, result


def main():
    """
    Parse command-line arguments and run coin counter analysis.

    Entry point for command-line usage. Parses arguments and creates
    AutomatedCoinCounter instance to run analysis with specified parameters.

    Command-line arguments:
        --custom PATH: Path to custom coin image (JPG, PNG, BMP, TIFF)
        --min-area INT: Minimum coin area threshold in pixels (default: 500)
        --kernel-size INT: Morphology kernel size (default: 9)
        --output-dir PATH: Output directory for results (default: ./output_coins)

    Examples:
        Generate synthetic coins:
            python automated_coin_counter.py

        Analyze custom image:
            python automated_coin_counter.py --custom coins.jpg

        With parameters:
            python automated_coin_counter.py --custom coins.jpg \\
                --min-area 300 --kernel-size 7 --output-dir ./results

    Prints:
        - Console statistics from analysis
        - File paths where results are saved
    """
    parser = argparse.ArgumentParser(
        description='Automated Coin Counter - Detect and count coins in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate synthetic coin image and detect coins
  python automated_coin_counter.py

  # Use custom image with default settings
  python automated_coin_counter.py --custom coins.jpg

  # Use custom image with optimized parameters
  python automated_coin_counter.py --custom coins.jpg --min-area 300 --kernel-size 7

  # Save results to specific directory
  python automated_coin_counter.py --custom coins.jpg --output-dir ./results
        '''
    )

    parser.add_argument('--custom', type=str, default=None,
                       help='Path to custom coin image (JPG, PNG, etc.)')
    parser.add_argument('--min-area', type=int, default=500,
                       help='Minimum coin area threshold in pixels (default: 500)')
    parser.add_argument('--kernel-size', type=int, default=9,
                       help='Kernel size for morphological operations (default: 9)')
    parser.add_argument('--output-dir', type=str, default='./output_coins',
                       help='Output directory for results (default: ./output_coins)')

    args = parser.parse_args()

    counter = AutomatedCoinCounter(args.output_dir)
    counter.run_analysis(args.custom, args.min_area, args.kernel_size)


if __name__ == '__main__':
    main()
