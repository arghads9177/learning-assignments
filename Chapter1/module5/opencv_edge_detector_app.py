"""
OpenCV Edge Detector Application - Interactive Hands-on Exercises

This script demonstrates an interactive application for evaluating edge detection
techniques with real-time parameter adjustment using OpenCV trackbars:
- Canny edge detection with adjustable thresholds
- Sobel edge detection with adjustable kernel size
- Laplacian edge detection (fixed kernel)
- Scharr edge detection with adjustable kernel size
- Real-time comparison of all methods
"""

import cv2
import numpy as np
import os


class EdgeDetectorApp:
    """
    Interactive edge detection application with real-time parameter adjustment.

    Provides trackbars to adjust Canny thresholds and Sobel/Scharr kernel sizes.
    Displays edge detection results in real-time as parameters change.
    """

    def __init__(self, image_path):
        """
        Initialize the edge detector application.

        Args:
            image_path (str): Path to the image file to process.

        Raises:
            ValueError: If image cannot be loaded.
        """
        self.image_path = image_path
        self.original = None
        self.gray = None
        self.output_dir = os.path.join(os.path.dirname(image_path), 'output_edge_detection')

        # Load image
        self._load_image()

        # Trackbar parameters
        self.canny_low = 50
        self.canny_high = 150
        self.sobel_ksize = 3

        # Edge detection results
        self.edges_canny = None
        self.edges_sobel = None
        self.edges_laplacian = None
        self.edges_scharr = None

    def _load_image(self):
        """
        Load image from file. Create sample if doesn't exist.

        Prints:
            - Confirmation of image loading
            - Image dimensions
        """
        if os.path.exists(self.image_path):
            self.original = cv2.imread(self.image_path)
        else:
            self._create_sample_image()
            self.original = cv2.imread(self.image_path)

        if self.original is None:
            raise ValueError(f"Failed to load image from: {self.image_path}")

        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        print(f"✓ Image loaded: {self.image_path}")
        print(f"  Dimensions: {self.original.shape}")

    def _create_sample_image(self):
        """
        Create a sample image with various edge patterns.

        Generates an image with geometric shapes, gradients, and text
        to provide diverse edge detection scenarios.
        """
        os.makedirs(os.path.dirname(self.image_path) if os.path.dirname(self.image_path) else '.', exist_ok=True)

        # Create blank image with white background
        img = np.ones((500, 700, 3), dtype=np.uint8) * 200

        # Add rectangles with different line thicknesses
        cv2.rectangle(img, (50, 50), (200, 150), (0, 255, 0), 3)  # Green rectangle
        cv2.rectangle(img, (250, 50), (400, 150), (255, 0, 0), 2)  # Blue rectangle
        cv2.rectangle(img, (450, 50), (600, 150), (0, 0, 255), 1)  # Red rectangle

        # Add circles with different thicknesses
        cv2.circle(img, (125, 250), 50, (0, 165, 255), 3)  # Orange circle
        cv2.circle(img, (325, 250), 50, (255, 255, 0), 2)  # Cyan circle
        cv2.circle(img, (525, 250), 50, (255, 0, 255), 1)  # Magenta circle

        # Add filled shapes
        cv2.ellipse(img, (125, 380), (60, 40), 0, 0, 360, (100, 200, 100), -1)
        triangle = np.array([(325, 320), (375, 380), (275, 380)], dtype=np.int32)
        cv2.drawContours(img, [triangle], 0, (150, 100, 200), -1)

        # Add gradient region
        for x in range(450, 650):
            intensity = int(255 * (x - 450) / 200)
            img[300:450, x] = intensity

        # Add text
        cv2.putText(img, "Edge Detection Test", (100, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)

        cv2.imwrite(self.image_path, img)
        print(f"✓ Created sample image: {self.image_path}")

    def compute_edges(self):
        """
        Compute edge detection using all methods with current parameters.

        Computes:
        - Canny edges with current threshold values
        - Sobel edges with current kernel size
        - Laplacian edges (fixed kernel)
        - Scharr edges with current kernel size

        Updates instance variables with results.
        """
        # Canny edge detection
        self.edges_canny = cv2.Canny(self.gray, self.canny_low, self.canny_high)

        # Sobel edge detection (X and Y)
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        self.edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        self.edges_sobel = np.clip(self.edges_sobel, 0, 255).astype(np.uint8)

        # Laplacian edge detection
        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F, ksize=3)
        self.edges_laplacian = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)

        # Scharr edge detection
        scharrx = cv2.Scharr(self.gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(self.gray, cv2.CV_64F, 0, 1)
        self.edges_scharr = np.sqrt(scharrx**2 + scharry**2)
        self.edges_scharr = np.clip(self.edges_scharr, 0, 255).astype(np.uint8)

    def on_trackbar_canny_low(self, value):
        """
        Trackbar callback for Canny low threshold.

        Args:
            value (int): Current trackbar value.
        """
        self.canny_low = value
        if self.canny_low >= self.canny_high:
            self.canny_high = self.canny_low + 1

    def on_trackbar_canny_high(self, value):
        """
        Trackbar callback for Canny high threshold.

        Args:
            value (int): Current trackbar value.
        """
        self.canny_high = value
        if self.canny_high <= self.canny_low:
            self.canny_low = self.canny_high - 1

    def on_trackbar_ksize(self, value):
        """
        Trackbar callback for Sobel/Scharr kernel size.

        Ensures kernel size is always odd and at least 3.

        Args:
            value (int): Current trackbar value.
        """
        # Keep kernel size odd (3, 5, 7, 9, ...)
        self.sobel_ksize = value * 2 + 1
        if self.sobel_ksize < 3:
            self.sobel_ksize = 3

    def display_with_trackbars(self):
        """
        Display edge detection with interactive trackbars.

        Creates windows with trackbars for:
        - Canny low threshold (0-300)
        - Canny high threshold (0-300)
        - Sobel/Scharr kernel size (1-5, maps to 3, 5, 7, 9, 11)

        Displays real-time edge detection results as parameters change.
        Press 'q' to quit, 's' to save results.
        """
        print("\n" + "="*70)
        print("INTERACTIVE EDGE DETECTION APPLICATION")
        print("="*70)

        # Create main window for display
        window_name = "Edge Detection - Press 'q' to quit, 's' to save"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 600)

        # Create trackbar window
        trackbar_window = "Parameter Control"
        cv2.namedWindow(trackbar_window, cv2.WINDOW_NORMAL)

        # Create trackbars
        cv2.createTrackbar("Canny Low", trackbar_window, self.canny_low, 300,
                          self.on_trackbar_canny_low)
        cv2.createTrackbar("Canny High", trackbar_window, self.canny_high, 300,
                          self.on_trackbar_canny_high)
        cv2.createTrackbar("Kernel Size", trackbar_window, 1, 5,
                          self.on_trackbar_ksize)

        print(f"\nControls:")
        print(f"  - Adjust trackbars to change parameters in real-time")
        print(f"  - 'q' key: Quit application")
        print(f"  - 's' key: Save results to {self.output_dir}")
        print(f"\nInitial Settings:")
        print(f"  - Canny Low Threshold: {self.canny_low}")
        print(f"  - Canny High Threshold: {self.canny_high}")
        print(f"  - Sobel/Scharr Kernel Size: {self.sobel_ksize}")

        # Main loop
        while True:
            # Compute edges with current parameters
            self.compute_edges()

            # Create comparison image
            comparison = self._create_comparison_image()

            # Display
            cv2.imshow(window_name, comparison)

            # Check for key press
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("\n✓ Exiting application...")
                break
            elif key == ord('s'):
                self._save_results()

        cv2.destroyAllWindows()

    def _create_comparison_image(self):
        """
        Create a 2x2 comparison grid of edge detection results.

        Returns:
            ndarray: Comparison image with 4 edge detection methods.
        """
        # Convert grayscale results to BGR for color display
        canny_bgr = cv2.cvtColor(self.edges_canny, cv2.COLOR_GRAY2BGR)
        sobel_bgr = cv2.cvtColor(self.edges_sobel, cv2.COLOR_GRAY2BGR)
        laplacian_bgr = cv2.cvtColor(self.edges_laplacian, cv2.COLOR_GRAY2BGR)
        scharr_bgr = cv2.cvtColor(self.edges_scharr, cv2.COLOR_GRAY2BGR)

        # Add text labels to each image
        h, w = canny_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)  # Green text

        cv2.putText(canny_bgr, f"Canny (T1:{self.canny_low}, T2:{self.canny_high})",
                   (10, 30), font, font_scale, color, thickness)
        cv2.putText(sobel_bgr, f"Sobel (K:{self.sobel_ksize})",
                   (10, 30), font, font_scale, color, thickness)
        cv2.putText(laplacian_bgr, "Laplacian (K:3)",
                   (10, 30), font, font_scale, color, thickness)
        cv2.putText(scharr_bgr, "Scharr (Fixed)",
                   (10, 30), font, font_scale, color, thickness)

        # Create 2x2 grid
        top_row = np.hstack([canny_bgr, sobel_bgr])
        bottom_row = np.hstack([laplacian_bgr, scharr_bgr])
        comparison = np.vstack([top_row, bottom_row])

        return comparison

    def _save_results(self):
        """
        Save edge detection results to files.

        Saves all 4 edge detection results with current parameters
        in the output directory.

        Prints:
            - Confirmation of saved files
            - Output directory path
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Save results
        canny_path = os.path.join(self.output_dir,
                                 f'canny_low{self.canny_low}_high{self.canny_high}.jpg')
        sobel_path = os.path.join(self.output_dir,
                                 f'sobel_ksize{self.sobel_ksize}.jpg')
        laplacian_path = os.path.join(self.output_dir, 'laplacian.jpg')
        scharr_path = os.path.join(self.output_dir, 'scharr.jpg')

        cv2.imwrite(canny_path, self.edges_canny)
        cv2.imwrite(sobel_path, self.edges_sobel)
        cv2.imwrite(laplacian_path, self.edges_laplacian)
        cv2.imwrite(scharr_path, self.edges_scharr)

        print(f"\n✓ Results saved:")
        print(f"  - Canny: {canny_path}")
        print(f"  - Sobel: {sobel_path}")
        print(f"  - Laplacian: {laplacian_path}")
        print(f"  - Scharr: {scharr_path}")

    def analyze_edges(self):
        """
        Perform statistical analysis of edge detection results.

        Analyzes edge detection quality metrics including:
        - Number of edge pixels detected
        - Edge intensity statistics
        - Comparison of detection methods

        Prints:
            - Statistics for each edge detection method
            - Comparative analysis
        """
        print("\n" + "="*70)
        print("EDGE DETECTION ANALYSIS")
        print("="*70)

        def analyze_method(name, edges):
            """Analyze a single edge detection result."""
            edge_pixels = np.count_nonzero(edges)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_percentage = (edge_pixels / total_pixels) * 100
            mean_intensity = edges.mean()
            std_intensity = edges.std()
            max_intensity = edges.max()

            print(f"\n{name}:")
            print(f"  Edge pixels detected: {edge_pixels} ({edge_percentage:.2f}%)")
            print(f"  Mean intensity: {mean_intensity:.2f}")
            print(f"  Std deviation: {std_intensity:.2f}")
            print(f"  Max intensity: {max_intensity}")

            return {
                'edge_pixels': edge_pixels,
                'edge_percentage': edge_percentage,
                'mean_intensity': mean_intensity
            }

        # Analyze each method
        print(f"\nCanny Detection (Low: {self.canny_low}, High: {self.canny_high}):")
        canny_stats = analyze_method("  Canny", self.edges_canny)

        print(f"\nSobel Detection (Kernel Size: {self.sobel_ksize}):")
        sobel_stats = analyze_method("  Sobel", self.edges_sobel)

        print(f"\nLaplacian Detection (Kernel Size: 3):")
        laplacian_stats = analyze_method("  Laplacian", self.edges_laplacian)

        print(f"\nScharr Detection (Fixed):")
        scharr_stats = analyze_method("  Scharr", self.edges_scharr)

        # Comparison
        print(f"\n" + "-"*70)
        print("COMPARATIVE ANALYSIS:")
        print("-"*70)

        methods = {
            'Canny': canny_stats,
            'Sobel': sobel_stats,
            'Laplacian': laplacian_stats,
            'Scharr': scharr_stats
        }

        sorted_by_edges = sorted(methods.items(),
                                key=lambda x: x[1]['edge_percentage'], reverse=True)
        sorted_by_intensity = sorted(methods.items(),
                                    key=lambda x: x[1]['mean_intensity'], reverse=True)

        print("\nBy number of edge pixels detected:")
        for rank, (name, stats) in enumerate(sorted_by_edges, 1):
            print(f"  {rank}. {name:<15} - {stats['edge_percentage']:>6.2f}% edges")

        print("\nBy edge intensity (sensitivity):")
        for rank, (name, stats) in enumerate(sorted_by_intensity, 1):
            print(f"  {rank}. {name:<15} - {stats['mean_intensity']:>6.2f} avg intensity")


def create_documentation(output_dir):
    """
    Create documentation file with edge detection algorithm information.

    Args:
        output_dir (str): Directory to save documentation.
    """
    os.makedirs(output_dir, exist_ok=True)

    doc_path = os.path.join(output_dir, 'EDGE_DETECTION_GUIDE.txt')

    doc_content = """
================================================================================
OPENCV EDGE DETECTION APPLICATION - USER GUIDE
================================================================================

OVERVIEW:
This interactive application demonstrates 4 edge detection methods with
real-time parameter adjustment using trackbars.

EDGE DETECTION METHODS:
================================================================================

1. CANNY EDGE DETECTION
   Algorithm: Multi-stage edge detection with Gaussian blur and gradient
   Thresholds:
     - Low threshold: Edges with gradient above this are considered edges
     - High threshold: Edges with gradient above this are definitely edges
     - Hysteresis: Edges connected to strong edges are included
   Parameters: Low threshold (0-300), High threshold (0-300)
   Best for: Precise edge detection with clear boundaries

   Advantages:
     - Excellent edge localization
     - Single-pixel edge thickness
     - Low false positive rate
   Disadvantages:
     - Sensitive to threshold selection
     - May miss weak edges

2. SOBEL EDGE DETECTION
   Algorithm: Discrete differentiation using Sobel operators
   Operators:
     - Sobelx: Horizontal gradient (left-right edges)
     - Sobely: Vertical gradient (up-down edges)
     - Combined: sqrt(Sobelx² + Sobely²)
   Parameters: Kernel size (3, 5, 7, 9, 11)
   Best for: General-purpose edge detection

   Advantages:
     - Fast computation
     - Works with any kernel size
     - Good for various edge types
   Disadvantages:
     - Thicker edge lines
     - May detect noise as edges

3. LAPLACIAN EDGE DETECTION
   Algorithm: Second derivative edge detection
   Properties:
     - Detects edges where gradient changes
     - Very sensitive to noise
     - Bi-directional (detects all edge directions)
   Parameters: Fixed kernel size (3)
   Best for: Quick edge detection, detecting intensity changes

   Advantages:
     - Simple and fast
     - Detects all edge directions equally
     - No threshold parameters
   Disadvantages:
     - Very noisy
     - Requires further processing (thresholding)
     - Can produce double edges

4. SCHARR EDGE DETECTION
   Algorithm: Optimized Sobel operator for small kernels
   Properties:
     - Better rotation invariance than Sobel
     - More accurate for small kernels
     - Similar computation to Sobel
   Parameters: Fixed (inherently uses optimal kernel)
   Best for: Medical imaging, precise edge detection

   Advantages:
     - Better accuracy than Sobel
     - Optimized for kernel size 3
     - More balanced in all directions
   Disadvantages:
     - Fixed kernel
     - Slower than Sobel
     - Similar noise sensitivity

INTERACTIVE CONTROLS:
================================================================================

Trackbars:
  • Canny Low Threshold: Adjust lower threshold for Canny detection (0-300)
  • Canny High Threshold: Adjust upper threshold for Canny detection (0-300)
  • Kernel Size: Adjust Sobel/Scharr kernel size (3, 5, 7, 9, 11)

Keyboard:
  • 'q': Quit application
  • 's': Save current results

RECOMMENDATIONS:
================================================================================

Use Canny when:
  - You need precise edge localization
  - Working with high-quality images
  - Binary edge maps are acceptable

Use Sobel when:
  - You need gradient information
  - Working with varying image qualities
  - You can adjust kernel size for your needs

Use Laplacian when:
  - You need a quick edge detection
  - Testing image quality
  - You want maximum simplicity

Use Scharr when:
  - Accuracy is critical
  - Working with small objects
  - You need rotation invariance

PARAMETER TUNING GUIDE:
================================================================================

Canny Thresholds:
  • Typical range: Low (50-100), High (150-250)
  • Ratio: High threshold usually 2-3x low threshold
  • Lower values: Detect more edges (including noise)
  • Higher values: Detect fewer, stronger edges

Sobel Kernel Size:
  • 3: Fast, fine details, more noise
  • 5: Balanced detection
  • 7+: Smoother edges, less noise, slower

OUTPUT:
================================================================================

Saved files include:
  - canny_low{value}_high{value}.jpg: Canny detection result
  - sobel_ksize{value}.jpg: Sobel detection result
  - laplacian.jpg: Laplacian detection result
  - scharr.jpg: Scharr detection result

================================================================================
"""

    with open(doc_path, 'w') as f:
        f.write(doc_content)

    print(f"✓ Documentation created: {doc_path}")


def main():
    """
    Execute the interactive edge detection application.

    Creates and runs an interactive GUI for evaluating and comparing
    all 4 edge detection methods with real-time parameter adjustment.
    """
    print("\n" + "="*70)
    print("OPENCV EDGE DETECTOR APPLICATION - HANDS-ON EXERCISES")
    print("="*70)

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'test_image.jpg')
    output_dir = os.path.join(current_dir, 'output_edge_detection')

    # Create application
    print("\nInitializing application...")
    app = EdgeDetectorApp(image_path)

    # Create documentation
    create_documentation(output_dir)

    # Run interactive application
    app.display_with_trackbars()

    # Perform analysis on final results
    app.analyze_edges()

    print("\n" + "="*70)
    print("✓ APPLICATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
