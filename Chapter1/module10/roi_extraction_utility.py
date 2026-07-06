"""
ROI Extraction Utility - Region of Interest Isolation Exercises

Demonstrates three fundamental ROI extraction techniques:
1. Face ROI (Bounding Box Slicing) - Extract faces using bounding coordinates
2. Circular ROI (Vignette Isolation) - Extract circular regions with soft vignette edges
3. Object ROI (Polygon Segment) - Extract irregular polygon-shaped regions

Techniques covered:
- Coordinate-based slicing: frame[y:y+h, x:x+w]
- Circular masking: cv2.circle() with cv2.bitwise_and()
- Polygon masking: cv2.fillPoly() with cv2.bitwise_and()
- Blending and vignette effects
- Combined visualization dashboards
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt


class ROIExtractionUtility:
    """
    Extract and visualize regions of interest (ROI) from images.

    This class implements three ROI extraction methods that are fundamental
    to computer vision applications: face detection regions, circular ROI
    masks for vignette effects, and polygon-based irregular shape extraction.

    Each method demonstrates both the extraction process and visualization
    of results with comprehensive analysis.
    """

    def __init__(self, output_dir=None):
        """
        Initialize ROI extraction utility with output directory.

        Args:
            output_dir (str, optional): Directory for saving outputs.
                If None, creates 'output_roi' in script directory.
        """
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, 'output_roi')

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_or_generate_image(self, image_path=None, width=600, height=400):
        """
        Load image from file or generate synthetic test image.

        Creates a test image with colored regions suitable for ROI extraction
        demonstrations if no custom image is provided.

        Args:
            image_path (str, optional): Path to custom image. Defaults to None.
            width (int): Width of synthetic image in pixels. Default is 600.
            height (int): Height of synthetic image in pixels. Default is 400.

        Returns:
            ndarray: BGR color image in OpenCV format.

        Raises:
            ValueError: If image_path is provided but cannot be read.
        """
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unable to load image from {image_path}")
            return img
        return self.generate_test_image(width, height)

    def generate_test_image(self, width=600, height=400):
        """
        Generate synthetic test image with colored regions and patterns.

        Creates an image with:
        - Colored rectangles in different regions
        - Gradient backgrounds
        - Patterns suitable for polygon ROI demonstration
        - Face-like region (rectangle) for bounding box demo

        Args:
            width (int): Image width in pixels. Default is 600.
            height (int): Image height in pixels. Default is 400.

        Returns:
            ndarray: Synthetic BGR test image.

        Notes:
            - Creates visually distinct regions for ROI extraction practice
            - Saved as 'generated_test_image.png' to output_dir
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Create background gradient
        for y in range(height):
            intensity = int(100 + 155 * (y / height))
            img[y, :] = [intensity, intensity, intensity]

        # Face ROI region (blue rectangle) - simulate face bounding box
        face_x, face_y = 150, 80
        face_w, face_h = 120, 150
        cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h),
                     (200, 150, 50), -1)
        cv2.putText(img, 'Face ROI', (face_x + 20, face_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Circular ROI region (green)
        circle_center = (width - 120, 100)
        circle_radius = 60
        cv2.circle(img, circle_center, circle_radius, (50, 200, 50), -1)
        cv2.putText(img, 'Circular', (circle_center[0] - 40, circle_center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Polygon ROI region (red triangle-like shape)
        triangle_points = np.array([
            [50, 250],
            [150, 300],
            [100, 380],
            [50, 350]
        ], dtype=np.int32)
        cv2.polylines(img, [triangle_points], True, (50, 50, 200), 2)
        cv2.fillPoly(img, [triangle_points], (100, 100, 255))
        cv2.putText(img, 'Polygon', (60, 310), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 255), 2)

        # Additional colored region
        cv2.rectangle(img, (350, 250), (550, 380), (200, 200, 50), -1)
        cv2.putText(img, 'Object Region', (360, 310),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(self.output_dir, 'generated_test_image.png'), img)
        return img

    def extract_face_roi(self, img, x, y, w, h):
        """
        Extract face ROI using bounding box coordinate slicing.

        Uses simple numpy array slicing to extract a rectangular region
        defined by (x, y, width, height) coordinates. This is the most
        common and efficient ROI extraction method for face detection.

        Args:
            img (ndarray): Source BGR image.
            x (int): X-coordinate of top-left corner.
            y (int): Y-coordinate of top-left corner.
            w (int): Width of bounding box in pixels.
            h (int): Height of bounding box in pixels.

        Returns:
            ndarray: Extracted rectangular ROI.

        Notes:
            - Uses img[y:y+h, x:x+w] slicing (note: y before x)
            - Fast operation, no mask computation needed
            - Extracted region maintains original colors
        """
        face_roi = img[y:y+h, x:x+w].copy()
        return face_roi

    def extract_circular_roi(self, img, center_x, center_y, radius, fade=False):
        """
        Extract circular ROI using mask-based vignette isolation.

        Creates a circular mask on black canvas, then uses bitwise AND operation
        to extract only the circular region from source image. Optional fade
        parameter creates soft vignette edges.

        Args:
            img (ndarray): Source BGR image.
            center_x (int): X-coordinate of circle center.
            center_y (int): Y-coordinate of circle center.
            radius (int): Radius of circular ROI in pixels.
            fade (bool): If True, creates gradient vignette edges. Default is False.

        Returns:
            tuple: Two-element tuple containing:
                - circular_roi (ndarray): Extracted circular region
                - mask (ndarray): Binary circle mask used for extraction

        Notes:
            - Creates mask on black canvas (zeros)
            - Draws white circle cv2.circle(mask, center, radius, 255, -1)
            - Uses cv2.bitwise_and(img, img, mask=mask) for extraction
            - Fade creates smooth transition with gradient mask
        """
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        if fade:
            # Create gradient mask for vignette effect
            X, Y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            gradient_mask = np.clip(255 * (1 - dist / radius), 0, 255).astype(np.uint8)
            mask = gradient_mask

        circular_roi = cv2.bitwise_and(img, img, mask=mask)
        return circular_roi, mask

    def extract_polygon_roi(self, img, vertices):
        """
        Extract polygon ROI using irregular polygon mask.

        Defines a polygon shape using vertex coordinates, fills it white on
        black canvas, then uses bitwise AND to extract the polygon region
        from the source image.

        Args:
            img (ndarray): Source BGR image.
            vertices (ndarray): Array of polygon vertex coordinates.
                Shape: (N, 2) where N is number of vertices.
                Format: [[x1, y1], [x2, y2], ..., [xN, yN]]

        Returns:
            tuple: Two-element tuple containing:
                - polygon_roi (ndarray): Extracted polygon region
                - mask (ndarray): Binary polygon mask used for extraction

        Notes:
            - Creates mask on black canvas (zeros)
            - Uses cv2.fillPoly(mask, [vertices], 255) for solid fill
            - Uses cv2.bitwise_and(img, img, mask=mask) for extraction
            - Vertices should be in clockwise or counter-clockwise order
            - Supports any number of vertices (triangles, quadrilaterals, etc.)
        """
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        vertices = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 255)

        polygon_roi = cv2.bitwise_and(img, img, mask=mask)
        return polygon_roi, mask

    def create_roi_dashboard(self, img, face_roi, circular_roi, polygon_roi,
                            face_mask, circular_mask, polygon_mask):
        """
        Generate 3×3 visualization dashboard showing all ROI extraction methods.

        Creates comprehensive matplotlib figure with:
        - Row 1: Original image, Face ROI, Face mask
        - Row 2: Original image, Circular ROI, Circular mask
        - Row 3: Original image, Polygon ROI, Polygon mask

        Args:
            img (ndarray): Original BGR image.
            face_roi (ndarray): Extracted face ROI.
            circular_roi (ndarray): Extracted circular ROI.
            polygon_roi (ndarray): Extracted polygon ROI.
            face_mask (ndarray): Face extraction mask (all ones for bounding box).
            circular_mask (ndarray): Circular extraction mask.
            polygon_mask (ndarray): Polygon extraction mask.

        Notes:
            - Saves as 'roi_extraction_dashboard.png' to output_dir
            - 150 DPI for publication-quality output
            - Displays masks in grayscale, ROIs in color
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('ROI Extraction Methods: Complete Visualization',
                    fontsize=16, fontweight='bold')

        # Row 1: Face ROI
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Face ROI (Bounding Box)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(face_mask, cmap='gray')
        axes[0, 2].set_title('Face Mask')
        axes[0, 2].axis('off')

        # Row 2: Circular ROI
        axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(circular_roi, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Circular ROI (Vignette)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(circular_mask, cmap='gray')
        axes[1, 2].set_title('Circular Mask')
        axes[1, 2].axis('off')

        # Row 3: Polygon ROI
        axes[2, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title('Original Image')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(cv2.cvtColor(polygon_roi, cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title('Polygon ROI (Irregular Shape)')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(polygon_mask, cmap='gray')
        axes[2, 2].set_title('Polygon Mask')
        axes[2, 2].axis('off')

        plt.tight_layout()
        dashboard_path = os.path.join(self.output_dir, 'roi_extraction_dashboard.png')
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()

    def print_statistics(self, img, face_roi, circular_roi, polygon_roi):
        """
        Print ROI extraction statistics to console.

        Outputs information about extracted regions including:
        - Image resolution
        - Face ROI dimensions and area
        - Circular ROI radius and area
        - Polygon ROI area
        - Comparison of extraction methods

        Args:
            img (ndarray): Original image.
            face_roi (ndarray): Extracted face ROI.
            circular_roi (ndarray): Extracted circular ROI.
            polygon_roi (ndarray): Extracted polygon ROI.

        Prints:
            - Formatted statistics table
            - Method descriptions
            - Performance notes
        """
        print("\n" + "="*70)
        print("ROI EXTRACTION - ANALYSIS REPORT")
        print("="*70)
        print(f"Image Resolution: {img.shape[1]}x{img.shape[0]} pixels")
        print(f"Image Total Area: {img.shape[0] * img.shape[1]} pixels²")
        print("-"*70)

        print(f"\nFACE ROI (BOUNDING BOX):")
        print(f"  Dimensions: {face_roi.shape[1]}x{face_roi.shape[0]} pixels")
        print(f"  Area: {face_roi.shape[0] * face_roi.shape[1]} pixels²")
        print(f"  Coverage: {100 * face_roi.shape[0] * face_roi.shape[1] / (img.shape[0] * img.shape[1]):.2f}%")
        print(f"  Method: Coordinate slicing img[y:y+h, x:x+w]")
        print(f"  Use Case: Face detection, object tracking")

        print(f"\nCIRCULAR ROI (VIGNETTE):")
        # Count non-zero pixels in circular ROI for area estimation
        circular_area = np.count_nonzero(cv2.cvtColor(circular_roi, cv2.COLOR_BGR2GRAY))
        print(f"  Non-zero Pixels: {circular_area}")
        print(f"  Coverage: {100 * circular_area / (img.shape[0] * img.shape[1]):.2f}%")
        print(f"  Method: Mask-based extraction with cv2.bitwise_and()")
        print(f"  Use Case: Focus regions, spotlight effects, attention maps")

        print(f"\nPOLYGON ROI (IRREGULAR SHAPE):")
        # Count non-zero pixels in polygon ROI for area estimation
        polygon_area = np.count_nonzero(cv2.cvtColor(polygon_roi, cv2.COLOR_BGR2GRAY))
        print(f"  Non-zero Pixels: {polygon_area}")
        print(f"  Coverage: {100 * polygon_area / (img.shape[0] * img.shape[1]):.2f}%")
        print(f"  Method: Polygon mask with cv2.fillPoly() + cv2.bitwise_and()")
        print(f"  Use Case: Object segmentation, road lane detection")

        print("-"*70)
        print(f"Output Directory: {os.path.abspath(self.output_dir)}")
        print("="*70 + "\n")

    def save_report(self, face_roi, circular_roi, polygon_roi):
        """
        Save detailed ROI extraction report to text file.

        Generates comprehensive report with:
        - Extraction method descriptions
        - Technical details and algorithms
        - Performance characteristics
        - Use cases and applications
        - Code examples for each method

        Args:
            face_roi (ndarray): Extracted face ROI.
            circular_roi (ndarray): Extracted circular ROI.
            polygon_roi (ndarray): Extracted polygon ROI.

        Notes:
            - Saves to '{output_dir}/roi_extraction_report.txt'
            - Includes timestamp for traceability
        """
        report_path = os.path.join(self.output_dir, 'roi_extraction_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ROI EXTRACTION - DETAILED ANALYSIS REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-"*70 + "\n")

            f.write("\nEXTRACTION METHODS:\n\n")

            f.write("1. FACE ROI (BOUNDING BOX SLICING)\n")
            f.write("   Algorithm: Coordinate-based array slicing\n")
            f.write("   Code: face = img[y:y+h, x:x+w]\n")
            f.write(f"   Output Dimensions: {face_roi.shape[1]}x{face_roi.shape[0]} pixels\n")
            f.write("   Speed: O(w*h) - Very fast\n")
            f.write("   Use Cases:\n")
            f.write("   - Face detection and recognition\n")
            f.write("   - Object tracking with bounding boxes\n")
            f.write("   - License plate detection\n")
            f.write("   Advantages:\n")
            f.write("   - Simple and fast\n")
            f.write("   - No mask computation needed\n")
            f.write("   - Preserves original colors\n")
            f.write("   Limitations:\n")
            f.write("   - Only rectangular regions\n")
            f.write("   - Includes background in corners\n\n")

            f.write("2. CIRCULAR ROI (VIGNETTE ISOLATION)\n")
            f.write("   Algorithm: Circle mask + bitwise AND\n")
            f.write("   Code: cv2.circle(mask, center, radius, 255, -1)\n")
            f.write("         roi = cv2.bitwise_and(img, img, mask=mask)\n")
            circular_area = np.count_nonzero(cv2.cvtColor(circular_roi, cv2.COLOR_BGR2GRAY))
            f.write(f"   Non-zero Pixels: {circular_area}\n")
            f.write("   Speed: O(w*h) - Fast\n")
            f.write("   Use Cases:\n")
            f.write("   - Iris/pupil detection\n")
            f.write("   - Attention mechanism visualization\n")
            f.write("   - Spotlight and vignette effects\n")
            f.write("   Advantages:\n")
            f.write("   - Circular precision\n")
            f.write("   - Soft edge capability (vignette)\n")
            f.write("   - Efficient mask reuse\n")
            f.write("   Variations:\n")
            f.write("   - Gradient mask for fade effect\n")
            f.write("   - Elliptical mask for eyes\n\n")

            f.write("3. POLYGON ROI (IRREGULAR SEGMENT)\n")
            f.write("   Algorithm: Polygon mask + bitwise AND\n")
            f.write("   Code: cv2.fillPoly(mask, [vertices], 255)\n")
            f.write("         roi = cv2.bitwise_and(img, img, mask=mask)\n")
            polygon_area = np.count_nonzero(cv2.cvtColor(polygon_roi, cv2.COLOR_BGR2GRAY))
            f.write(f"   Non-zero Pixels: {polygon_area}\n")
            f.write("   Speed: O(w*h) - Fast\n")
            f.write("   Use Cases:\n")
            f.write("   - Road lane detection\n")
            f.write("   - Building/roof detection\n")
            f.write("   - Semantic segmentation\n")
            f.write("   Advantages:\n")
            f.write("   - Arbitrary polygon shapes\n")
            f.write("   - Precise boundary control\n")
            f.write("   - Multiple vertices support\n")
            f.write("   Vertex Ordering:\n")
            f.write("   - Clockwise or counter-clockwise both work\n")
            f.write("   - Order determines fill direction\n")
            f.write("   - Closed polygon (last connects to first)\n\n")

            f.write("-"*70 + "\n")
            f.write("COMPARISON:\n")
            f.write("Method          | Speed | Precision | Flexibility | Boundaries\n")
            f.write("Face (Box)      | Fast  | Lower     | Low         | Sharp\n")
            f.write("Circular        | Fast  | High      | Medium      | Smooth\n")
            f.write("Polygon         | Fast  | Highest   | High        | Precise\n")
            f.write("-"*70 + "\n")

    def run_analysis(self, image_path=None, face_coords=None, circle_coords=None,
                    polygon_coords=None):
        """
        Execute complete ROI extraction pipeline and generate visualizations.

        Orchestrates the entire ROI extraction workflow:
        1. Load or generate test image
        2. Extract face ROI (bounding box)
        3. Extract circular ROI (vignette)
        4. Extract polygon ROI (irregular shape)
        5. Create visualization dashboard
        6. Print console statistics
        7. Save detailed report
        8. Save all extracted ROIs and masks

        Args:
            image_path (str, optional): Path to custom image. Default is None.
            face_coords (tuple, optional): Face ROI coordinates (x, y, w, h).
                Default uses auto-detected from test image.
            circle_coords (tuple, optional): Circle coordinates (cx, cy, radius).
                Default uses auto-detected from test image.
            polygon_coords (list, optional): Polygon vertices [[x1,y1],[x2,y2],...].
                Default uses auto-detected from test image.

        Returns:
            tuple: Three-element tuple containing:
                - face_roi (ndarray): Extracted face ROI
                - circular_roi (ndarray): Extracted circular ROI
                - polygon_roi (ndarray): Extracted polygon ROI

        Notes:
            - Saves 11 output files to output_dir
            - Prints statistics to console
            - Generates visualization dashboard
        """
        img = self.load_or_generate_image(image_path)

        # Default coordinates from test image
        if face_coords is None:
            face_coords = (150, 80, 120, 150)
        if circle_coords is None:
            circle_coords = (480, 100, 60)
        if polygon_coords is None:
            polygon_coords = [[50, 250], [150, 300], [100, 380], [50, 350]]

        # Extract ROIs
        face_x, face_y, face_w, face_h = face_coords
        face_roi = self.extract_face_roi(img, face_x, face_y, face_w, face_h)
        face_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        face_mask[:face_y, :] = 0
        face_mask[face_y+face_h:, :] = 0
        face_mask[:, :face_x] = 0
        face_mask[:, face_x+face_w:] = 0

        circular_roi, circular_mask = self.extract_circular_roi(img, *circle_coords, fade=False)
        polygon_roi, polygon_mask = self.extract_polygon_roi(img, polygon_coords)

        # Create visualizations and reports
        self.create_roi_dashboard(img, face_roi, circular_roi, polygon_roi,
                                face_mask, circular_mask, polygon_mask)
        self.print_statistics(img, face_roi, circular_roi, polygon_roi)
        self.save_report(face_roi, circular_roi, polygon_roi)

        # Save individual ROIs and masks
        cv2.imwrite(os.path.join(self.output_dir, 'original_image.png'), img)
        cv2.imwrite(os.path.join(self.output_dir, 'face_roi.png'), face_roi)
        cv2.imwrite(os.path.join(self.output_dir, 'face_mask.png'), face_mask)
        cv2.imwrite(os.path.join(self.output_dir, 'circular_roi.png'), circular_roi)
        cv2.imwrite(os.path.join(self.output_dir, 'circular_mask.png'), circular_mask)
        cv2.imwrite(os.path.join(self.output_dir, 'polygon_roi.png'), polygon_roi)
        cv2.imwrite(os.path.join(self.output_dir, 'polygon_mask.png'), polygon_mask)

        return face_roi, circular_roi, polygon_roi


def main():
    """
    Parse command-line arguments and run ROI extraction analysis.

    Entry point for command-line usage. Parses arguments and creates
    ROIExtractionUtility instance to run analysis with specified parameters.

    Command-line arguments:
        --custom PATH: Path to custom image (JPG, PNG, BMP, TIFF)
        --output-dir PATH: Output directory for results (default: module10/output_roi)

    Examples:
        Generate synthetic test image:
            python roi_extraction_utility.py

        Analyze custom image:
            python roi_extraction_utility.py --custom photo.jpg

        With custom output directory:
            python roi_extraction_utility.py --custom photo.jpg \\
                --output-dir ./results

    Prints:
        - Console statistics from analysis
        - File paths where results are saved
    """
    parser = argparse.ArgumentParser(
        description='ROI Extraction Utility - Extract regions of interest from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate synthetic test image and extract ROIs
  python roi_extraction_utility.py

  # Use custom image with default settings
  python roi_extraction_utility.py --custom photo.jpg

  # Save results to specific directory
  python roi_extraction_utility.py --custom photo.jpg --output-dir ./results
        '''
    )

    parser.add_argument('--custom', type=str, default=None,
                       help='Path to custom image (JPG, PNG, etc.)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: module10/output_roi)')

    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'output_roi')

    extractor = ROIExtractionUtility(output_dir)
    extractor.run_analysis(args.custom)


if __name__ == '__main__':
    main()
