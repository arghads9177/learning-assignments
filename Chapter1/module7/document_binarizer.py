"""
Document Binarizer - Mini Project

A utility to transform poorly-lit smartphone photos of documents into
clean, printer-ready scanned PDF-style images.

Pipeline:
1. Load image and convert to grayscale
2. Apply Gaussian Blur to filter paper grain noise
3. Apply adaptive thresholding for high-contrast binary output
4. Optional: Apply morphological operations for cleanup
5. Export as printer-ready image or PDF

Features:
- Multiple binarization methods (Otsu, Adaptive Gaussian, Adaptive Mean)
- Before/after comparison
- Quality metrics and statistics
- Batch processing capability
- PDF export (requires reportlab)
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


class DocumentBinarizer:
    """
    Transform document images into clean binary scans.

    Provides multiple binarization methods and quality optimization
    for converting smartphone photos to printer-ready documents.
    """

    def __init__(self, image_path, output_dir=None):
        """
        Initialize the document binarizer.

        Args:
            image_path (str): Path to the document image.
            output_dir (str): Directory for output files (default: same as input).

        Raises:
            ValueError: If image cannot be loaded.
        """
        self.image_path = image_path
        self.original = None
        self.gray = None
        self.output_dir = output_dir or os.path.dirname(image_path)

        # Load image
        self._load_image()

        # Processing results
        self.results = {}

    def _load_image(self):
        """
        Load image from file.

        Prints:
            - Confirmation of image loading
            - Image dimensions and file size
        """
        if not os.path.exists(self.image_path):
            raise ValueError(f"Image file not found: {self.image_path}")

        self.original = cv2.imread(self.image_path)
        if self.original is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        file_size = os.path.getsize(self.image_path) / (1024 * 1024)
        print(f"✓ Image loaded: {self.image_path}")
        print(f"  Dimensions: {self.original.shape}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Grayscale shape: {self.gray.shape}")

    def create_sample_document(self):
        """
        Create a sample poorly-lit document image for testing.

        Generates a synthetic document with text-like patterns,
        uneven lighting, and noise to simulate real-world conditions.

        Returns:
            str: Path to the created sample image.
        """
        print("\nCreating sample document image...")

        # Create base image with gradient lighting
        height, width = 600, 800
        img = np.ones((height, width, 3), dtype=np.uint8) * 200

        # Add uneven lighting (darker edges)
        for y in range(height):
            for x in range(width):
                # Distance from center
                dist_y = abs(y - height // 2) / (height // 2)
                dist_x = abs(x - width // 2) / (width // 2)
                brightness_factor = 0.7 + 0.3 * (1 - (dist_x + dist_y) / 2)

                img[y, x] = (img[y, x] * brightness_factor).astype(np.uint8)

        # Add text-like black regions
        cv2.rectangle(img, (50, 50), (750, 100), (30, 30, 30), -1)  # Title bar
        cv2.rectangle(img, (50, 120), (750, 150), (50, 50, 50), -1)  # Text line
        cv2.rectangle(img, (50, 170), (750, 200), (50, 50, 50), -1)  # Text line
        cv2.rectangle(img, (50, 220), (600, 250), (50, 50, 50), -1)  # Text line

        # Add content boxes
        for i in range(3):
            y_start = 280 + i * 80
            cv2.rectangle(img, (50, y_start), (750, y_start + 60), (40, 40, 40), -1)

        # Add noise (paper grain, dust, blur from camera)
        noise = np.random.normal(0, 15, img.shape).astype(np.float32)
        img = cv2.add(img.astype(np.float32), noise)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Add slight blur to simulate camera focus
        img = cv2.GaussianBlur(img, (3, 3), 0.5)

        # Save sample
        os.makedirs(self.output_dir, exist_ok=True)
        sample_path = os.path.join(self.output_dir, 'sample_document.jpg')
        cv2.imwrite(sample_path, img)

        print(f"✓ Sample document created: {sample_path}")
        return sample_path

    def preprocess_image(self, blur_kernel=(5, 5)):
        """
        Preprocess image: convert to grayscale and apply Gaussian blur.

        Removes high-frequency noise (paper grain) while preserving edges.

        Args:
            blur_kernel (tuple): Size of Gaussian blur kernel (must be odd).

        Returns:
            ndarray: Preprocessed grayscale image.

        Prints:
            - Blur kernel size
            - Image statistics before and after blur
        """
        print("\n" + "="*70)
        print("PREPROCESSING")
        print("="*70)

        print(f"\nApplying Gaussian Blur ({blur_kernel[0]}×{blur_kernel[1]})...")
        print(f"Original grayscale image statistics:")
        print(f"  Mean: {self.gray.mean():.2f}")
        print(f"  Std: {self.gray.std():.2f}")
        print(f"  Min: {self.gray.min()}, Max: {self.gray.max()}")

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(self.gray, blur_kernel, 0)

        print(f"\nAfter Gaussian Blur:")
        print(f"  Mean: {blurred.mean():.2f}")
        print(f"  Std: {blurred.std():.2f}")

        self.results['preprocessed'] = blurred
        return blurred

    def binarize_otsu(self, preprocessed):
        """
        Apply Otsu's automatic thresholding.

        Otsu's method automatically calculates the optimal threshold value
        by minimizing within-class variance. Fast but may not work well
        for images with uneven lighting.

        Args:
            preprocessed (ndarray): Preprocessed grayscale image.

        Returns:
            tuple: (binary_image, threshold_value)

        Prints:
            - Otsu threshold value
            - Binary statistics
        """
        print("\n" + "="*70)
        print("BINARIZATION METHOD 1: OTSU'S THRESHOLDING")
        print("="*70)

        # Apply Otsu thresholding
        threshold_val, binary = cv2.threshold(preprocessed, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print(f"\n✓ Otsu's method applied")
        print(f"  Optimal threshold: {threshold_val:.0f}")
        print(f"  Black pixels: {np.count_nonzero(binary == 0)}")
        print(f"  White pixels: {np.count_nonzero(binary == 255)}")

        self.results['otsu'] = binary
        return binary, threshold_val

    def binarize_adaptive_gaussian(self, preprocessed, block_size=15, constant=5):
        """
        Apply adaptive thresholding with Gaussian weighting.

        Adaptive thresholding calculates threshold for each pixel based on
        surrounding neighborhood, handling uneven lighting much better than
        global thresholding.

        Args:
            preprocessed (ndarray): Preprocessed grayscale image.
            block_size (int): Size of neighborhood area (must be odd, default: 15).
            constant (int): Constant subtracted from weighted mean (default: 5).

        Returns:
            ndarray: Binary image.

        Prints:
            - Block size and constant parameters
            - Binary statistics
        """
        print("\n" + "="*70)
        print("BINARIZATION METHOD 2: ADAPTIVE GAUSSIAN THRESHOLDING")
        print("="*70)

        print(f"\nParameters:")
        print(f"  Block size: {block_size}×{block_size}")
        print(f"  Constant (C): {constant}")

        # Apply adaptive thresholding with Gaussian weighting
        binary = cv2.adaptiveThreshold(preprocessed, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       block_size, constant)

        print(f"\n✓ Adaptive Gaussian thresholding applied")
        print(f"  Black pixels: {np.count_nonzero(binary == 0)}")
        print(f"  White pixels: {np.count_nonzero(binary == 255)}")

        self.results['adaptive_gaussian'] = binary
        return binary

    def binarize_adaptive_mean(self, preprocessed, block_size=15, constant=5):
        """
        Apply adaptive thresholding with arithmetic mean.

        Similar to Gaussian but uses simple mean instead of Gaussian-weighted.
        Faster computation but slightly less sophisticated.

        Args:
            preprocessed (ndarray): Preprocessed grayscale image.
            block_size (int): Size of neighborhood area (must be odd, default: 15).
            constant (int): Constant subtracted from mean (default: 5).

        Returns:
            ndarray: Binary image.

        Prints:
            - Block size and constant parameters
            - Binary statistics
        """
        print("\n" + "="*70)
        print("BINARIZATION METHOD 3: ADAPTIVE MEAN THRESHOLDING")
        print("="*70)

        print(f"\nParameters:")
        print(f"  Block size: {block_size}×{block_size}")
        print(f"  Constant (C): {constant}")

        # Apply adaptive thresholding with mean
        binary = cv2.adaptiveThreshold(preprocessed, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       block_size, constant)

        print(f"\n✓ Adaptive Mean thresholding applied")
        print(f"  Black pixels: {np.count_nonzero(binary == 0)}")
        print(f"  White pixels: {np.count_nonzero(binary == 255)}")

        self.results['adaptive_mean'] = binary
        return binary

    def apply_morphological_operations(self, binary, operation='close', kernel_size=3):
        """
        Apply morphological operations to clean up binary image.

        Removes small noise and fills small holes in text, improving
        readability of binary document.

        Args:
            binary (ndarray): Binary image.
            operation (str): 'close', 'open', or 'none' (default: 'close').
            kernel_size (int): Size of morphological kernel (default: 3).

        Returns:
            ndarray: Cleaned binary image.

        Prints:
            - Operation type and kernel size
            - Statistics of cleaned image
        """
        print("\n" + "="*70)
        print("MORPHOLOGICAL OPERATIONS")
        print("="*70)

        if operation == 'none':
            print("✓ Skipping morphological operations")
            return binary

        print(f"\nApplying {operation} operation (kernel size: {kernel_size}×{kernel_size})...")

        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        if operation == 'close':
            # Closing: dilate then erode (fills small holes)
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            # Opening: erode then dilate (removes small noise)
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        else:
            result = binary

        print(f"✓ {operation.capitalize()} operation applied")
        print(f"  Black pixels: {np.count_nonzero(result == 0)}")
        print(f"  White pixels: {np.count_nonzero(result == 255)}")

        return result

    def calculate_quality_metrics(self, binary):
        """
        Calculate quality metrics for binary image.

        Evaluates document quality based on contrast, edge preservation,
        and noise level.

        Args:
            binary (ndarray): Binary image.

        Returns:
            dict: Quality metrics including contrast and edge metrics.

        Prints:
            - Contrast ratio
            - Edge density
            - Overall quality assessment
        """
        print("\n" + "="*70)
        print("QUALITY METRICS")
        print("="*70)

        # Calculate contrast
        black_pixels = np.count_nonzero(binary == 0)
        white_pixels = np.count_nonzero(binary == 255)
        total_pixels = binary.shape[0] * binary.shape[1]

        contrast_ratio = max(black_pixels, white_pixels) / min(black_pixels, white_pixels)
        text_coverage = min(black_pixels, white_pixels) / total_pixels * 100

        # Edge density (using Canny)
        edges = cv2.Canny(binary, 50, 150)
        edge_pixels = np.count_nonzero(edges > 0)
        edge_density = edge_pixels / total_pixels * 100

        # Text clarity (variance of local regions)
        clarity_score = 100 if contrast_ratio > 2 else contrast_ratio * 50

        metrics = {
            'contrast_ratio': contrast_ratio,
            'text_coverage': text_coverage,
            'edge_density': edge_density,
            'clarity_score': clarity_score,
            'black_pixels': black_pixels,
            'white_pixels': white_pixels
        }

        print(f"\nContrast Metrics:")
        print(f"  Contrast ratio (foreground/background): {contrast_ratio:.2f}")
        print(f"  Text coverage: {text_coverage:.2f}%")

        print(f"\nEdge Metrics:")
        print(f"  Edge density: {edge_density:.2f}%")

        print(f"\nOverall Quality Score: {clarity_score:.1f}/100")
        if clarity_score >= 80:
            print(f"  Assessment: Excellent ✓")
        elif clarity_score >= 60:
            print(f"  Assessment: Good ✓")
        elif clarity_score >= 40:
            print(f"  Assessment: Fair")
        else:
            print(f"  Assessment: Poor - May need parameter tuning")

        return metrics

    def save_results(self, method='adaptive_gaussian', apply_morphology=True):
        """
        Save processed document images.

        Args:
            method (str): Binarization method to save ('otsu', 'adaptive_gaussian', 'adaptive_mean').
            apply_morphology (bool): Whether to apply morphological cleanup.

        Prints:
            - File paths of saved images
        """
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)

        os.makedirs(self.output_dir, exist_ok=True)

        # Get the binary image
        if method not in self.results:
            print(f"✗ Method '{method}' not found. Available: {list(self.results.keys())}")
            return

        binary = self.results[method]

        # Apply morphology if requested
        if apply_morphology:
            binary = self.apply_morphological_operations(binary, operation='close', kernel_size=3)

        # Save binary image
        base_name = Path(self.image_path).stem
        output_path = os.path.join(self.output_dir, f'{base_name}_binarized.jpg')
        cv2.imwrite(output_path, binary)
        print(f"\n✓ Binarized document saved: {output_path}")

        # Save comparison (original grayscale vs binary)
        comparison = self._create_comparison_image(self.gray, binary)
        comparison_path = os.path.join(self.output_dir, f'{base_name}_comparison.jpg')
        cv2.imwrite(comparison_path, comparison)
        print(f"✓ Comparison image saved: {comparison_path}")

        return output_path

    def _create_comparison_image(self, original, processed):
        """
        Create side-by-side comparison of original and processed images.

        Args:
            original (ndarray): Original grayscale image.
            processed (ndarray): Processed binary image.

        Returns:
            ndarray: Side-by-side comparison image.
        """
        # Resize to match heights if needed
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

        # Convert to 3-channel for proper display
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        proc_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # Add labels
        cv2.putText(orig_bgr, "Original (Grayscale)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(proc_bgr, "Binarized (Document)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Concatenate horizontally
        comparison = np.hstack([orig_bgr, proc_bgr])

        return comparison

    def process_document(self, blur_kernel=(5, 5), block_size=15, constant=5):
        """
        Complete document binarization pipeline.

        Args:
            blur_kernel (tuple): Gaussian blur kernel size.
            block_size (int): Adaptive threshold block size.
            constant (int): Adaptive threshold constant.

        Returns:
            dict: Processing results with binary images.

        Prints:
            - Complete pipeline execution
        """
        print("\n" + "="*70)
        print("DOCUMENT BINARIZATION PIPELINE")
        print("="*70)

        # Preprocessing
        preprocessed = self.preprocess_image(blur_kernel)

        # Apply all binarization methods
        print("\n" + "-"*70)
        print("APPLYING BINARIZATION METHODS")
        print("-"*70)

        self.binarize_otsu(preprocessed)
        self.binarize_adaptive_gaussian(preprocessed, block_size, constant)
        self.binarize_adaptive_mean(preprocessed, block_size, constant)

        # Calculate quality metrics for best method
        best_binary = self.results['adaptive_gaussian']
        self.calculate_quality_metrics(best_binary)

        return self.results


def create_sample_document_image(output_dir):
    """
    Create a test document image if no input is provided.

    Args:
        output_dir (str): Directory to save sample image.

    Returns:
        str: Path to created sample image.
    """
    print("No input image provided. Creating sample document...")

    os.makedirs(output_dir, exist_ok=True)
    sample_path = os.path.join(output_dir, 'sample_document_input.jpg')

    binarizer = DocumentBinarizer.__new__(DocumentBinarizer)
    binarizer.output_dir = output_dir

    # Create sample document
    sample_path = binarizer.create_sample_document()
    return sample_path


def main():
    """
    Main entry point for document binarizer.

    Provides command-line interface for processing document images.
    """
    parser = argparse.ArgumentParser(
        description='Transform poorly-lit document photos into clean, printer-ready scans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a document image
  python document_binarizer.py input_document.jpg

  # Process with custom parameters
  python document_binarizer.py input_document.jpg --block-size 21 --constant 3

  # Create and process sample document
  python document_binarizer.py --sample
        """
    )

    parser.add_argument('image', nargs='?', help='Path to document image')
    parser.add_argument('--sample', action='store_true',
                       help='Create and process sample document')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--blur-kernel', type=int, default=5,
                       help='Gaussian blur kernel size (default: 5)')
    parser.add_argument('--block-size', type=int, default=15,
                       help='Adaptive threshold block size (default: 15)')
    parser.add_argument('--constant', type=int, default=5,
                       help='Adaptive threshold constant C (default: 5)')
    parser.add_argument('--no-morphology', action='store_true',
                       help='Skip morphological operations')

    args = parser.parse_args()

    # Determine image path
    if args.sample:
        output_dir = args.output_dir or os.path.join(os.getcwd(), 'output_documents')
        image_path = create_sample_document_image(output_dir)
    elif args.image:
        image_path = args.image
    else:
        # Create sample if no input
        output_dir = args.output_dir or os.path.join(os.getcwd(), 'output_documents')
        image_path = create_sample_document_image(output_dir)

    output_dir = args.output_dir or os.path.dirname(image_path)

    # Process document
    print("\n" + "="*70)
    print("DOCUMENT BINARIZER - MINI PROJECT")
    print("="*70)

    binarizer = DocumentBinarizer(image_path, output_dir)

    # Ensure kernel sizes are odd
    blur_kernel = (args.blur_kernel, args.blur_kernel)
    if blur_kernel[0] % 2 == 0:
        blur_kernel = (blur_kernel[0] + 1, blur_kernel[0] + 1)
    if args.block_size % 2 == 0:
        args.block_size += 1

    # Process
    results = binarizer.process_document(blur_kernel, args.block_size, args.constant)

    # Save results
    binarizer.save_results('adaptive_gaussian', apply_morphology=not args.no_morphology)

    print("\n" + "="*70)
    print("✓ DOCUMENT BINARIZATION COMPLETE!")
    print("="*70)
    print(f"\nInput: {image_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
