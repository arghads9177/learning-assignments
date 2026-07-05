"""
Morphological Operations Visual Comparison - Hands-on Exercises

This module demonstrates the visual effects of morphological operations
on binary images by creating comprehensive visualization dashboards.

Operations covered:
1. Erosion - Shrinks white regions
2. Dilation - Expands white regions
3. Opening - Erosion followed by dilation (removes small objects)
4. Closing - Dilation followed by erosion (fills small holes)
5. Gradient (Morphological) - Dilation minus erosion (edges)
6. Top Hat - Original minus opening (small objects)
7. Black Hat - Closing minus original (dark spots in light areas)

Structuring elements:
- Rectangle (MORPH_RECT)
- Ellipse (MORPH_ELLIPSE)
- Cross (MORPH_CROSS)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import os


class MorphologicalVisualComparison:
    """
    Create comprehensive visual comparisons of morphological operations.

    Generates multi-panel dashboards showing how different morphological
    operations affect binary images with various structuring elements.
    """

    def __init__(self, output_dir, custom_image_path=None):
        """
        Initialize the visual comparison tool.

        Args:
            output_dir (str): Directory to save visualizations.
            custom_image_path (str): Optional path to custom image for analysis.
                If provided, this image will be used instead of generating one.
        """
        self.output_dir = output_dir
        self.custom_image_path = custom_image_path
        os.makedirs(output_dir, exist_ok=True)

    def load_custom_image(self, image_path, threshold_value=127):
        """
        Load and preprocess a custom image for morphological analysis.

        Converts the image to binary (black and white) format suitable for
        morphological operations. If the image is already binary, it's used
        as-is. Otherwise, it's converted to grayscale and thresholded.

        Args:
            image_path (str): Path to the custom image.
            threshold_value (int): Threshold value for binarization (0-255, default: 127).
                Only used if image is not already binary.

        Returns:
            tuple: (clean_mask, corrupted_mask) where:
                - clean_mask: Original binary image (no corruption added)
                - corrupted_mask: Same as clean_mask (no synthetic noise added)

        Prints:
            - Image loading confirmation
            - Preprocessing details
            - Binary image statistics

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        print("\n" + "="*70)
        print("LOADING CUSTOM IMAGE")
        print("="*70)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        print(f"✓ Image loaded: {image_path}")
        print(f"  Original shape: {img.shape}")
        print(f"  File size: {os.path.getsize(image_path) / 1024:.2f} KB")

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"✓ Converted to grayscale")
        else:
            gray = img

        # Convert to binary
        print(f"\nBinarizing image (threshold: {threshold_value})...")
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        print(f"✓ Image converted to binary")
        print(f"  Binary shape: {binary.shape}")
        print(f"  Black pixels: {np.count_nonzero(binary == 0)}")
        print(f"  White pixels: {np.count_nonzero(binary == 255)}")

        # For custom images, use the binary as both clean and corrupted
        # (user can add corruption manually if desired)
        return binary, binary

    def create_corrupted_binary_mask(self, width=400, height=400):
        """
        Create a clean binary mask with geometric shapes and add synthetic noise/cracks.

        Generates a binary image with:
        - Geometric shapes (rectangles, circles, lines)
        - Synthetic noise (salt-and-pepper)
        - Cracks and small gaps to simulate real-world degradation

        Args:
            width (int): Image width (default: 400).
            height (int): Image height (default: 400).

        Returns:
            tuple: (clean_mask, corrupted_mask) both ndarray uint8.

        Prints:
            - Mask dimensions
            - Corruption statistics
        """
        print("\n" + "="*70)
        print("CREATING CORRUPTED BINARY MASK")
        print("="*70)

        # Create clean binary mask
        clean = np.ones((height, width), dtype=np.uint8) * 255

        # Add geometric shapes (white on black background)
        clean[50:150, 50:150] = 0      # Rectangle (top-left)
        cv2.circle(clean, (300, 100), 50, 0, -1)  # Circle (top-right)

        # Add rectangular region
        cv2.rectangle(clean, (100, 250), (300, 350), 0, -1)

        # Add lines and patterns
        cv2.line(clean, (50, 200), (350, 200), 0, 3)
        cv2.line(clean, (200, 50), (200, 350), 0, 2)

        # Add cross pattern
        for i in range(0, width, 30):
            cv2.line(clean, (i, 0), (i, height), 0, 1)
        for i in range(0, height, 30):
            cv2.line(clean, (0, i), (width, i), 0, 1)

        print(f"✓ Clean binary mask created: {clean.shape}")
        print(f"  Black pixels (text/shapes): {np.count_nonzero(clean == 0)}")
        print(f"  White pixels (background): {np.count_nonzero(clean == 255)}")

        # Create corrupted version
        corrupted = clean.copy()

        # Add salt-and-pepper noise
        num_noise_pixels = (width * height) // 30  # ~3% noise
        for _ in range(num_noise_pixels):
            y = np.random.randint(0, height)
            x = np.random.randint(0, width)
            # Randomly add salt (white) or pepper (black) noise
            corrupted[y, x] = np.random.choice([0, 255])

        # Add random cracks/gaps
        for _ in range(20):
            y1, y2 = sorted(np.random.randint(0, height, 2))
            x1, x2 = sorted(np.random.randint(0, width, 2))
            if corrupted[y1, x1] == 0:  # Only in black regions
                cv2.line(corrupted, (x1, y1), (x2, y2), 255, 1)

        print(f"\n✓ Corrupted mask created with noise and cracks")
        print(f"  Black pixels: {np.count_nonzero(corrupted == 0)}")
        print(f"  White pixels: {np.count_nonzero(corrupted == 255)}")

        return clean, corrupted

    def create_structuring_elements(self, kernel_size=5):
        """
        Create structuring elements of different shapes.

        Uses cv2.getStructuringElement() to create:
        - Rectangle (MORPH_RECT)
        - Ellipse (MORPH_ELLIPSE)
        - Cross (MORPH_CROSS)

        Args:
            kernel_size (int): Size of structuring element (default: 5).

        Returns:
            dict: Dictionary with structuring element shapes.

        Prints:
            - Kernel size and shape information
            - Visual representation of kernels
        """
        print("\n" + "="*70)
        print("CREATING STRUCTURING ELEMENTS")
        print("="*70)

        print(f"\nKernel size: {kernel_size}×{kernel_size}")

        # Create structuring elements
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        kernels = {
            'rect': rect_kernel,
            'ellipse': ellipse_kernel,
            'cross': cross_kernel
        }

        # Print kernel information
        for name, kernel in kernels.items():
            non_zero = np.count_nonzero(kernel)
            print(f"\n{name.upper()} kernel (non-zero elements: {non_zero}):")
            print(kernel.astype(int))

        return kernels

    def apply_morphological_operations(self, corrupted_mask, kernel):
        """
        Apply all 7 morphological operations to the corrupted mask.

        Computes:
        1. Erosion - Shrinks white regions
        2. Dilation - Expands white regions
        3. Opening - Erosion then dilation
        4. Closing - Dilation then erosion
        5. Gradient - Dilation minus erosion
        6. Top Hat - Original minus opening
        7. Black Hat - Closing minus original

        Args:
            corrupted_mask (ndarray): Binary image (corrupted).
            kernel (ndarray): Structuring element.

        Returns:
            dict: Dictionary with all operation results.

        Prints:
            - Operation statistics (black/white pixel counts)
        """
        results = {}

        # 1. Erosion
        results['erosion'] = cv2.erode(corrupted_mask, kernel, iterations=1)

        # 2. Dilation
        results['dilation'] = cv2.dilate(corrupted_mask, kernel, iterations=1)

        # 3. Opening (erosion then dilation)
        results['opening'] = cv2.morphologyEx(corrupted_mask, cv2.MORPH_OPEN, kernel)

        # 4. Closing (dilation then erosion)
        results['closing'] = cv2.morphologyEx(corrupted_mask, cv2.MORPH_CLOSE, kernel)

        # 5. Gradient (dilation - erosion)
        results['gradient'] = cv2.morphologyEx(corrupted_mask, cv2.MORPH_GRADIENT, kernel)

        # 6. Top Hat (original - opening)
        results['top_hat'] = cv2.morphologyEx(corrupted_mask, cv2.MORPH_TOPHAT, kernel)

        # 7. Black Hat (closing - original)
        results['black_hat'] = cv2.morphologyEx(corrupted_mask, cv2.MORPH_BLACKHAT, kernel)

        return results

    def create_8panel_dashboard(self, corrupted_mask, results, kernel_name='rect',
                               kernel_size=5, output_path=None):
        """
        Create an 8-panel dashboard showing original and all morphological operations.

        Visualizes:
        - Panel 1: Original corrupted mask
        - Panel 2: Erosion result
        - Panel 3: Dilation result
        - Panel 4: Opening result
        - Panel 5: Closing result
        - Panel 6: Morphological Gradient
        - Panel 7: Top Hat
        - Panel 8: Black Hat

        Args:
            corrupted_mask (ndarray): Original binary image.
            results (dict): Dictionary of morphological operation results.
            kernel_name (str): Name of kernel used.
            kernel_size (int): Size of kernel.
            output_path (str): Path to save the dashboard (optional).

        Prints:
            - Dashboard creation confirmation
            - File save path
        """
        print("\n" + "="*70)
        print("CREATING 8-PANEL DASHBOARD")
        print("="*70)

        # Create figure with 2x4 grid
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Panel titles and images
        panels = [
            ('Original\n(Corrupted)', corrupted_mask),
            ('Erosion\n(Shrink)', results['erosion']),
            ('Dilation\n(Expand)', results['dilation']),
            ('Opening\n(Denoise)', results['opening']),
            ('Closing\n(Fill Holes)', results['closing']),
            ('Gradient\n(Edges)', results['gradient']),
            ('Top Hat\n(Details)', results['top_hat']),
            ('Black Hat\n(Dark Spots)', results['black_hat'])
        ]

        # Plot panels
        for idx, (title, img) in enumerate(panels):
            row = idx // 4
            col = idx % 4
            ax = fig.add_subplot(gs[row, col])

            # Ensure image is proper type for display
            if len(img.shape) == 2:
                # Grayscale
                ax.imshow(img, cmap='gray')
            else:
                # Color
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

            # Add pixel statistics as text
            black_px = np.count_nonzero(img == 0) if len(img.shape) == 2 else 0
            white_px = np.count_nonzero(img == 255) if len(img.shape) == 2 else 0
            total_px = img.shape[0] * img.shape[1]

            if len(img.shape) == 2:
                stats_text = f"Black: {black_px}\nWhite: {white_px}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add overall title
        fig.suptitle(f'Morphological Operations Dashboard\n'
                    f'Kernel: {kernel_name.upper()} ({kernel_size}×{kernel_size})',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save figure
        if output_path is None:
            output_path = os.path.join(self.output_dir,
                                      f'morphological_dashboard_{kernel_name}_{kernel_size}x{kernel_size}.png')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Dashboard saved: {output_path}")

        plt.close()

    def create_comparison_table(self, corrupted_mask, results, kernel_name, kernel_size):
        """
        Create a text report with quantitative analysis of morphological operations.

        Calculates and reports:
        - Pixel count changes
        - Change percentages
        - Operation effects summary

        Args:
            corrupted_mask (ndarray): Original binary image.
            results (dict): Dictionary of operation results.
            kernel_name (str): Name of kernel used.
            kernel_size (int): Size of kernel.

        Prints:
            - Comprehensive analysis table
            - Effect summary for each operation
        """
        print("\n" + "="*70)
        print("MORPHOLOGICAL OPERATIONS ANALYSIS")
        print("="*70)

        print(f"\nKernel: {kernel_name.upper()} ({kernel_size}×{kernel_size})")
        print(f"\n{'Operation':<15} {'Black Pixels':<15} {'White Pixels':<15} {'Change %':<12}")
        print("-" * 60)

        total_pixels = corrupted_mask.shape[0] * corrupted_mask.shape[1]
        original_black = np.count_nonzero(corrupted_mask == 0)
        original_white = np.count_nonzero(corrupted_mask == 255)

        print(f"{'Original':<15} {original_black:<15} {original_white:<15} {'0.0%':<12}")

        for op_name in ['erosion', 'dilation', 'opening', 'closing', 'gradient', 'top_hat', 'black_hat']:
            img = results[op_name]
            black_px = np.count_nonzero(img == 0)
            white_px = np.count_nonzero(img == 255)

            # Calculate change
            if op_name != 'gradient' and op_name != 'top_hat' and op_name != 'black_hat':
                change = ((black_px - original_black) / original_black * 100) if original_black > 0 else 0
            else:
                change = 0

            print(f"{op_name.replace('_', ' ').title():<15} {black_px:<15} {white_px:<15} {change:>10.1f}%")

        # Detailed effect analysis
        print("\n" + "="*70)
        print("OPERATION EFFECTS SUMMARY")
        print("="*70)

        erosion_black = np.count_nonzero(results['erosion'] == 0)
        dilation_black = np.count_nonzero(results['dilation'] == 0)
        opening_black = np.count_nonzero(results['opening'] == 0)
        closing_black = np.count_nonzero(results['closing'] == 0)

        print(f"\n1. EROSION:")
        print(f"   - Shrinks white regions / expands black regions")
        print(f"   - Black pixels change: {original_black} → {erosion_black} "
              f"({(erosion_black - original_black):+d})")
        print(f"   - Use case: Remove small white noise, thin lines")

        print(f"\n2. DILATION:")
        print(f"   - Expands white regions / shrinks black regions")
        print(f"   - Black pixels change: {original_black} → {dilation_black} "
              f"({(dilation_black - original_black):+d})")
        print(f"   - Use case: Fill small black holes, connect nearby objects")

        print(f"\n3. OPENING (Erosion → Dilation):")
        print(f"   - Removes small white objects / opens small black cracks")
        print(f"   - Black pixels: {opening_black}")
        print(f"   - Use case: Denoise, remove salt noise")

        print(f"\n4. CLOSING (Dilation → Erosion):")
        print(f"   - Fills small black holes / closes small white gaps")
        print(f"   - Black pixels: {closing_black}")
        print(f"   - Use case: Close holes in text/objects")

        gradient = results['gradient']
        gradient_black = np.count_nonzero(gradient == 0)
        print(f"\n5. MORPHOLOGICAL GRADIENT (Dilation - Erosion):")
        print(f"   - Extracts edges/boundaries between black and white regions")
        print(f"   - Non-zero pixels: {gradient_black}")
        print(f"   - Use case: Edge detection, boundary extraction")

        top_hat = results['top_hat']
        top_hat_black = np.count_nonzero(top_hat == 0)
        print(f"\n6. TOP HAT (Original - Opening):")
        print(f"   - Extracts small objects removed by opening")
        print(f"   - Non-zero pixels: {top_hat_black}")
        print(f"   - Use case: Extract fine details, small objects")

        black_hat = results['black_hat']
        black_hat_black = np.count_nonzero(black_hat == 0)
        print(f"\n7. BLACK HAT (Closing - Original):")
        print(f"   - Extracts dark spots in light areas")
        print(f"   - Non-zero pixels: {black_hat_black}")
        print(f"   - Use case: Find surface imperfections")

    def run_complete_analysis(self, kernel_size=5, use_custom=None, threshold=127):
        """
        Execute complete morphological analysis with all kernel types.

        Can use either a generated synthetic mask or a custom image.

        Args:
            kernel_size (int): Size of structuring elements (default: 5).
            use_custom (bool): If True, use custom image; if False, generate mask.
                If None, automatically detect based on custom_image_path (default: None).
            threshold (int): Threshold value for binarizing custom images (default: 127).

        Prints:
            - Progress for each kernel type
            - Summary of all analyses
            - Source of input (generated or custom)

        Raises:
            ValueError: If use_custom is True but no custom image path provided.
        """
        print("\n" + "="*70)
        print("MORPHOLOGICAL VISUAL COMPARISON EXERCISE")
        print("="*70)

        # Determine image source
        if use_custom is None:
            use_custom = self.custom_image_path is not None

        if use_custom:
            if not self.custom_image_path:
                raise ValueError("Custom image path not provided. "
                               "Set custom_image_path in __init__ or provide use_custom=False")
            print(f"\n→ Using CUSTOM IMAGE: {self.custom_image_path}")
            clean_mask, corrupted_mask = self.load_custom_image(self.custom_image_path, threshold)
        else:
            print(f"\n→ Using GENERATED SYNTHETIC MASK")
            clean_mask, corrupted_mask = self.create_corrupted_binary_mask()

        # Save original masks
        clean_path = os.path.join(self.output_dir, 'clean_mask.png')
        corrupted_path = os.path.join(self.output_dir, 'corrupted_mask.png')
        cv2.imwrite(clean_path, clean_mask)
        cv2.imwrite(corrupted_path, corrupted_mask)
        print(f"\n✓ Masks saved:")
        print(f"  - Clean: {clean_path}")
        print(f"  - Corrupted: {corrupted_path}")

        # Create structuring elements
        kernels = self.create_structuring_elements(kernel_size)

        # Process with each kernel
        for kernel_name, kernel in kernels.items():
            print(f"\n" + "="*70)
            print(f"PROCESSING WITH {kernel_name.upper()} KERNEL")
            print("="*70)

            # Apply morphological operations
            results = self.apply_morphological_operations(corrupted_mask, kernel)

            # Create dashboard
            self.create_8panel_dashboard(corrupted_mask, results, kernel_name, kernel_size)

            # Create analysis
            self.create_comparison_table(corrupted_mask, results, kernel_name, kernel_size)

        print("\n" + "="*70)
        print("✓ MORPHOLOGICAL VISUAL COMPARISON COMPLETE!")
        print("="*70)
        print(f"\nAll visualizations saved to: {self.output_dir}")


def main():
    """
    Main entry point for morphological visual comparison exercise.

    Creates comprehensive visualizations of morphological operations
    with different structuring elements.

    Supports both synthetic masks (generated) and custom images.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Morphological Operations Visual Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:

1. Generate synthetic corrupted mask:
   python morphological_visual_comparison.py

2. Use custom image:
   python morphological_visual_comparison.py --custom /path/to/image.jpg

3. Custom image with threshold:
   python morphological_visual_comparison.py --custom document.png --threshold 150

4. Custom kernel size:
   python morphological_visual_comparison.py --custom image.jpg --kernel-size 7

5. Custom output directory:
   python morphological_visual_comparison.py --custom image.jpg --output-dir ./results

IMAGE REQUIREMENTS:
- Format: PNG, JPG, BMP, TIFF, etc. (anything OpenCV can read)
- Size: Any (recommended: 300-600 pixels for clarity)
- Color: Color or grayscale (both are converted to binary)
- Content: Text, shapes, or any image to analyze

THRESHOLD GUIDE:
- 100-120: Good for dark images
- 127-150: Standard (default: 127)
- 150-200: Good for light images
        """
    )

    parser.add_argument('--custom', type=str, help='Path to custom image for analysis')
    parser.add_argument('--threshold', type=int, default=127,
                       help='Binarization threshold (0-255, default: 127)')
    parser.add_argument('--kernel-size', type=int, default=5,
                       help='Structuring element size (default: 5)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Define output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'output_morphology')

    # Create and run comparison
    comparison = MorphologicalVisualComparison(output_dir, custom_image_path=args.custom)

    print("\n" + "="*70)
    print("MORPHOLOGICAL OPERATIONS VISUAL COMPARISON")
    print("="*70)

    if args.custom:
        print(f"\nConfiguration:")
        print(f"  Input: Custom image")
        print(f"  Path: {args.custom}")
        print(f"  Threshold: {args.threshold}")
        print(f"  Kernel size: {args.kernel_size}×{args.kernel_size}")
        print(f"  Output: {output_dir}")
    else:
        print(f"\nConfiguration:")
        print(f"  Input: Generated synthetic mask")
        print(f"  Kernel size: {args.kernel_size}×{args.kernel_size}")
        print(f"  Output: {output_dir}")

    # Run analysis
    comparison.run_complete_analysis(kernel_size=args.kernel_size,
                                     use_custom=args.custom is not None,
                                     threshold=args.threshold)


if __name__ == "__main__":
    main()
