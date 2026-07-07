import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime


class LogoOverlayUtility:
    """
    Logo overlay utility for seamlessly blending irregular logos onto background images.

    This utility implements a sophisticated logo-blending pipeline that preserves logo geometry
    while eliminating rectangular framing artifacts. It uses binary masking, inverse operations,
    and bitwise image addition to create natural-looking overlays.
    """

    def __init__(self, output_dir='./output_logo_overlay'):
        """
        Initialize the LogoOverlayUtility with output directory configuration.

        Args:
            output_dir (str): Directory to save analysis outputs and visualizations.
                Default is './output_logo_overlay'.

        Returns:
            None

        Notes:
            - Creates output directory if it doesn't exist
            - All output files are saved relative to script location, not working directory
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, os.path.basename(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_test_images(self):
        """
        Generate synthetic test images: a background canvas and a logo with transparency.

        Args:
            None

        Returns:
            tuple: (background_img, logo_img) where:
                - background_img: 800×600 BGR image with gradient and pattern
                - logo_img: 300×300 BGR image with colored star logo on white background

        Notes:
            - Background includes gradient fill and geometric shapes for visual interest
            - Logo features a blue star shape on white background for clear isolation
            - Both images are suitable for testing the overlay pipeline
        """
        height, width = 600, 800

        # Create background canvas
        background = np.zeros((height, width, 3), dtype=np.uint8)
        background[:] = (200, 180, 160)  # Beige background

        # Add gradient overlay to background
        for i in range(height):
            background[i, :] = (
                int(200 - (i / height) * 50),
                int(180 - (i / height) * 40),
                int(160 - (i / height) * 30)
            )

        # Add decorative shapes on background
        cv2.rectangle(background, (50, 50), (300, 200), (100, 150, 200), -1)
        cv2.circle(background, (650, 150), 80, (150, 100, 50), -1)
        cv2.rectangle(background, (400, 350), (750, 550), (120, 180, 100), -1)

        # Create logo image with star shape
        logo_size = 300
        logo = np.ones((logo_size, logo_size, 3), dtype=np.uint8) * 255  # White background

        # Draw blue star
        center = (logo_size // 2, logo_size // 2)
        star_points = []
        for i in range(10):
            angle = i * np.pi / 5
            if i % 2 == 0:
                radius = 80
            else:
                radius = 30
            x = int(center[0] + radius * np.sin(angle))
            y = int(center[1] - radius * np.cos(angle))
            star_points.append([x, y])

        star_array = np.array(star_points, dtype=np.int32)
        cv2.fillPoly(logo, [star_array], (200, 100, 50))  # Blue star
        cv2.polylines(logo, [star_array], True, (50, 50, 200), 2)  # Dark blue outline

        return background, logo

    def resize_logo_if_needed(self, logo_img, background_img, threshold_percentage=30):
        """
        Resize logo if it exceeds specified percentage of background image area.

        Args:
            logo_img (np.ndarray): Logo image in BGR format (height, width, 3).
            background_img (np.ndarray): Background image in BGR format.
            threshold_percentage (float): Maximum allowed logo area as % of background.
                Default is 30. Logo is resized to this percentage if exceeded.

        Returns:
            np.ndarray: Resized logo image (if needed) or original logo.

        Notes:
            - Calculates logo area percentage relative to background dimensions
            - Resizes to exactly threshold_percentage of background if exceeded
            - Uses cv2.INTER_AREA for downsampling quality
            - Maintains aspect ratio during resize
            - threshold_percentage valid range: 1-99 (typical: 20-40)
        """
        logo_h, logo_w = logo_img.shape[:2]
        bg_h, bg_w = background_img.shape[:2]

        logo_percentage = (logo_h * logo_w) / (bg_h * bg_w) * 100

        if logo_percentage > threshold_percentage:
            target_area = (bg_h * bg_w) * (threshold_percentage / 100)
            scale_factor = np.sqrt(target_area / (logo_h * logo_w))
            new_w = int(logo_w * scale_factor)
            new_h = int(logo_h * scale_factor)
            resized = cv2.resize(logo_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized

        return logo_img

    def isolate_roi(self, background_img, logo_img, position=(100, 100)):
        """
        Extract target ROI (Region of Interest) from background matching logo dimensions.

        Args:
            background_img (np.ndarray): Background image in BGR format.
            logo_img (np.ndarray): Logo image to determine ROI size.
            position (tuple): (x, y) coordinates for ROI top-left corner.

        Returns:
            np.ndarray: Extracted ROI patch from background matching logo dimensions.

        Notes:
            - ROI dimensions match logo dimensions exactly
            - Position is clamped to ensure ROI stays within background bounds
            - Returns a copy of the background region for safe modification
        """
        logo_h, logo_w = logo_img.shape[:2]
        bg_h, bg_w = background_img.shape[:2]

        x, y = position
        x = max(0, min(x, bg_w - logo_w))
        y = max(0, min(y, bg_h - logo_h))

        roi = background_img[y:y + logo_h, x:x + logo_w].copy()
        return roi, (x, y)

    def create_binary_mask(self, logo_img):
        """
        Convert logo to grayscale and threshold to create precise binary mask.

        Args:
            logo_img (np.ndarray): Logo image in BGR format.

        Returns:
            np.ndarray: Binary mask (uint8) where logo objects are 255 and background is 0.

        Notes:
            - Converts BGR to grayscale using cv2.cvtColor
            - Uses Otsu's thresholding (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            - Inversion ensures non-white pixels (logo) become 255 in mask
            - Produces high-contrast binary mask suitable for bitwise operations
        """
        gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    def create_inverse_mask(self, mask):
        """
        Create inverse of binary mask using bitwise NOT operation.

        Args:
            mask (np.ndarray): Binary mask (uint8) where 255 = foreground, 0 = background.

        Returns:
            np.ndarray: Inverse mask where 0 = foreground, 255 = background.

        Notes:
            - Uses cv2.bitwise_not for efficient inversion
            - Inverse mask is used to zero out logo geometry in background
            - Result is uint8 array with same dimensions as input
        """
        inverse_mask = cv2.bitwise_not(mask)
        return inverse_mask

    def clear_logo_from_roi(self, roi, inverse_mask):
        """
        Black out the exact logo geometry from the background ROI using inverse mask.

        Args:
            roi (np.ndarray): Background ROI patch in BGR format (height, width, 3).
            inverse_mask (np.ndarray): Inverse binary mask (0 = clear, 255 = keep).

        Returns:
            np.ndarray: Modified ROI with logo area blacked out (logo geometry cleared).

        Notes:
            - Applies cv2.bitwise_and(roi, roi, mask=inverse_mask)
            - Inverse mask operation clears exact pixels where logo will be placed
            - Preserves background colors where logo won't overlap
            - Result prepares background for seamless logo blending
        """
        cleared_roi = cv2.bitwise_and(roi, roi, mask=inverse_mask)
        return cleared_roi

    def isolate_logo_colors(self, logo_img, mask):
        """
        Extract logo color channels while turning logo background black using mask.

        Args:
            logo_img (np.ndarray): Logo image in BGR format.
            mask (np.ndarray): Binary mask (255 = logo, 0 = background).

        Returns:
            np.ndarray: Logo image with isolated colors; background is black (0,0,0).

        Notes:
            - Applies cv2.bitwise_and(logo_img, logo_img, mask=mask)
            - Mask determines which pixels to keep from original logo
            - Background pixels (where mask is 0) become black
            - Result is ready to be added to cleared background ROI
        """
        isolated_logo = cv2.bitwise_and(logo_img, logo_img, mask=mask)
        return isolated_logo

    def blend_images(self, cleared_roi, isolated_logo):
        """
        Sum the cleared background patch and isolated logo using bitwise addition.

        Args:
            cleared_roi (np.ndarray): Background ROI with logo area blacked out.
            isolated_logo (np.ndarray): Logo with isolated colors; background is black.

        Returns:
            np.ndarray: Blended composite image combining background and logo.

        Notes:
            - Uses cv2.add for proper uint8 addition with overflow handling
            - Adding to black background (0,0,0) is non-destructive
            - Result is saturated to [0, 255] range for valid pixel values
            - Composite appears naturally integrated without artifacts
        """
        blended = cv2.add(cleared_roi, isolated_logo)
        return blended

    def inject_composite(self, background_img, composite_patch, position):
        """
        Inject the modified composite patch back into the original background matrix.

        Args:
            background_img (np.ndarray): Original background image in BGR format.
            composite_patch (np.ndarray): Composite patch with blended logo.
            position (tuple): (x, y) coordinates for injection location.

        Returns:
            np.ndarray: Updated background image with embedded composite patch.

        Notes:
            - Creates a copy of background to avoid modifying original
            - Directly assigns composite_patch to corresponding background region
            - Position is same as ROI extraction for consistent placement
            - Result is the final overlay image ready for output
        """
        result = background_img.copy()
        x, y = position
        patch_h, patch_w = composite_patch.shape[:2]
        result[y:y + patch_h, x:x + patch_w] = composite_patch
        return result

    def run_analysis(self, background_path=None, logo_path=None, position=(100, 100),
                     resize_threshold=30):
        """
        Execute complete logo overlay pipeline with visualization and reporting.

        Args:
            background_path (str): Path to background image file. If None, generates synthetic.
            logo_path (str): Path to logo image file. If None, generates synthetic.
            position (tuple): (x, y) coordinates for logo placement on background.
            resize_threshold (float): Maximum logo area as % of background before resizing.
                Default is 30. Valid range: 1-99. Typical values: 20-40.

        Returns:
            np.ndarray: Final composite image with logo overlaid on background.

        Notes:
            - Loads or generates test images
            - Resizes logo if exceeds resize_threshold percentage of background area
            - Creates binary mask and inverse mask
            - Clears logo region from background ROI
            - Isolates logo colors
            - Blends images using addition
            - Injects composite back into background
            - Saves all intermediate images and creates visualization dashboard
            - Generates detailed analysis report
        """
        start_time = time.time()

        # Load or generate images
        if background_path and os.path.exists(background_path):
            background = cv2.imread(background_path)
            if background is None:
                background, logo = self.generate_test_images()
        else:
            background, logo = self.generate_test_images()

        if logo_path and os.path.exists(logo_path):
            logo = cv2.imread(logo_path)
            if logo is None:
                _, logo = self.generate_test_images()
        else:
            _, logo = self.generate_test_images()

        # Save original images
        cv2.imwrite(os.path.join(self.output_dir, 'background_original.png'), background)
        cv2.imwrite(os.path.join(self.output_dir, 'logo_original.png'), logo)

        # Resize logo if needed
        logo = self.resize_logo_if_needed(logo, background, resize_threshold)
        cv2.imwrite(os.path.join(self.output_dir, 'logo_resized.png'), logo)

        # Isolate ROI from background
        roi, actual_position = self.isolate_roi(background, logo, position)
        cv2.imwrite(os.path.join(self.output_dir, 'roi_extracted.png'), roi)

        # Create masks
        binary_mask = self.create_binary_mask(logo)
        cv2.imwrite(os.path.join(self.output_dir, 'binary_mask.png'), binary_mask)

        inverse_mask = self.create_inverse_mask(binary_mask)
        cv2.imwrite(os.path.join(self.output_dir, 'inverse_mask.png'), inverse_mask)

        # Clear logo from ROI
        cleared_roi = self.clear_logo_from_roi(roi, inverse_mask)
        cv2.imwrite(os.path.join(self.output_dir, 'cleared_roi.png'), cleared_roi)

        # Isolate logo colors
        isolated_logo = self.isolate_logo_colors(logo, binary_mask)
        cv2.imwrite(os.path.join(self.output_dir, 'isolated_logo.png'), isolated_logo)

        # Blend images
        blended_composite = self.blend_images(cleared_roi, isolated_logo)
        cv2.imwrite(os.path.join(self.output_dir, 'blended_composite.png'), blended_composite)

        # Inject composite into background
        final_image = self.inject_composite(background, blended_composite, actual_position)
        cv2.imwrite(os.path.join(self.output_dir, 'final_overlay.png'), final_image)

        # Create visualization dashboard
        self.create_overlay_dashboard(
            background, logo, binary_mask, inverse_mask,
            cleared_roi, isolated_logo, blended_composite, final_image
        )

        # Print statistics
        processing_time = time.time() - start_time
        self.print_statistics(background.shape, logo.shape, actual_position, processing_time)

        # Save report
        self.save_report(background.shape, logo.shape, actual_position, processing_time)

        return final_image

    def create_overlay_dashboard(self, background, logo, binary_mask, inverse_mask,
                                 cleared_roi, isolated_logo, blended_composite, final_image):
        """
        Create comprehensive 4×2 visualization dashboard of the overlay pipeline.

        Args:
            background (np.ndarray): Original background image.
            logo (np.ndarray): Logo image.
            binary_mask (np.ndarray): Binary mask of logo.
            inverse_mask (np.ndarray): Inverse binary mask.
            cleared_roi (np.ndarray): Background ROI with logo area cleared.
            isolated_logo (np.ndarray): Logo with isolated colors.
            blended_composite (np.ndarray): Blended composite patch.
            final_image (np.ndarray): Final overlay result.

        Returns:
            None

        Notes:
            - Creates 4×2 grid showing pipeline progression
            - Row 1: Background + Logo
            - Row 2: Binary Mask + Inverse Mask
            - Row 3: Cleared ROI + Isolated Logo
            - Row 4: Blended Composite + Final Result
            - Saves as high-DPI PNG (150 DPI)
        """
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle('Logo Overlay Pipeline Visualization', fontsize=16, fontweight='bold')

        # Row 1: Input images
        axes[0, 0].imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Background Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(logo, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Logo Image')
        axes[0, 1].axis('off')

        # Row 2: Masks
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title('Binary Mask (Logo=255)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(inverse_mask, cmap='gray')
        axes[1, 1].set_title('Inverse Mask (Logo=0)')
        axes[1, 1].axis('off')

        # Row 3: Processing steps
        axes[2, 0].imshow(cv2.cvtColor(cleared_roi, cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title('Cleared ROI (Logo Area Blacked Out)')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(cv2.cvtColor(isolated_logo, cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title('Isolated Logo (Colors Only)')
        axes[2, 1].axis('off')

        # Row 4: Final results
        axes[3, 0].imshow(cv2.cvtColor(blended_composite, cv2.COLOR_BGR2RGB))
        axes[3, 0].set_title('Blended Composite')
        axes[3, 0].axis('off')

        axes[3, 1].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        axes[3, 1].set_title('Final Overlay Result')
        axes[3, 1].axis('off')

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, 'logo_overlay_dashboard.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()

    def print_statistics(self, bg_shape, logo_shape, position, processing_time):
        """
        Print overlay analysis statistics to console.

        Args:
            bg_shape (tuple): Background image shape (height, width, channels).
            logo_shape (tuple): Logo image shape (height, width, channels).
            position (tuple): (x, y) coordinates of logo placement.
            processing_time (float): Total processing time in seconds.

        Returns:
            None

        Notes:
            - Displays background and logo dimensions
            - Shows logo area percentage relative to background
            - Reports placement coordinates
            - Shows processing performance metrics
        """
        bg_h, bg_w = bg_shape[:2]
        logo_h, logo_w = logo_shape[:2]
        logo_percentage = (logo_h * logo_w) / (bg_h * bg_w) * 100

        print("\n" + "=" * 70)
        print("LOGO OVERLAY ANALYSIS - STATISTICS")
        print("=" * 70)
        print(f"Background Resolution: {bg_w}×{bg_h} pixels")
        print(f"Logo Dimensions: {logo_w}×{logo_h} pixels")
        print(f"Logo Area Coverage: {logo_percentage:.2f}% of background")
        print(f"Placement Position: ({position[0]}, {position[1]})")
        print(f"Processing Time: {processing_time:.3f} seconds")
        print("=" * 70 + "\n")

    def save_report(self, bg_shape, logo_shape, position, processing_time):
        """
        Save detailed overlay analysis report to text file.

        Args:
            bg_shape (tuple): Background image shape (height, width, channels).
            logo_shape (tuple): Logo image shape (height, width, channels).
            position (tuple): (x, y) coordinates of logo placement.
            processing_time (float): Total processing time in seconds.

        Returns:
            None

        Notes:
            - Creates structured text report in output directory
            - Includes timestamp, dimensions, and processing details
            - Documents pipeline step descriptions
            - Suitable for archival and analysis record-keeping
        """
        bg_h, bg_w = bg_shape[:2]
        logo_h, logo_w = logo_shape[:2]
        logo_percentage = (logo_h * logo_w) / (bg_h * bg_w) * 100

        report_path = os.path.join(self.output_dir, 'logo_overlay_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("LOGO OVERLAY UTILITY - DETAILED ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("IMAGE DIMENSIONS:\n")
            f.write(f"  Background Resolution: {bg_w}×{bg_h} pixels\n")
            f.write(f"  Logo Dimensions: {logo_w}×{logo_h} pixels\n")
            f.write(f"  Logo Area Coverage: {logo_percentage:.2f}% of background\n\n")

            f.write("PLACEMENT INFORMATION:\n")
            f.write(f"  Logo Position (Top-Left): ({position[0]}, {position[1]})\n")
            f.write(f"  Right Edge: {position[0] + logo_w} pixels\n")
            f.write(f"  Bottom Edge: {position[1] + logo_h} pixels\n\n")

            f.write("PROCESSING PIPELINE:\n")
            f.write("  1. Load or generate background and logo images\n")
            f.write("  2. Resize logo if area exceeds 30% of background\n")
            f.write("  3. Isolate ROI from background matching logo dimensions\n")
            f.write("  4. Create high-contrast binary mask from logo\n")
            f.write("  5. Create inverse mask using cv2.bitwise_not\n")
            f.write("  6. Clear logo geometry from background ROI\n")
            f.write("  7. Isolate logo colors (background becomes black)\n")
            f.write("  8. Blend cleared ROI and isolated logo using cv2.add\n")
            f.write("  9. Inject composite patch back into background\n\n")

            f.write("OUTPUT FILES:\n")
            f.write("  - background_original.png: Input background image\n")
            f.write("  - logo_original.png: Input logo image\n")
            f.write("  - logo_resized.png: Resized logo (if needed)\n")
            f.write("  - roi_extracted.png: Extracted background ROI\n")
            f.write("  - binary_mask.png: Logo binary mask (255=logo, 0=bg)\n")
            f.write("  - inverse_mask.png: Inverse mask (0=logo, 255=bg)\n")
            f.write("  - cleared_roi.png: Background ROI with logo area cleared\n")
            f.write("  - isolated_logo.png: Logo with isolated colors\n")
            f.write("  - blended_composite.png: Blended composite patch\n")
            f.write("  - final_overlay.png: Final result with logo overlaid\n")
            f.write("  - logo_overlay_dashboard.png: 4×2 pipeline visualization\n")
            f.write("  - logo_overlay_report.txt: This report\n\n")

            f.write("KEY CONCEPTS:\n")
            f.write("  Binary Masking: High-contrast mask isolates logo geometry precisely\n")
            f.write("  Inverse Mask: Used to clear logo area from background safely\n")
            f.write("  Bitwise Operations: cv2.bitwise_and selectively copies pixels\n")
            f.write("  Image Addition: cv2.add blends cleared background and isolated logo\n")
            f.write("  No Framing: Result contains no rectangular box artifacts\n\n")

            f.write(f"PERFORMANCE:\n")
            f.write(f"  Total Processing Time: {processing_time:.3f} seconds\n")
            f.write("=" * 70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Logo Overlay Utility - Seamlessly blend logos onto backgrounds'
    )
    parser.add_argument(
        '--background', type=str, default=None,
        help='Path to background image (JPG, PNG, BMP, TIFF)'
    )
    parser.add_argument(
        '--logo', type=str, default=None,
        help='Path to logo image (JPG, PNG, BMP, TIFF)'
    )
    parser.add_argument(
        '--position', type=int, nargs=2, default=[100, 100],
        help='Logo placement position (x, y) - default: 100 100'
    )
    parser.add_argument(
        '--resize-threshold', type=float, default=30,
        help='Max logo area as %% of background before resizing (default: 30, range: 1-99)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )

    args = parser.parse_args()

    if args.output_dir:
        overlay = LogoOverlayUtility(args.output_dir)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        overlay = LogoOverlayUtility(os.path.join(current_dir, 'output_logo_overlay'))

    overlay.run_analysis(
        background_path=args.background,
        logo_path=args.logo,
        position=tuple(args.position),
        resize_threshold=args.resize_threshold
    )


if __name__ == '__main__':
    main()
