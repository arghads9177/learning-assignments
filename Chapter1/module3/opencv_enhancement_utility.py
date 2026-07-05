"""
OpenCV Enhancement Utility - Hands-on Exercises

This script demonstrates advanced image enhancement techniques for correcting
underexposed (dark) images:
- Global histogram equalization on grayscale images
- Adaptive histogram equalization using CLAHE on LAB color space
- Adaptive histogram equalization using CLAHE on HSV color space
"""

import cv2
import numpy as np
import os


def create_dark_sample_image(image_path):
    """
    Create an underexposed (dark) sample image for demonstration.

    Generates a dark image with various shapes and colors to simulate
    an underexposed photo. Creates the image only if it doesn't already exist.

    Args:
        image_path (str): Path where the sample dark image will be saved.

    Returns:
        ndarray: The created underexposed image.

    Prints:
        - Confirmation of image creation with path
        - Image dimensions and characteristics
    """
    if os.path.exists(image_path):
        return cv2.imread(image_path)

    # Create a blank image with dark gray background (simulating underexposed photo)
    img = np.ones((400, 600, 3), dtype=np.uint8) * 40

    # Add colored rectangles with reduced brightness
    cv2.rectangle(img, (50, 50), (200, 150), (0, 100, 0), -1)  # Dark green
    cv2.rectangle(img, (250, 100), (450, 250), (100, 0, 0), -1)  # Dark blue
    cv2.rectangle(img, (100, 250), (300, 350), (0, 0, 100), -1)  # Dark red

    # Add circles with reduced brightness
    cv2.circle(img, (150, 200), 40, (0, 80, 120), -1)  # Dark orange
    cv2.circle(img, (500, 150), 50, (100, 100, 0), -1)  # Dark cyan

    # Add text
    cv2.putText(img, "Underexposed Image", (100, 380), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (50, 50, 50), 2)

    cv2.imwrite(image_path, img)
    return img


def load_dark_image(image_path):
    """
    Load a dark/underexposed image and convert to grayscale.

    Loads the image in both BGR (color) and grayscale formats.
    If the image doesn't exist, creates a sample dark image.

    Args:
        image_path (str): Path to the dark image file.

    Returns:
        tuple: (img_bgr, img_gray) where:
            - img_bgr (ndarray): Image in BGR color format
            - img_gray (ndarray): Grayscale version of the image

    Prints:
        - Confirmation of image loading
        - Image dimensions and shape information
    """
    print("\n" + "="*60)
    print("LOADING UNDEREXPOSED IMAGE")
    print("="*60)

    # Create sample if doesn't exist
    img = create_dark_sample_image(image_path)

    # Load in BGR format
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from: {image_path}")
    print(f"✓ Dark image loaded: {image_path}")
    print(f"  Shape: {img_bgr.shape}")

    # Load in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Failed to load grayscale image")
    print(f"✓ Grayscale image loaded, Shape: {img_gray.shape}")
    print(f"  Mean brightness: {img_gray.mean():.2f} (low = underexposed)")

    return img_bgr, img_gray


def display_image(title, img, wait_key=True):
    """
    Display a single image in an OpenCV window.

    Args:
        title (str): Window title for the image.
        img (ndarray): Image to display.
        wait_key (bool): Whether to wait for key press before continuing.
    """
    cv2.imshow(title, img)
    if wait_key:
        print(f"  (Displaying: {title} - Press any key to continue...)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def global_histogram_equalization(img_gray, output_dir):
    """
    Apply global histogram equalization on grayscale image.

    Uses cv2.equalizeHist() to enhance contrast by redistributing pixel
    intensity values. This technique is simple but can cause over-brightening
    in certain regions. Works on single-channel (grayscale) images.

    Global equalization formula:
    - Computes histogram of input image
    - Creates cumulative distribution function (CDF)
    - Maps old pixel values to new values using CDF

    Args:
        img_gray (ndarray): Grayscale image (single-channel).
        output_dir (str): Directory to save the equalized image.

    Returns:
        ndarray: Histogram-equalized grayscale image.

    Prints:
        - Confirmation of equalization
        - Before/after brightness statistics
        - File save confirmation
    """
    print("\n" + "="*60)
    print("1. GLOBAL HISTOGRAM EQUALIZATION (Grayscale)")
    print("="*60)

    print(f"Original Image Statistics:")
    print(f"  Mean: {img_gray.mean():.2f}")
    print(f"  Std Dev: {img_gray.std():.2f}")
    print(f"  Min: {img_gray.min()}, Max: {img_gray.max()}")

    # Apply histogram equalization
    equalized = cv2.equalizeHist(img_gray)

    print(f"\n✓ Global Histogram Equalization Applied")
    print(f"Equalized Image Statistics:")
    print(f"  Mean: {equalized.mean():.2f}")
    print(f"  Std Dev: {equalized.std():.2f}")
    print(f"  Min: {equalized.min()}, Max: {equalized.max()}")

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '1_global_histogram_equalization.jpg')
    cv2.imwrite(output_path, equalized)
    print(f"✓ Equalized image saved: {output_path}")

    # Display comparison
    display_comparison(img_gray, equalized, "Original (Dark)", "Global Histogram Equalization")

    return equalized


def clahe_on_lab(img_bgr, output_dir):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to LAB color space.

    Converts BGR image to LAB (Lightness, a*, b*) color space, applies CLAHE
    only to the L (Lightness) channel to enhance brightness while preserving
    color information, then converts back to BGR.

    LAB color space separates lightness from color components, making it ideal
    for brightness adjustments without affecting hue or saturation.

    Args:
        img_bgr (ndarray): Image in BGR color format.
        output_dir (str): Directory to save the enhanced image.

    Returns:
        ndarray: Enhanced image in BGR format.

    Prints:
        - LAB color space info
        - CLAHE parameters (clip limit, tile grid size)
        - Before/after L channel statistics
        - File save confirmation
    """
    print("\n" + "="*60)
    print("2. CLAHE ON LAB COLOR SPACE (Lightness Channel)")
    print("="*60)

    # Convert BGR to LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    print(f"✓ Converted BGR to LAB color space")
    print(f"  LAB Shape: {lab.shape}")

    # Split LAB channels
    l, a, b = cv2.split(lab)

    print(f"\nOriginal L Channel (Lightness) Statistics:")
    print(f"  Mean: {l.mean():.2f}")
    print(f"  Std Dev: {l.std():.2f}")
    print(f"  Min: {l.min()}, Max: {l.max()}")

    # Create CLAHE object
    # clipLimit: Threshold for contrast limiting (default 2.0)
    # tileGridSize: Size of grid cells (default 8×8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print(f"\nCLAHE Parameters:")
    print(f"  Clip Limit: 2.0")
    print(f"  Tile Grid Size: 8×8")

    # Apply CLAHE only to L channel
    l_clahe = clahe.apply(l)

    print(f"\nEnhanced L Channel (Lightness) Statistics:")
    print(f"  Mean: {l_clahe.mean():.2f}")
    print(f"  Std Dev: {l_clahe.std():.2f}")
    print(f"  Min: {l_clahe.min()}, Max: {l_clahe.max()}")

    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])
    print(f"\n✓ L channel enhanced, channels merged")

    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    print(f"✓ Converted back to BGR color space")

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '2_clahe_lab_lightness.jpg')
    cv2.imwrite(output_path, enhanced)
    print(f"✓ Enhanced image saved: {output_path}")

    # Display comparison
    display_comparison(img_bgr, enhanced, "Original (Dark)", "CLAHE on LAB (L Channel)")

    return enhanced


def clahe_on_hsv(img_bgr, output_dir):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to HSV color space.

    Converts BGR image to HSV (Hue, Saturation, Value) color space, applies CLAHE
    only to the V (Value/Brightness) channel to enhance brightness while preserving
    hue and saturation, then converts back to BGR.

    HSV color space separates brightness (V) from color information (H and S),
    allowing selective brightness enhancement without color shift.

    Args:
        img_bgr (ndarray): Image in BGR color format.
        output_dir (str): Directory to save the enhanced image.

    Returns:
        ndarray: Enhanced image in BGR format.

    Prints:
        - HSV color space info
        - CLAHE parameters (clip limit, tile grid size)
        - Before/after V channel statistics
        - File save confirmation
    """
    print("\n" + "="*60)
    print("3. CLAHE ON HSV COLOR SPACE (Value Channel)")
    print("="*60)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    print(f"✓ Converted BGR to HSV color space")
    print(f"  HSV Shape: {hsv.shape}")

    # Split HSV channels
    h, s, v = cv2.split(hsv)

    print(f"\nOriginal V Channel (Value/Brightness) Statistics:")
    print(f"  Mean: {v.mean():.2f}")
    print(f"  Std Dev: {v.std():.2f}")
    print(f"  Min: {v.min()}, Max: {v.max()}")

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print(f"\nCLAHE Parameters:")
    print(f"  Clip Limit: 2.0")
    print(f"  Tile Grid Size: 8×8")

    # Apply CLAHE only to V channel
    v_clahe = clahe.apply(v)

    print(f"\nEnhanced V Channel (Value/Brightness) Statistics:")
    print(f"  Mean: {v_clahe.mean():.2f}")
    print(f"  Std Dev: {v_clahe.std():.2f}")
    print(f"  Min: {v_clahe.min()}, Max: {v_clahe.max()}")

    # Merge channels back
    hsv_clahe = cv2.merge([h, s, v_clahe])
    print(f"\n✓ V channel enhanced, channels merged")

    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    print(f"✓ Converted back to BGR color space")

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '3_clahe_hsv_value.jpg')
    cv2.imwrite(output_path, enhanced)
    print(f"✓ Enhanced image saved: {output_path}")

    # Display comparison
    display_comparison(img_bgr, enhanced, "Original (Dark)", "CLAHE on HSV (V Channel)")

    return enhanced


def compare_all_methods(img_bgr, enhanced_gray, enhanced_lab, enhanced_hsv):
    """
    Create and save a comparison of all enhancement methods.

    Generates side-by-side visualizations comparing the original underexposed
    image with results from all three enhancement techniques.

    Args:
        img_bgr (ndarray): Original BGR image.
        enhanced_gray (ndarray): Result from global histogram equalization.
        enhanced_lab (ndarray): Result from CLAHE on LAB.
        enhanced_hsv (ndarray): Result from CLAHE on HSV.

    Prints:
        - Confirmation of comparison visualization
        - Summary of all methods
        - File save confirmation
    """
    print("\n" + "="*60)
    print("COMPARISON OF ALL ENHANCEMENT METHODS")
    print("="*60)

    # Create comparison images by displaying side by side
    print(f"\nEnhancement Methods Summary:")
    print(f"  1. Global Histogram Equalization (Grayscale)")
    print(f"     - Simple and fast")
    print(f"     - Can cause over-brightening and artifacts")
    print(f"     - Best for grayscale images")
    print(f"\n  2. CLAHE on LAB Color Space (L Channel)")
    print(f"     - Adaptive and contrast-limited")
    print(f"     - Preserves color information")
    print(f"     - Better result for color images")
    print(f"\n  3. CLAHE on HSV Color Space (V Channel)")
    print(f"     - Adaptive and contrast-limited")
    print(f"     - Preserves hue and saturation")
    print(f"     - Good color preservation")

    # Display all results
    print(f"\nDisplaying all enhancement methods...")
    cv2.imshow("Original (Dark)", img_bgr)
    cv2.imshow("Method 1: Global Histogram Eq.", cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR))
    cv2.imshow("Method 2: CLAHE on LAB", enhanced_lab)
    cv2.imshow("Method 3: CLAHE on HSV", enhanced_hsv)

    print("  (Press any key to continue...)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_comparison(original, enhanced, title1, title2):
    """
    Display original and enhanced images for comparison.

    Helper function to visualize enhancement results side by side.

    Args:
        original (ndarray): Original image.
        enhanced (ndarray): Enhanced image.
        title1 (str): Title for original image window.
        title2 (str): Title for enhanced image window.
    """
    cv2.imshow(title1, original)
    cv2.imshow(title2, enhanced)
    print(f"  (Comparing: {title1} vs {title2} - Press any key to continue...)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_summary(output_dir):
    """
    Print a summary of all enhancement techniques.

    Args:
        output_dir (str): Output directory path.

    Prints:
        - Enhancement technique descriptions
        - When to use each method
        - Output file paths
    """
    print("\n" + "="*60)
    print("ENHANCEMENT TECHNIQUES SUMMARY")
    print("="*60)

    print(f"\n1. GLOBAL HISTOGRAM EQUALIZATION")
    print(f"   Purpose: Stretch histogram to full range [0, 255]")
    print(f"   Use Case: Grayscale images with poor contrast")
    print(f"   Pros: Simple, fast, effective for grayscale")
    print(f"   Cons: Can cause artifacts, limited color preservation")

    print(f"\n2. CLAHE ON LAB (Lightness Channel)")
    print(f"   Purpose: Adaptive contrast enhancement on perceptual lightness")
    print(f"   Use Case: Color images with uneven lighting")
    print(f"   Pros: Better color preservation, prevents over-brightening")
    print(f"   Cons: More complex, slower than global equalization")

    print(f"\n3. CLAHE ON HSV (Value Channel)")
    print(f"   Purpose: Adaptive brightness enhancement preserving hue/saturation")
    print(f"   Use Case: Color images requiring natural color appearance")
    print(f"   Pros: Excellent color preservation, adaptive enhancement")
    print(f"   Cons: More complex, computational overhead")

    print(f"\nRecommendations:")
    print(f"  - For general grayscale enhancement: Use Method 1")
    print(f"  - For perceptually uniform color: Use Method 2 (LAB)")
    print(f"  - For natural color appearance: Use Method 3 (HSV)")

    print(f"\nOutput files saved to: {output_dir}")


def main():
    """
    Execute all image enhancement exercises in sequence.

    Orchestrates the complete workflow:
    1. Loads an underexposed (dark) image
    2. Applies global histogram equalization on grayscale
    3. Applies CLAHE on LAB color space (L channel)
    4. Applies CLAHE on HSV color space (V channel)
    5. Displays comparison of all methods
    6. Prints comprehensive summary

    Prints:
        - Progress headers for each exercise section
        - Enhancement parameters and statistics
        - Output directory path
        - Overall completion confirmation
    """
    print("\n" + "="*60)
    print("OPENCV ENHANCEMENT UTILITY - HANDS-ON EXERCISES")
    print("="*60)

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # image_path = os.path.join(current_dir, 'dark_sample_image.jpg')
    image_path = os.path.join(current_dir, 'dark_sample_3.jpg')
    output_dir = os.path.join(current_dir, 'output_enhancements')

    # 1. Load dark image
    img_bgr, img_gray = load_dark_image(image_path)

    # 2. Global histogram equalization on grayscale
    enhanced_gray = global_histogram_equalization(img_gray, output_dir)

    # 3. CLAHE on LAB color space
    enhanced_lab = clahe_on_lab(img_bgr, output_dir)

    # 4. CLAHE on HSV color space
    enhanced_hsv = clahe_on_hsv(img_bgr, output_dir)

    # 5. Compare all methods
    compare_all_methods(img_bgr, enhanced_gray, enhanced_lab, enhanced_hsv)

    # 6. Print summary
    print_summary(output_dir)

    print("\n" + "="*60)
    print("✓ ALL ENHANCEMENT EXERCISES COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
