"""
OpenCV Blueprint & Roadmap Hands-on Exercises

This script demonstrates advanced image transformation operations:
- Thumbnail generation with aspect ratio preservation
- Passport photo resizing with precise dimensions
- Image rotation using transformation matrices
- Perspective correction for 3D objects
- Region of Interest (ROI) extraction and manipulation
"""

import cv2
import numpy as np
import os


def create_sample_image(image_path):
    """
    Create a sample image for demonstration if it doesn't exist.

    Generates a sample image with various shapes and colors to use
    for transformation demonstrations.

    Args:
        image_path (str): Path where the sample image will be saved.

    Returns:
        ndarray: The created sample image.
    """
    if os.path.exists(image_path):
        return cv2.imread(image_path)

    # Create a blank image with white background
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Add colored rectangles
    cv2.rectangle(img, (50, 50), (200, 150), (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (250, 100), (450, 250), (255, 0, 0), -1)  # Blue
    cv2.rectangle(img, (100, 250), (300, 350), (0, 0, 255), -1)  # Red

    # Add circles
    cv2.circle(img, (150, 200), 40, (0, 165, 255), -1)  # Orange
    cv2.circle(img, (500, 150), 50, (255, 255, 0), -1)  # Cyan

    # Add text
    cv2.putText(img, "OpenCV Roadmap", (150, 380), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2)

    cv2.imwrite(image_path, img)
    return img


def thumbnail_generation(img, output_dir):
    """
    Generate a thumbnail by downscaling image with aspect ratio preservation.

    Uses cv2.INTER_AREA for high-quality downsampling. The thumbnail maintains
    the original image's aspect ratio while reducing dimensions to fit within
    a maximum size (e.g., 200x200 pixels).

    Args:
        img (ndarray): Input image to create thumbnail from.
        output_dir (str): Directory to save the thumbnail.

    Returns:
        ndarray: The generated thumbnail image.

    Prints:
        - Original image dimensions
        - Thumbnail dimensions
        - Aspect ratio information
        - File save confirmation
    """
    print("\n" + "="*60)
    print("1. THUMBNAIL GENERATION (Aspect Ratio Preservation)")
    print("="*60)

    original_height, original_width = img.shape[:2]
    print(f"Original Image Dimensions: {original_width}×{original_height}")

    # Define maximum thumbnail size
    max_dimension = 200
    aspect_ratio = original_width / original_height

    # Calculate new dimensions maintaining aspect ratio
    if original_width > original_height:
        thumbnail_width = max_dimension
        thumbnail_height = int(max_dimension / aspect_ratio)
    else:
        thumbnail_height = max_dimension
        thumbnail_width = int(max_dimension * aspect_ratio)

    print(f"Target Max Dimension: {max_dimension}px")
    print(f"Aspect Ratio: {aspect_ratio:.2f}")

    # Resize using cv2.INTER_AREA for downsampling
    thumbnail = cv2.resize(img, (thumbnail_width, thumbnail_height),
                          interpolation=cv2.INTER_AREA)

    print(f"✓ Thumbnail Created: {thumbnail_width}×{thumbnail_height}")
    print(f"  Aspect Ratio Preserved: {(thumbnail_width / thumbnail_height):.2f}")

    # Save thumbnail
    os.makedirs(output_dir, exist_ok=True)
    thumbnail_path = os.path.join(output_dir, 'thumbnail.jpg')
    cv2.imwrite(thumbnail_path, thumbnail)
    print(f"✓ Thumbnail saved: {thumbnail_path}")

    return thumbnail


def passport_photo_resizing(img, output_dir):
    """
    Resize and crop image to passport photo specifications.

    Extracts a region of interest (face area) and resizes it to exact
    passport photo dimensions. Standard passport size is typically
    2×2 inches (200×200 pixels at 100 DPI or 400×400 pixels at 200 DPI).

    Args:
        img (ndarray): Input image.
        output_dir (str): Directory to save the passport photo.

    Returns:
        ndarray: The resized passport photo.

    Prints:
        - Original dimensions
        - ROI extraction details
        - Final passport photo dimensions
        - Specifications in inches and pixels
    """
    print("\n" + "="*60)
    print("2. PASSPORT PHOTO RESIZING")
    print("="*60)

    original_height, original_width = img.shape[:2]
    print(f"Original Image: {original_width}×{original_height}")

    # Define ROI (Region of Interest) for face area
    # Assuming face is roughly in the center, extracting a square region
    roi_size = min(original_width, original_height) - 50
    start_x = (original_width - roi_size) // 2
    start_y = (original_height - roi_size) // 2

    # Extract ROI using NumPy array slicing
    roi = img[start_y:start_y + roi_size, start_x:start_x + roi_size]
    print(f"✓ ROI Extracted: {roi_size}×{roi_size} from center")
    print(f"  Extraction Coordinates: ({start_x}, {start_y})")

    # Passport photo specifications
    # Standard: 2×2 inches at 100 DPI = 200×200 pixels
    # High quality: 2×2 inches at 200 DPI = 400×400 pixels
    passport_size = 400  # 200 DPI standard
    print(f"\nPassport Photo Specifications:")
    print(f"  Dimensions: 2×2 inches (at 200 DPI)")
    print(f"  Pixel Size: {passport_size}×{passport_size}")

    # Resize ROI to passport dimensions
    passport_photo = cv2.resize(roi, (passport_size, passport_size),
                               interpolation=cv2.INTER_CUBIC)

    print(f"✓ Resized to Passport Dimensions: {passport_size}×{passport_size}")

    # Save passport photo
    passport_path = os.path.join(output_dir, 'passport_photo.jpg')
    cv2.imwrite(passport_path, passport_photo)
    print(f"✓ Passport photo saved: {passport_path}")

    return passport_photo


def image_rotation(img, output_dir):
    """
    Rotate an image using a transformation matrix.

    Creates a rotation matrix using cv2.getRotationMatrix2D with specified
    center point, rotation angle, and scale. Applies the transformation
    using cv2.warpAffine to generate rotated images at various angles.

    Args:
        img (ndarray): Input image.
        output_dir (str): Directory to save rotated images.

    Returns:
        dict: Dictionary containing rotated images at different angles.

    Prints:
        - Image dimensions
        - Rotation matrix details for each angle
        - Center point and scale information
    """
    print("\n" + "="*60)
    print("3. IMAGE ROTATION (Transformation Matrix)")
    print("="*60)

    height, width = img.shape[:2]
    print(f"Image Dimensions: {width}×{height}")

    # Define center point and rotation angles
    center = (width // 2, height // 2)
    scale = 1.0
    angles = [15, 30, 45, 90]
    rotated_images = {}

    print(f"\nRotation Configuration:")
    print(f"  Center Point: {center}")
    print(f"  Scale Factor: {scale}")

    for angle in angles:
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        print(f"\n✓ Rotation Matrix for {angle}° angle:")
        print(f"  Matrix Shape: {rotation_matrix.shape}")
        print(f"  Matrix:\n{rotation_matrix}")

        # Apply rotation using warpAffine
        rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
        rotated_images[angle] = rotated

        # Save rotated image
        rotated_path = os.path.join(output_dir, f'rotated_{angle}_degrees.jpg')
        cv2.imwrite(rotated_path, rotated)
        print(f"  ✓ Saved: {rotated_path}")

    return rotated_images


def perspective_correction(img, output_dir):
    """
    Apply perspective correction to an image.

    Finds 4 corner points representing the current perspective, maps them
    to target destination coordinates (rectangular), calculates the
    perspective transformation matrix using cv2.getPerspectiveTransform,
    and applies it via cv2.warpPerspective to correct the perspective.

    Args:
        img (ndarray): Input image.
        output_dir (str): Directory to save corrected image.

    Returns:
        ndarray: The perspective-corrected image.

    Prints:
        - Source and destination point coordinates
        - Perspective transformation matrix
        - Corrected image dimensions
    """
    print("\n" + "="*60)
    print("4. PERSPECTIVE CORRECTION")
    print("="*60)

    height, width = img.shape[:2]
    print(f"Original Image Dimensions: {width}×{height}")

    # Define source points (4 corners of the trapezoid)
    # Simulating a perspective-skewed image
    src_points = np.array([
        [50, 50],           # Top-left
        [width - 50, 30],   # Top-right (skewed)
        [0, height - 50],   # Bottom-left (skewed)
        [width, height - 30] # Bottom-right
    ], dtype=np.float32)

    # Define destination points (rectangular corners)
    dst_points = np.array([
        [0, 0],             # Top-left
        [width, 0],         # Top-right
        [0, height],        # Bottom-left
        [width, height]     # Bottom-right
    ], dtype=np.float32)

    print(f"\nSource Points (Skewed Perspective):")
    for i, point in enumerate(src_points):
        print(f"  Point {i+1}: ({point[0]:.1f}, {point[1]:.1f})")

    print(f"\nDestination Points (Rectangular):")
    for i, point in enumerate(dst_points):
        print(f"  Point {i+1}: ({point[0]:.1f}, {point[1]:.1f})")

    # Calculate perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    print(f"\n✓ Perspective Transformation Matrix (3×3):")
    print(f"  Matrix Shape: {perspective_matrix.shape}")
    print(f"  Matrix:\n{perspective_matrix}")

    # Apply perspective correction
    corrected = cv2.warpPerspective(img, perspective_matrix, (width, height))

    print(f"\n✓ Perspective Correction Applied")
    print(f"  Corrected Image: {width}×{height}")

    # Save corrected image
    corrected_path = os.path.join(output_dir, 'perspective_corrected.jpg')
    cv2.imwrite(corrected_path, corrected)
    print(f"✓ Corrected image saved: {corrected_path}")

    return corrected


def roi_extraction(img, output_dir):
    """
    Extract and manipulate regions of interest (ROI) from an image.

    Demonstrates multiple ROI extraction techniques using direct NumPy array
    slicing (img[y1:y2, x1:x2]) to isolate specific portions of the image.
    Includes extraction of rectangular regions, circular regions (using masks),
    and pixel value manipulation.

    Args:
        img (ndarray): Input image.
        output_dir (str): Directory to save ROI images.

    Returns:
        dict: Dictionary containing extracted ROI images.

    Prints:
        - ROI extraction details for each region
        - Coordinate information
        - Dimension and pixel statistics for each ROI
    """
    print("\n" + "="*60)
    print("5. REGION OF INTEREST (ROI) EXTRACTION")
    print("="*60)

    height, width = img.shape[:2]
    print(f"Full Image Dimensions: {width}×{height}")

    roi_images = {}

    # ROI 1: Top-left rectangular region
    print(f"\n--- ROI 1: Top-left Rectangle ---")
    x1, y1, x2, y2 = 50, 50, 250, 200
    roi_1 = img[y1:y2, x1:x2]  # NumPy array slicing
    print(f"Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Dimensions: {roi_1.shape[1]}×{roi_1.shape[0]}")
    print(f"✓ ROI 1 Extracted")

    roi_path_1 = os.path.join(output_dir, 'roi_1_topleft.jpg')
    cv2.imwrite(roi_path_1, roi_1)
    print(f"  Saved: {roi_path_1}")
    roi_images['roi_1'] = roi_1

    # ROI 2: Center rectangular region
    print(f"\n--- ROI 2: Center Rectangle ---")
    roi_size = 150
    start_x = (width - roi_size) // 2
    start_y = (height - roi_size) // 2
    roi_2 = img[start_y:start_y + roi_size, start_x:start_x + roi_size]
    print(f"Coordinates: ({start_x}, {start_y}) to ({start_x + roi_size}, {start_y + roi_size})")
    print(f"Dimensions: {roi_2.shape[1]}×{roi_2.shape[0]}")
    print(f"✓ ROI 2 Extracted")

    roi_path_2 = os.path.join(output_dir, 'roi_2_center.jpg')
    cv2.imwrite(roi_path_2, roi_2)
    print(f"  Saved: {roi_path_2}")
    roi_images['roi_2'] = roi_2

    # ROI 3: Bottom-right region
    print(f"\n--- ROI 3: Bottom-right Rectangle ---")
    x1, y1, x2, y2 = 350, 250, 600, 400
    roi_3 = img[y1:y2, x1:x2]
    print(f"Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Dimensions: {roi_3.shape[1]}×{roi_3.shape[0]}")
    print(f"✓ ROI 3 Extracted")

    roi_path_3 = os.path.join(output_dir, 'roi_3_bottomright.jpg')
    cv2.imwrite(roi_path_3, roi_3)
    print(f"  Saved: {roi_path_3}")
    roi_images['roi_3'] = roi_3

    # ROI 4: Circular ROI using mask
    print(f"\n--- ROI 4: Circular Region (Using Mask) ---")
    circle_center = (width // 2, height // 2)
    circle_radius = 80
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, circle_center, circle_radius, 255, -1)

    roi_4 = cv2.bitwise_and(img, img, mask=mask)
    print(f"Center: {circle_center}, Radius: {circle_radius}px")
    print(f"✓ Circular ROI Extracted (using mask)")

    roi_path_4 = os.path.join(output_dir, 'roi_4_circular.jpg')
    cv2.imwrite(roi_path_4, roi_4)
    print(f"  Saved: {roi_path_4}")
    roi_images['roi_4'] = roi_4

    # ROI 5: Modify ROI values (highlighting)
    print(f"\n--- ROI 5: Modified ROI (Brightened) ---")
    roi_5 = img.copy()
    x1, y1, x2, y2 = 100, 100, 350, 250
    # Increase brightness in ROI
    roi_5[y1:y2, x1:x2] = cv2.convertScaleAbs(roi_5[y1:y2, x1:x2], alpha=1.5, beta=20)
    print(f"Modified Region: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Modification: Brightness increased (alpha=1.5, beta=20)")
    print(f"✓ ROI Modified")

    roi_path_5 = os.path.join(output_dir, 'roi_5_brightened.jpg')
    cv2.imwrite(roi_path_5, roi_5)
    print(f"  Saved: {roi_path_5}")
    roi_images['roi_5'] = roi_5

    return roi_images


def display_comparison(original, results, titles, wait_key=True):
    """
    Display original and processed images side by side.

    Helper function to visualize transformation results by showing
    original and processed images in separate windows.

    Args:
        original (ndarray): Original input image.
        results (list): List of processed images.
        titles (list): Window titles for each image.
        wait_key (bool): Whether to wait for key press.
    """
    cv2.imshow('Original', original)
    for result, title in zip(results, titles):
        if result is not None:
            cv2.imshow(title, result)

    if wait_key:
        print("  (Press any key to continue...)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """
    Execute all OpenCV Blueprint & Roadmap exercises in sequence.

    Orchestrates the complete workflow:
    1. Creates or loads a sample image
    2. Generates thumbnails with aspect ratio preservation
    3. Resizes images to passport photo specifications
    4. Applies image rotation using transformation matrices
    5. Performs perspective correction on skewed images
    6. Extracts and manipulates regions of interest

    Prints:
        - Progress headers for each exercise section
        - Detailed transformation parameters and matrices
        - Output directory paths
        - Overall completion confirmation
    """
    print("\n" + "="*60)
    print("OPENCV BLUEPRINT & ROADMAP - HANDS-ON EXERCISES")
    print("="*60)

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'sample_image.jpg')
    output_dir = os.path.join(current_dir, 'output_transformations')

    # Create or load sample image
    print("\nLoading sample image...")
    img = create_sample_image(image_path)
    print(f"✓ Image loaded: {image_path}")

    # 1. Thumbnail Generation
    _ = thumbnail_generation(img, output_dir)

    # 2. Passport Photo Resizing
    _ = passport_photo_resizing(img, output_dir)

    # 3. Image Rotation
    _ = image_rotation(img, output_dir)

    # 4. Perspective Correction
    _ = perspective_correction(img, output_dir)

    # 5. ROI Extraction
    _ = roi_extraction(img, output_dir)

    print("\n" + "="*60)
    print("✓ ALL EXERCISES COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput images saved to: {output_dir}")


if __name__ == "__main__":
    main()
