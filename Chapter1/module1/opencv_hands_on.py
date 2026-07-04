"""
OpenCV Hands-on Exercises Module

This script demonstrates fundamental OpenCV operations including:
- Loading and displaying images
- Saving images with different formats and qualities
- Reading image metadata
- Splitting and merging color channels
- Converting between different color spaces
"""

import cv2
import numpy as np
import os
from datetime import datetime


def load_images(image_path):
    """
    Load images from file in multiple formats.

    Demonstrates loading images in BGR (default OpenCV format), grayscale,
    and with alpha channel. If the image doesn't exist, creates a sample
    image with colored rectangles for demonstration purposes.

    Args:
        image_path (str): Path to the image file to load.

    Returns:
        tuple: (img_bgr, img_gray) where:
            - img_bgr (ndarray): Image in BGR color format (default OpenCV)
            - img_gray (ndarray): Grayscale version of the image

    Prints:
        - Image loading confirmation and shape information
        - Data types and dimensions for each loaded format
    """
    print("\n" + "="*50)
    print("1. LOADING IMAGES")
    print("="*50)

    if not os.path.exists(image_path):
        print(f"Creating a sample image for demonstration...")
        # Create a sample image if file doesn't exist
        sample_img = np.zeros((300, 300, 3), dtype=np.uint8)
        sample_img[50:150, 50:150] = [0, 255, 0]  # Green rectangle
        sample_img[150:250, 150:250] = [255, 0, 0]  # Blue rectangle
        sample_img[50:150, 150:250] = [0, 0, 255]  # Red rectangle
        cv2.imwrite(image_path, sample_img)

    # Load image in BGR format (default OpenCV format)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from: {image_path}")
    print(f"✓ Image loaded successfully from: {image_path}")
    print(f"  Data type: {type(img_bgr)}, Shape: {img_bgr.shape}")

    # Load image in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Failed to load grayscale image from: {image_path}")
    print(f"✓ Grayscale image loaded, Shape: {img_gray.shape}")

    # Load image with alpha channel
    img_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_alpha is None:
        raise ValueError(f"Failed to load image with alpha channel from: {image_path}")
    print(f"✓ Image with alpha channel loaded, Shape: {img_alpha.shape}")

    return img_bgr, img_gray


def display_images(img_bgr, img_gray):
    """
    Display images in separate OpenCV windows.

    Opens display windows for both color and grayscale images. The function
    blocks execution until the user presses a key, then closes all windows.
    Note: BGR images displayed using cv2.imshow() will show colors in BGR order.

    Args:
        img_bgr (ndarray): Image in BGR color format.
        img_gray (ndarray): Grayscale image.

    Prints:
        - Confirmation messages for displayed images
        - Instructions for continuing (press any key)
    """
    print("\n" + "="*50)
    print("2. DISPLAYING IMAGES")
    print("="*50)

    # Display BGR image
    cv2.imshow('Original Image (BGR)', img_bgr)
    print("✓ Displayed original image (BGR format)")

    # Display grayscale image
    cv2.imshow('Grayscale Image', img_gray)
    print("✓ Displayed grayscale image")

    print("  (Press any key to continue...)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✓ Windows closed")


def save_images(img_bgr, img_gray, output_dir):
    """
    Save images to disk in multiple formats and quality settings.

    Saves the color and grayscale images in both JPEG and PNG formats.
    Demonstrates saving with different quality levels (JPEG quality: 0-100,
    where 95 is high quality). Creates the output directory if it doesn't exist.

    Args:
        img_bgr (ndarray): Image in BGR color format.
        img_gray (ndarray): Grayscale image.
        output_dir (str): Directory path where images will be saved.

    Prints:
        - File paths of saved images
        - Confirmation messages for each save operation
    """
    print("\n" + "="*50)
    print("3. SAVING IMAGES")
    print("="*50)

    os.makedirs(output_dir, exist_ok=True)

    # Save original image
    original_path = os.path.join(output_dir, 'original_image.jpg')
    cv2.imwrite(original_path, img_bgr)
    print(f"✓ Original image saved: {original_path}")

    # Save grayscale image
    gray_path = os.path.join(output_dir, 'grayscale_image.jpg')
    cv2.imwrite(gray_path, img_gray)
    print(f"✓ Grayscale image saved: {gray_path}")

    # Save with different quality
    quality_path = os.path.join(output_dir, 'high_quality_image.jpg')
    cv2.imwrite(quality_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✓ High quality image saved: {quality_path}")

    # Save as PNG
    png_path = os.path.join(output_dir, 'image.png')
    cv2.imwrite(png_path, img_bgr)
    print(f"✓ PNG image saved: {png_path}")


def read_metadata(image_path):
    """
    Read and display comprehensive image metadata.

    Extracts and displays file-level metadata (size, timestamps) and
    image-level properties (dimensions, data type, channels, memory usage).
    Provides complete information about the image file and its properties.

    Args:
        image_path (str): Path to the image file.

    Prints:
        - File name, path, size, and modification times (in YYYY-MM-DD HH:MM:SS format)
        - Image dimensions (height, width, channels)
        - Data type and total pixel count
        - Memory consumption
    """
    print("\n" + "="*50)
    print("4. READING METADATA")
    print("="*50)

    # Get file information
    file_size = os.path.getsize(image_path)
    file_stat = os.stat(image_path)

    # Convert timestamps to readable datetime format
    creation_time = datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
    modification_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    print(f"File Name: {os.path.basename(image_path)}")
    print(f"File Path: {image_path}")
    print(f"File Size: {file_size} bytes ({file_size / 1024:.2f} KB)")
    print(f"Creation Time: {creation_time}")
    print(f"Modification Time: {modification_time}")

    # Load image and get its properties
    img = cv2.imread(image_path)
    if img is not None:
        print(f"\nImage Properties:")
        print(f"  Dimensions (H×W×C): {img.shape[0]}×{img.shape[1]}×{img.shape[2]}")
        print(f"  Data Type: {img.dtype}")
        print(f"  Number of Channels: {img.shape[2] if len(img.shape) > 2 else 1}")
        print(f"  Total Pixels: {img.shape[0] * img.shape[1]}")
        print(f"  Memory Size: {img.nbytes / 1024:.2f} KB")


def print_image_shape(img_bgr, img_gray):
    """
    Print detailed shape and dimension information for images.

    Displays the shape tuple (height, width, channels) for both color and
    grayscale images, along with individual dimension details and pixel
    statistics (min/max values).

    Args:
        img_bgr (ndarray): Image in BGR color format.
        img_gray (ndarray): Grayscale image.

    Prints:
        - Shape tuples for both images
        - Individual dimensions (height, width, channels)
        - Total element count and data type
        - Min/max pixel values for color image
    """
    print("\n" + "="*50)
    print("5. IMAGE SHAPE AND DIMENSIONS")
    print("="*50)

    print(f"Color Image Shape: {img_bgr.shape}")
    print(f"  Height: {img_bgr.shape[0]} pixels")
    print(f"  Width: {img_bgr.shape[1]} pixels")
    print(f"  Channels: {img_bgr.shape[2]}")

    print(f"\nGrayscale Image Shape: {img_gray.shape}")
    print(f"  Height: {img_gray.shape[0]} pixels")
    print(f"  Width: {img_gray.shape[1]} pixels")
    print(f"  Channels: 1 (single channel)")

    print(f"\nColor Image Info:")
    print(f"  Total Elements: {img_bgr.size}")
    print(f"  Data Type: {img_bgr.dtype}")
    print(f"  Min Value: {img_bgr.min()}, Max Value: {img_bgr.max()}")


def split_channels(img_bgr):
    """
    Decompose an image into its individual color channels.

    Splits a BGR image into three separate single-channel arrays (Blue, Green, Red).
    This is useful for analyzing individual color components or applying
    channel-specific operations. Also displays statistics for each channel.

    Args:
        img_bgr (ndarray): Image in BGR color format with shape (H, W, 3).

    Returns:
        tuple: (b, g, r) where each is a single-channel array of shape (H, W).
            - b (ndarray): Blue channel
            - g (ndarray): Green channel
            - r (ndarray): Red channel

    Prints:
        - Confirmation of channel split
        - Shape information for each channel
        - Statistics (min, max, mean) for each channel
    """
    print("\n" + "="*50)
    print("6. SPLITTING CHANNELS")
    print("="*50)

    # OpenCV uses BGR format by default
    b, g, r = cv2.split(img_bgr)

    print(f"Original Image Shape: {img_bgr.shape}")
    print(f"✓ Image split into 3 channels")
    print(f"  Blue Channel Shape: {b.shape}")
    print(f"  Green Channel Shape: {g.shape}")
    print(f"  Red Channel Shape: {r.shape}")

    # Display channel statistics
    print(f"\nChannel Statistics:")
    print(f"  Blue   - Min: {b.min()}, Max: {b.max()}, Mean: {b.mean():.2f}")
    print(f"  Green  - Min: {g.min()}, Max: {g.max()}, Mean: {g.mean():.2f}")
    print(f"  Red    - Min: {r.min()}, Max: {r.max()}, Mean: {r.mean():.2f}")

    return b, g, r


def merge_channels(b, g, r):
    """
    Combine individual color channels back into a multi-channel image.

    Takes three separate single-channel arrays and merges them into a single
    BGR image. Demonstrates channel merging and also creates isolated channel
    views (red-only, green-only, blue-only) by zeroing out other channels.

    Args:
        b (ndarray): Blue channel array of shape (H, W).
        g (ndarray): Green channel array of shape (H, W).
        r (ndarray): Red channel array of shape (H, W).

    Returns:
        tuple: (merged, red_only, green_only, blue_only) where:
            - merged (ndarray): Full BGR image created from channels
            - red_only (ndarray): Image showing only red channel
            - green_only (ndarray): Image showing only green channel
            - blue_only (ndarray): Image showing only blue channel

    Prints:
        - Confirmation of merge operation
        - Shape of merged image
        - Confirmation of each isolated channel creation
    """
    print("\n" + "="*50)
    print("7. MERGING CHANNELS")
    print("="*50)

    # Merge channels back
    merged = cv2.merge([b, g, r])
    print(f"✓ Channels merged successfully")
    print(f"  Merged Image Shape: {merged.shape}")

    # Create channel combinations
    print(f"\nChannel Combinations:")

    # Red channel only
    red_only = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
    print(f"✓ Red channel isolated, Shape: {red_only.shape}")

    # Green channel only
    green_only = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
    print(f"✓ Green channel isolated, Shape: {green_only.shape}")

    # Blue channel only
    blue_only = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])
    print(f"✓ Blue channel isolated, Shape: {blue_only.shape}")

    return merged, red_only, green_only, blue_only


def convert_color_spaces(img_bgr):
    """
    Convert an image between various color spaces.

    Demonstrates conversion from BGR to multiple color spaces including RGB,
    Grayscale, HSV, LAB, YCrCb, XYZ, and LUV. Each color space represents
    color information differently and is useful for different applications:
    - RGB/BGR: Standard color representation
    - Grayscale: Single intensity channel
    - HSV: Hue, Saturation, Value (useful for color detection)
    - LAB: Perceptually uniform color space
    - YCrCb: Used in video compression
    - XYZ: Device-independent color space
    - LUV: Perceptually uniform alternative to LAB

    Args:
        img_bgr (ndarray): Image in BGR color format.

    Returns:
        dict: Dictionary with color space names as keys and converted images as values.
            Example: {'BGR to RGB': rgb_image, 'BGR to HSV': hsv_image, ...}

    Prints:
        - List of all color space conversions performed
        - Shape information for each converted image
        - Displays converted images in separate windows
    """
    print("\n" + "="*50)
    print("8. COLOR SPACE CONVERSION")
    print("="*50)

    conversions = {
        'BGR to RGB': (img_bgr, cv2.COLOR_BGR2RGB),
        'BGR to Grayscale': (img_bgr, cv2.COLOR_BGR2GRAY),
        'BGR to HSV': (img_bgr, cv2.COLOR_BGR2HSV),
        'BGR to LAB': (img_bgr, cv2.COLOR_BGR2LAB),
        'BGR to YCrCb': (img_bgr, cv2.COLOR_BGR2YCrCb),
        'BGR to XYZ': (img_bgr, cv2.COLOR_BGR2XYZ),
        'BGR to LUV': (img_bgr, cv2.COLOR_BGR2LUV),
    }

    results = {}
    for name, (image, code) in conversions.items():
        converted = cv2.cvtColor(image, code)
        results[name] = converted

        shape_info = f"Shape: {converted.shape}" if len(converted.shape) == 3 else f"Shape: {converted.shape} (2D)"
        print(f"✓ {name:<25} - {shape_info}")

    # Display conversions
    print(f"\nDisplaying color space conversions...")
    cv2.imshow('BGR (Original)', img_bgr)
    cv2.imshow('RGB', results['BGR to RGB'])
    cv2.imshow('Grayscale', cv2.cvtColor(results['BGR to Grayscale'], cv2.COLOR_GRAY2BGR))
    cv2.imshow('HSV', cv2.cvtColor(results['BGR to HSV'], cv2.COLOR_HSV2BGR))
    cv2.imshow('LAB', cv2.cvtColor(results['BGR to LAB'], cv2.COLOR_LAB2BGR))

    print("  (Press any key to continue...)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


def main():
    """
    Execute all OpenCV hands-on exercises in sequence.

    Orchestrates the complete workflow:
    1. Creates sample image if needed and loads images in multiple formats
    2. Displays images in windows
    3. Saves images in various formats and quality levels
    4. Reads and displays file metadata
    5. Prints image shape and dimension information
    6. Splits image into color channels and displays statistics
    7. Merges channels back and creates isolated channel views
    8. Converts image to multiple color spaces and displays results

    Prints:
        - Progress headers for each exercise section
        - Output directory path at completion
        - Overall completion confirmation
    """
    print("\n" + "="*50)
    print("OPENCV HANDS-ON EXERCISES")
    print("="*50)

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'sample_image.jpeg')
    output_dir = os.path.join(current_dir, 'output_images')

    # 1. Load images
    img_bgr, img_gray = load_images(image_path)

    # 2. Display images
    display_images(img_bgr, img_gray)

    # 3. Save images
    save_images(img_bgr, img_gray, output_dir)

    # 4. Read metadata
    read_metadata(image_path)

    # 5. Print image shape
    print_image_shape(img_bgr, img_gray)

    # 6. Split channels
    b, g, r = split_channels(img_bgr)

    # 7. Merge channels
    _ = merge_channels(b, g, r)

    # 8. Convert color spaces
    _ = convert_color_spaces(img_bgr)

    print("\n" + "="*50)
    print("✓ ALL EXERCISES COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"\nOutput images saved to: {output_dir}")


if __name__ == "__main__":
    main()
