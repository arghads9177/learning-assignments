"""
OpenCV Histogram Visualization - Hands-on Exercises

This script demonstrates comprehensive histogram visualization techniques:
- Grayscale histogram calculation and plotting
- RGB/BGR histogram calculation for individual channels
- Color-coded histogram visualization
- Cumulative distribution function (CDF) analysis
- Color balance analysis and statistics
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def create_sample_image(image_path):
    """
    Create a sample image with various brightness levels and color balances.

    Generates an image with color gradients, colored rectangles, and varying
    brightness to demonstrate histogram patterns.

    Args:
        image_path (str): Path where the sample image will be saved.

    Returns:
        ndarray: The created sample image.

    Prints:
        - Confirmation of image creation
        - Path to saved image
    """
    if os.path.exists(image_path):
        return cv2.imread(image_path)

    # Create base image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 128

    # Add colored regions with gradients
    # Red region (left)
    for x in range(0, 200):
        intensity = int(100 + 155 * (x / 200))
        img[0:400, x, 2] = intensity  # Red channel
        img[0:400, x, 0] = 50          # Blue channel (low)
        img[0:400, x, 1] = 50          # Green channel (low)

    # Green region (middle)
    for x in range(200, 400):
        intensity = int(100 + 155 * ((x - 200) / 200))
        img[0:400, x, 1] = intensity  # Green channel
        img[0:400, x, 0] = 50          # Blue channel (low)
        img[0:400, x, 2] = 50          # Red channel (low)

    # Blue region (right)
    for x in range(400, 600):
        intensity = int(100 + 155 * ((x - 400) / 200))
        img[0:400, x, 0] = intensity  # Blue channel
        img[0:400, x, 1] = 50          # Green channel (low)
        img[0:400, x, 2] = 50          # Red channel (low)

    # Add white rectangle
    cv2.rectangle(img, (50, 50), (150, 100), (255, 255, 255), -1)

    # Add black rectangle
    cv2.rectangle(img, (450, 50), (550, 100), (0, 0, 0), -1)

    # Add gray rectangle
    cv2.rectangle(img, (250, 250), (350, 350), (128, 128, 128), -1)

    os.makedirs(os.path.dirname(image_path) if os.path.dirname(image_path) else '.', exist_ok=True)
    cv2.imwrite(image_path, img)
    print(f"✓ Created sample image: {image_path}")

    return img


def load_image(image_path):
    """
    Load an image for histogram analysis.

    Creates a sample image if the file doesn't exist. Loads the image
    in both BGR (color) and grayscale formats.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (img_bgr, img_gray) where:
            - img_bgr (ndarray): Image in BGR color format
            - img_gray (ndarray): Grayscale version of the image

    Prints:
        - Image dimensions
        - Confirmation of loading
    """
    print("\n" + "="*70)
    print("LOADING IMAGE FOR HISTOGRAM ANALYSIS")
    print("="*70)

    create_sample_image(image_path)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from: {image_path}")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    print(f"✓ Image loaded successfully")
    print(f"  BGR Shape: {img_bgr.shape}")
    print(f"  Grayscale Shape: {img_gray.shape}")

    return img_bgr, img_gray


def calculate_grayscale_histogram(img_gray, bins=256):
    """
    Calculate histogram for a grayscale image.

    Uses cv2.calcHist to compute the distribution of pixel intensities
    in the grayscale image. Computes histogram over 256 bins (0-255 range).

    Args:
        img_gray (ndarray): Grayscale image (single-channel).
        bins (int): Number of histogram bins (default: 256).

    Returns:
        ndarray: Histogram array of shape (bins, 1).

    Prints:
        - Confirmation of histogram calculation
        - Histogram statistics (mean, std deviation, min, max)

    Formula:
        histogram[i] = count of pixels with intensity value i
    """
    print("\n" + "="*70)
    print("GRAYSCALE HISTOGRAM CALCULATION")
    print("="*70)

    # Calculate histogram
    hist = cv2.calcHist([img_gray], [0], None, [bins], [0, 256])

    print(f"✓ Histogram calculated successfully")
    print(f"  Bins: {bins}")
    print(f"  Histogram shape: {hist.shape}")

    # Calculate statistics
    hist_flat = hist.flatten()
    print(f"\nHistogram Statistics:")
    print(f"  Total pixels: {hist_flat.sum():.0f}")
    print(f"  Mean count: {hist_flat.mean():.2f}")
    print(f"  Std deviation: {hist_flat.std():.2f}")
    print(f"  Min count: {hist_flat.min():.0f}")
    print(f"  Max count: {hist_flat.max():.0f}")

    # Calculate percentiles
    cumsum = np.cumsum(hist_flat)
    total = cumsum[-1]
    dark_pixels = np.sum(img_gray < 85)
    mid_pixels = np.sum((img_gray >= 85) & (img_gray < 170))
    bright_pixels = np.sum(img_gray >= 170)

    print(f"\nPixel Distribution:")
    print(f"  Dark pixels (0-84): {dark_pixels} ({dark_pixels/total*100:.1f}%)")
    print(f"  Mid pixels (85-169): {mid_pixels} ({mid_pixels/total*100:.1f}%)")
    print(f"  Bright pixels (170-255): {bright_pixels} ({bright_pixels/total*100:.1f}%)")

    return hist


def calculate_color_histogram(img_bgr, bins=256):
    """
    Calculate histograms for each BGR color channel.

    Separates BGR image into individual color channels and calculates
    the histogram for each channel independently. This preserves spatial
    color information unlike a global color histogram.

    Args:
        img_bgr (ndarray): Image in BGR color format.
        bins (int): Number of histogram bins per channel (default: 256).

    Returns:
        dict: Dictionary with keys 'blue', 'green', 'red' containing histograms.

    Prints:
        - Confirmation of color histogram calculation
        - Statistics for each channel
        - Channel dominance analysis
    """
    print("\n" + "="*70)
    print("RGB/BGR HISTOGRAM CALCULATION")
    print("="*70)

    # Channel names and indices (BGR order in OpenCV)
    channels = {
        'blue': (0, 'Blue (B)'),
        'green': (1, 'Green (G)'),
        'red': (2, 'Red (R)')
    }

    histograms = {}
    channel_stats = {}

    for key, (idx, name) in channels.items():
        # Extract channel
        channel = img_bgr[:, :, idx]

        # Calculate histogram
        hist = cv2.calcHist([img_bgr], [idx], None, [bins], [0, 256])
        histograms[key] = hist

        # Calculate statistics
        hist_flat = hist.flatten()
        mean_intensity = np.mean(channel)
        std_intensity = np.std(channel)

        channel_stats[key] = {
            'mean': mean_intensity,
            'std': std_intensity,
            'min': channel.min(),
            'max': channel.max()
        }

        print(f"\n{name}:")
        print(f"  Mean intensity: {mean_intensity:.2f}")
        print(f"  Std deviation: {std_intensity:.2f}")
        print(f"  Min: {channel.min()}, Max: {channel.max()}")
        print(f"  Histogram count - Min: {hist_flat.min():.0f}, Max: {hist_flat.max():.0f}")

    # Analyze color balance
    print(f"\n" + "-"*70)
    print("COLOR BALANCE ANALYSIS:")
    print("-"*70)

    blue_mean = channel_stats['blue']['mean']
    green_mean = channel_stats['green']['mean']
    red_mean = channel_stats['red']['mean']

    total_mean = (blue_mean + green_mean + red_mean) / 3

    print(f"\nChannel Means:")
    print(f"  Blue:  {blue_mean:>6.2f}")
    print(f"  Green: {green_mean:>6.2f}")
    print(f"  Red:   {red_mean:>6.2f}")
    print(f"  Overall average: {total_mean:.2f}")

    # Identify dominant color
    means = {'Blue': blue_mean, 'Green': green_mean, 'Red': red_mean}
    dominant = max(means.items(), key=lambda x: x[1])[0]
    print(f"\nDominant color channel: {dominant}")

    # Calculate deviation from neutral
    deviations = {
        'Blue': abs(blue_mean - total_mean),
        'Green': abs(green_mean - total_mean),
        'Red': abs(red_mean - total_mean)
    }

    print(f"\nDeviation from neutral gray ({total_mean:.2f}):")
    for color, dev in deviations.items():
        print(f"  {color}: {dev:.2f}")

    return histograms, channel_stats


def plot_grayscale_histogram(img_gray, hist, output_dir):
    """
    Plot grayscale histogram with visualization.

    Creates a figure with the grayscale image, histogram plot, and CDF curve.
    Saves the visualization to a file.

    Args:
        img_gray (ndarray): Original grayscale image.
        hist (ndarray): Histogram array from cv2.calcHist.
        output_dir (str): Directory to save the plot.

    Prints:
        - Confirmation of plot creation and file save
    """
    print("\n" + "="*70)
    print("PLOTTING GRAYSCALE HISTOGRAM")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Display image
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Grayscale Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot histogram
    hist_flat = hist.flatten()
    axes[1].plot(hist_flat, color='black', linewidth=2, label='Histogram')
    axes[1].fill_between(range(len(hist_flat)), hist_flat, alpha=0.3, color='gray')
    axes[1].set_xlabel('Pixel Intensity (0-255)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Grayscale Histogram', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Set x-axis to show intensity range
    axes[1].set_xlim(0, 256)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'grayscale_histogram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Grayscale histogram plot saved: {output_path}")

    plt.close()


def plot_color_histogram(img_bgr, histograms, output_dir):
    """
    Plot color histograms for all BGR channels side-by-side.

    Creates a figure showing the original image, individual channel histograms
    with color-coded curves (blue, green, red), and a combined histogram plot.

    Args:
        img_bgr (ndarray): Original BGR image.
        histograms (dict): Dictionary with 'blue', 'green', 'red' histograms.
        output_dir (str): Directory to save the plot.

    Prints:
        - Confirmation of plot creation and file save
    """
    print("\n" + "="*70)
    print("PLOTTING COLOR HISTOGRAMS")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Display original image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original BGR Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Plot individual channel histograms
    hist_blue = histograms['blue'].flatten()
    hist_green = histograms['green'].flatten()
    hist_red = histograms['red'].flatten()

    # Blue channel histogram
    axes[0, 1].plot(hist_blue, color='blue', linewidth=2, label='Blue')
    axes[0, 1].fill_between(range(len(hist_blue)), hist_blue, alpha=0.3, color='blue')
    axes[0, 1].set_title('Blue Channel Histogram', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Intensity (0-255)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 256)

    # Green channel histogram
    axes[1, 0].plot(hist_green, color='green', linewidth=2, label='Green')
    axes[1, 0].fill_between(range(len(hist_green)), hist_green, alpha=0.3, color='green')
    axes[1, 0].set_title('Green Channel Histogram', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Intensity (0-255)', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 256)

    # Red channel histogram
    axes[1, 1].plot(hist_red, color='red', linewidth=2, label='Red')
    axes[1, 1].fill_between(range(len(hist_red)), hist_red, alpha=0.3, color='red')
    axes[1, 1].set_title('Red Channel Histogram', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Intensity (0-255)', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 256)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'color_histograms_separate.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Color histograms (separate) saved: {output_path}")

    plt.close()


def plot_combined_histogram(histograms, output_dir):
    """
    Plot all color channel histograms on a single graph for comparison.

    Creates a single plot with all three BGR channel histograms overlaid,
    using color-coded lines (blue, green, red) to indicate each channel.

    Args:
        histograms (dict): Dictionary with 'blue', 'green', 'red' histograms.
        output_dir (str): Directory to save the plot.

    Prints:
        - Confirmation of plot creation and file save
    """
    print("\nCreating combined histogram plot...")

    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all histograms
    hist_blue = histograms['blue'].flatten()
    hist_green = histograms['green'].flatten()
    hist_red = histograms['red'].flatten()

    ax.plot(hist_blue, color='blue', linewidth=2.5, label='Blue Channel', alpha=0.8)
    ax.plot(hist_green, color='green', linewidth=2.5, label='Green Channel', alpha=0.8)
    ax.plot(hist_red, color='red', linewidth=2.5, label='Red Channel', alpha=0.8)

    ax.set_xlabel('Pixel Intensity (0-255)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Combined BGR Channel Histograms', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 256)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'color_histograms_combined.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Combined color histogram saved: {output_path}")

    plt.close()


def calculate_cumulative_distribution(hist):
    """
    Calculate cumulative distribution function (CDF) from histogram.

    CDF shows the cumulative count of pixels up to each intensity level.
    Useful for understanding overall brightness distribution.

    Args:
        hist (ndarray): Histogram array from cv2.calcHist.

    Returns:
        ndarray: Normalized CDF values (0 to 1).

    Formula:
        CDF[i] = sum(histogram[0:i]) / total_pixels
    """
    hist_flat = hist.flatten()
    cdf = np.cumsum(hist_flat)
    cdf_normalized = cdf / cdf[-1]  # Normalize to 0-1 range

    return cdf_normalized


def plot_cdf(img_gray, output_dir):
    """
    Plot cumulative distribution function for grayscale image.

    Creates a plot showing both the histogram and its CDF curve.

    Args:
        img_gray (ndarray): Grayscale image.
        output_dir (str): Directory to save the plot.

    Prints:
        - Confirmation of plot creation and file save
    """
    print("\n" + "="*70)
    print("PLOTTING CUMULATIVE DISTRIBUTION FUNCTION (CDF)")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Calculate histogram and CDF
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    cdf = calculate_cumulative_distribution(hist)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot histogram on first axis
    ax1.bar(range(256), hist.flatten(), color='gray', alpha=0.6, label='Histogram')
    ax1.set_xlabel('Pixel Intensity (0-255)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency (Histogram)', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(0, 256)

    # Plot CDF on second axis
    ax2 = ax1.twinx()
    ax2.plot(range(256), cdf, color='red', linewidth=3, label='CDF', marker='o', markersize=1)
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.05)

    # Title and legend
    fig.suptitle('Histogram and Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax2.legend(loc='upper right', fontsize=11)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'cdf_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ CDF plot saved: {output_path}")

    plt.close()


def create_visualization_summary(img_bgr, img_gray, histograms, output_dir):
    """
    Create a comprehensive visualization summary.

    Generates a document summarizing histogram analysis with key insights
    and visualizations.

    Args:
        img_bgr (ndarray): BGR image.
        img_gray (ndarray): Grayscale image.
        histograms (dict): Color histograms.
        output_dir (str): Directory to save files.

    Prints:
        - Summary information
    """
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    gray_mean = img_gray.mean()
    gray_std = img_gray.std()

    print(f"\nGrayscale Image Statistics:")
    print(f"  Mean brightness: {gray_mean:.2f}")
    print(f"  Std deviation: {gray_std:.2f}")
    print(f"  Min: {img_gray.min()}, Max: {img_gray.max()}")

    # Create summary document
    summary_path = os.path.join(output_dir, 'histogram_analysis_summary.txt')

    summary_content = f"""
{'='*70}
HISTOGRAM VISUALIZATION ANALYSIS SUMMARY
{'='*70}

IMAGE PROPERTIES:
  BGR Shape: {img_bgr.shape}
  Grayscale Shape: {img_gray.shape}

GRAYSCALE STATISTICS:
  Mean brightness: {gray_mean:.2f}
  Std deviation: {gray_std:.2f}
  Min intensity: {img_gray.min()}
  Max intensity: {img_gray.max()}

COLOR CHANNEL STATISTICS:
  Blue channel:
    Mean: {img_bgr[:,:,0].mean():.2f}
    Std: {img_bgr[:,:,0].std():.2f}

  Green channel:
    Mean: {img_bgr[:,:,1].mean():.2f}
    Std: {img_bgr[:,:,1].std():.2f}

  Red channel:
    Mean: {img_bgr[:,:,2].mean():.2f}
    Std: {img_bgr[:,:,2].std():.2f}

KEY OBSERVATIONS:
  - Histogram shows the distribution of pixel intensities
  - Individual channel histograms reveal color balance
  - CDF indicates cumulative brightness distribution
  - Peaks in histogram represent dominant intensity levels
  - Spread indicates image contrast (std deviation)

VISUALIZATION FILES CREATED:
  1. grayscale_histogram.png - Grayscale histogram with frequency distribution
  2. color_histograms_separate.png - Individual BGR channel histograms
  3. color_histograms_combined.png - All channels overlaid for comparison
  4. cdf_plot.png - Cumulative distribution function visualization

{'='*70}
"""

    with open(summary_path, 'w') as f:
        f.write(summary_content)

    print(f"✓ Summary created: {summary_path}")


def main():
    """
    Execute all histogram visualization exercises.

    Orchestrates the complete workflow:
    1. Load image (create sample if needed)
    2. Calculate grayscale histogram
    3. Calculate color channel histograms
    4. Generate all visualizations
    5. Create comprehensive summary
    6. Display results

    Prints:
        - Progress headers for each section
        - Detailed statistics and analysis
        - File paths of saved visualizations
    """
    print("\n" + "="*70)
    print("OPENCV HISTOGRAM VISUALIZATION - HANDS-ON EXERCISES")
    print("="*70)

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'sample_image.jpg')
    output_dir = os.path.join(current_dir, 'output_visualizations')

    # 1. Load image
    img_bgr, img_gray = load_image(image_path)

    # 2. Calculate grayscale histogram
    hist_gray = calculate_grayscale_histogram(img_gray)

    # 3. Calculate color histograms
    histograms, channel_stats = calculate_color_histogram(img_bgr)

    # 4. Generate visualizations
    plot_grayscale_histogram(img_gray, hist_gray, output_dir)
    plot_color_histogram(img_bgr, histograms, output_dir)
    plot_combined_histogram(histograms, output_dir)
    plot_cdf(img_gray, output_dir)

    # 5. Create summary
    create_visualization_summary(img_bgr, img_gray, histograms, output_dir)

    print("\n" + "="*70)
    print("✓ ALL VISUALIZATION EXERCISES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated visualizations:")
    print("  1. grayscale_histogram.png")
    print("  2. color_histograms_separate.png")
    print("  3. color_histograms_combined.png")
    print("  4. cdf_plot.png")
    print("  5. histogram_analysis_summary.txt")


if __name__ == "__main__":
    main()
