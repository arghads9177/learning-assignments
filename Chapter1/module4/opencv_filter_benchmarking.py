"""
OpenCV Filter Benchmarking - Hands-on Exercises

This script demonstrates comprehensive benchmarking of image smoothing filters:
- Speed benchmarking using time.perf_counter() over multiple iterations
- Quality assessment using PSNR (Peak Signal-to-Noise Ratio)
- Quality assessment using SSIM (Structural Similarity Index)
- Comparison of 4 smoothing techniques: Box, Gaussian, Bilateral, Median
- Edge preservation vs noise suppression analysis
"""

import cv2
import numpy as np
import os
import time


def create_noisy_image(image_path):
    """
    Create a clean sample image and generate a noisy version for testing.

    Creates a clean image with geometric shapes and adds Gaussian noise
    to simulate real-world noisy images.

    Args:
        image_path (str): Base path for saving images (without extension).

    Returns:
        tuple: (clean_image, noisy_image) where:
            - clean_image (ndarray): Clean reference image
            - noisy_image (ndarray): Noisy version for filtering

    Prints:
        - Confirmation of image creation
        - Image dimensions and noise characteristics
    """
    clean_path = image_path + "_clean.jpg"
    noisy_path = image_path + "_noisy.jpg"

    if os.path.exists(clean_path) and os.path.exists(noisy_path):
        clean = cv2.imread(clean_path)
        noisy = cv2.imread(noisy_path)
        return clean, noisy

    # Create clean image with geometric shapes
    clean = np.ones((400, 600, 3), dtype=np.uint8) * 200  # Light gray background

    # Add colored rectangles
    cv2.rectangle(clean, (50, 50), (200, 150), (0, 255, 0), -1)  # Green
    cv2.rectangle(clean, (250, 100), (450, 250), (255, 0, 0), -1)  # Blue
    cv2.rectangle(clean, (100, 250), (300, 350), (0, 0, 255), -1)  # Red

    # Add circles
    cv2.circle(clean, (150, 200), 40, (0, 165, 255), -1)  # Orange
    cv2.circle(clean, (500, 150), 50, (255, 255, 0), -1)  # Cyan

    # Add gradients for smooth transitions
    for y in range(clean.shape[0]):
        for x in range(clean.shape[1]):
            if x < 300:
                clean[y, x] = (clean[y, x] * (1 - x / 300.0)).astype(np.uint8)

    # Save clean image
    os.makedirs(os.path.dirname(image_path) if os.path.dirname(image_path) else '.', exist_ok=True)
    cv2.imwrite(clean_path, clean)

    # Create noisy image by adding Gaussian noise
    noise = np.random.normal(0, 25, clean.shape).astype(np.float32)
    noisy = cv2.add(clean.astype(np.float32), noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    cv2.imwrite(noisy_path, noisy)
    return clean, noisy


def load_test_images(image_path):
    """
    Load clean and noisy test images for benchmarking.

    Creates test images if they don't exist and loads them.

    Args:
        image_path (str): Base path for image files.

    Returns:
        tuple: (clean_image, noisy_image) where:
            - clean_image (ndarray): Clean reference image (ground truth)
            - noisy_image (ndarray): Noisy image for filtering

    Prints:
        - Image dimensions
        - Noise statistics
    """
    print("\n" + "="*70)
    print("LOADING TEST IMAGES")
    print("="*70)

    clean, noisy = create_noisy_image(image_path)

    print(f"✓ Clean image loaded: Shape {clean.shape}")
    print(f"✓ Noisy image loaded: Shape {noisy.shape}")

    # Calculate noise statistics
    noise_diff = cv2.absdiff(clean.astype(np.float32), noisy.astype(np.float32))
    noise_mean = noise_diff.mean()
    noise_std = noise_diff.std()

    print(f"\nNoise Statistics:")
    print(f"  Mean absolute difference: {noise_mean:.2f}")
    print(f"  Std deviation: {noise_std:.2f}")
    print(f"  Noise estimate (σ): ~25.0")

    return clean, noisy


def psnr(clean, filtered):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX²/MSE) where MAX=255 for uint8 images.
    Higher values indicate better quality (more similar to clean image).

    Args:
        clean (ndarray): Clean reference image (ground truth).
        filtered (ndarray): Filtered image to evaluate.

    Returns:
        float: PSNR value in dB.

    Formula:
        MSE = mean((clean - filtered)²)
        PSNR = 10 * log10(255² / MSE)
    """
    clean = clean.astype(np.float32)
    filtered = filtered.astype(np.float32)

    mse = np.mean((clean - filtered) ** 2)
    if mse == 0:
        return 100.0  # Perfect reconstruction

    max_pixel = 255.0
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr_value


def ssim_score(clean, filtered):
    """
    Calculate Structural Similarity Index (SSIM).

    SSIM measures perceived quality considering luminance, contrast, and structure.
    Values range from -1 to 1, where 1 is perfect similarity.

    Args:
        clean (ndarray): Clean reference image (ground truth).
        filtered (ndarray): Filtered image to evaluate.

    Returns:
        float: SSIM value (range: -1 to 1).

    Formula:
        SSIM = (2μx*μy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
    """
    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) if len(clean.shape) == 3 else clean
    filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY) if len(filtered.shape) == 3 else filtered

    clean = clean.astype(np.float32)
    filtered = filtered.astype(np.float32)

    # Constants for SSIM
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Calculate local means
    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel = kernel @ kernel.T

    mu1 = cv2.filter2D(clean, -1, kernel)
    mu2 = cv2.filter2D(filtered, -1, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(clean ** 2, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(filtered ** 2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(clean * filtered, -1, kernel) - mu1_mu2

    # Calculate SSIM map
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return np.mean(ssim_map)


def benchmark_speed(noisy, kernel_size=5, iterations=100):
    """
    Benchmark execution time of all smoothing filters.

    Measures wall-clock time using time.perf_counter() for accurate benchmarking.
    Each filter is applied multiple iterations to reduce timing noise.

    Args:
        noisy (ndarray): Input noisy image.
        kernel_size (int): Size of convolution kernel (default: 5).
        iterations (int): Number of iterations per filter (default: 100).

    Returns:
        dict: Dictionary with filter names as keys and execution times (ms) as values.

    Prints:
        - Execution time for each filter
        - Time per iteration
        - Speed ranking (fastest to slowest)
    """
    print("\n" + "="*70)
    print("SPEED BENCHMARKING (100 iterations)")
    print("="*70)

    results = {}
    times = {}

    # Box Filter (cv2.blur)
    print(f"\n1. Box Filter (cv2.blur):")
    start = time.perf_counter()
    for _ in range(iterations):
        cv2.blur(noisy, (kernel_size, kernel_size))
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    times['Box Filter'] = elapsed
    results['box'] = cv2.blur(noisy, (kernel_size, kernel_size))
    print(f"   Total time: {elapsed:.2f} ms")
    print(f"   Per iteration: {elapsed/iterations:.4f} ms")

    # Gaussian Filter (cv2.GaussianBlur)
    print(f"\n2. Gaussian Filter (cv2.GaussianBlur):")
    start = time.perf_counter()
    for _ in range(iterations):
        cv2.GaussianBlur(noisy, (kernel_size, kernel_size), 1.0)
    elapsed = (time.perf_counter() - start) * 1000
    times['Gaussian Filter'] = elapsed
    results['gaussian'] = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), 1.0)
    print(f"   Total time: {elapsed:.2f} ms")
    print(f"   Per iteration: {elapsed/iterations:.4f} ms")

    # Median Filter (cv2.medianBlur)
    print(f"\n3. Median Filter (cv2.medianBlur):")
    start = time.perf_counter()
    for _ in range(iterations):
        cv2.medianBlur(noisy, kernel_size)
    elapsed = (time.perf_counter() - start) * 1000
    times['Median Filter'] = elapsed
    results['median'] = cv2.medianBlur(noisy, kernel_size)
    print(f"   Total time: {elapsed:.2f} ms")
    print(f"   Per iteration: {elapsed/iterations:.4f} ms")

    # Bilateral Filter (cv2.bilateralFilter)
    print(f"\n4. Bilateral Filter (cv2.bilateralFilter):")
    start = time.perf_counter()
    for _ in range(iterations):
        cv2.bilateralFilter(noisy, kernel_size, 75, 75)
    elapsed = (time.perf_counter() - start) * 1000
    times['Bilateral Filter'] = elapsed
    results['bilateral'] = cv2.bilateralFilter(noisy, kernel_size, 75, 75)
    print(f"   Total time: {elapsed:.2f} ms")
    print(f"   Per iteration: {elapsed/iterations:.4f} ms")

    # Ranking
    print(f"\n" + "-"*70)
    print("SPEED RANKING (Fastest to Slowest):")
    for rank, (name, time_ms) in enumerate(sorted(times.items(), key=lambda x: x[1]), 1):
        print(f"  {rank}. {name:<20} - {time_ms:>8.2f} ms")

    return results, times


def benchmark_quality(clean, filtered_results, output_dir):
    """
    Benchmark quality of filtered images using PSNR and SSIM metrics.

    Compares each filtered result against the clean reference image
    using objective quality metrics.

    Args:
        clean (ndarray): Clean reference image (ground truth).
        filtered_results (dict): Dictionary of filtered images.
        output_dir (str): Directory to save results.

    Returns:
        dict: Dictionary with quality metrics for each filter.

    Prints:
        - PSNR and SSIM for each filter
        - Quality ranking
        - Interpretation of results
    """
    print("\n" + "="*70)
    print("QUALITY BENCHMARKING (Against Clean Reference)")
    print("="*70)

    metrics = {}

    for name, filtered in filtered_results.items():
        psnr_val = psnr(clean, filtered)
        ssim_val = ssim_score(clean, filtered)
        metrics[name] = {'psnr': psnr_val, 'ssim': ssim_val}

        print(f"\n{name.upper()}:")
        print(f"  PSNR: {psnr_val:.2f} dB")
        print(f"  SSIM: {ssim_val:.4f}")

    # Ranking by PSNR
    print(f"\n" + "-"*70)
    print("QUALITY RANKING (by PSNR - Higher is Better):")
    sorted_psnr = sorted(metrics.items(), key=lambda x: x[1]['psnr'], reverse=True)
    for rank, (name, values) in enumerate(sorted_psnr, 1):
        print(f"  {rank}. {name.upper():<20} - PSNR: {values['psnr']:>6.2f} dB, SSIM: {values['ssim']:>6.4f}")

    return metrics


def analyze_edge_preservation(clean, noisy, filtered_results, output_dir):
    """
    Analyze edge preservation capability of each filter.

    Uses Sobel edge detection to measure edge sharpness before and after filtering.
    Quantifies how well each filter preserves important image features while
    suppressing noise. Saves edge maps to output directory.

    Args:
        clean (ndarray): Clean reference image.
        noisy (ndarray): Noisy image.
        filtered_results (dict): Dictionary of filtered images.
        output_dir (str): Directory to save edge maps.

    Prints:
        - Edge energy for each filter
        - Edge preservation analysis
        - Recommendations for different use cases
    """
    print("\n" + "="*70)
    print("EDGE PRESERVATION ANALYSIS")
    print("="*70)

    # Convert to grayscale for edge detection
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)

    # Compute edge maps using Sobel
    sobelx_clean = cv2.Sobel(clean_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely_clean = cv2.Sobel(clean_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_clean = np.sqrt(sobelx_clean**2 + sobely_clean**2)

    sobelx_noisy = cv2.Sobel(noisy_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely_noisy = cv2.Sobel(noisy_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_noisy = np.sqrt(sobelx_noisy**2 + sobely_noisy**2)

    edge_energy_clean = np.sum(edge_clean)
    edge_energy_noisy = np.sum(edge_noisy)

    print(f"\nReference Edge Energy:")
    print(f"  Clean image: {edge_energy_clean:.0f}")
    print(f"  Noisy image: {edge_energy_noisy:.0f}")

    # Analyze filtered images
    edge_analysis = {}
    for name, filtered in filtered_results.items():
        filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(filtered_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_filtered = np.sqrt(sobelx**2 + sobely**2)

        edge_energy = np.sum(edge_filtered)
        preservation_ratio = edge_energy / edge_energy_clean

        edge_analysis[name] = {
            'edge_energy': edge_energy,
            'preservation_ratio': preservation_ratio
        }

        print(f"\n{name.upper()}:")
        print(f"  Edge energy: {edge_energy:.0f}")
        print(f"  Preservation ratio: {preservation_ratio:.4f} (1.0 = perfect preservation)")

    # Save edge maps
    os.makedirs(output_dir, exist_ok=True)
    for name, filtered in filtered_results.items():
        filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(filtered_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = cv2.magnitude(sobelx, sobely)
        edge_map_normalized = np.zeros_like(edge_map)
        cv2.normalize(edge_map, edge_map_normalized, 0, 255, cv2.NORM_MINMAX)
        edge_map_uint8 = edge_map_normalized.astype(np.uint8)

        edge_path = os.path.join(output_dir, f'edge_map_{name}.jpg')
        cv2.imwrite(edge_path, edge_map_uint8)

    return edge_analysis


def display_comparison(clean, noisy, filtered_results, wait_key=True):
    """
    Display original, noisy, and filtered images for visual comparison.

    Args:
        clean (ndarray): Clean reference image.
        noisy (ndarray): Noisy image.
        filtered_results (dict): Dictionary of filtered images.
        wait_key (bool): Whether to wait for key press.
    """
    cv2.imshow("Original (Clean)", clean)
    cv2.imshow("Noisy Image", noisy)

    for name, filtered in filtered_results.items():
        cv2.imshow(f"Filtered ({name.upper()})", filtered)

    if wait_key:
        print("  (Press any key to continue...)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_comparison_report(times, metrics, edge_analysis, output_dir):
    """
    Create a comprehensive comparison report and save to file.

    Args:
        times (dict): Speed benchmark results.
        metrics (dict): Quality metrics results.
        edge_analysis (dict): Edge preservation analysis results.
        output_dir (str): Directory to save report.

    Prints:
        - Comprehensive comparison table
        - Recommendations for each filter
    """
    filter_names = {
        'box': 'Box Filter',
        'gaussian': 'Gaussian Filter',
        'median': 'Median Filter',
        'bilateral': 'Bilateral Filter'
    }

    print("\n" + "="*70)
    print("COMPREHENSIVE BENCHMARKING REPORT")
    print("="*70)

    print(f"\n{'FILTER':<20} {'TIME (ms)':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'EDGE PRES.':<12}")
    print("-"*70)

    for name in ['box', 'gaussian', 'median', 'bilateral']:
        time_ms = times.get(filter_names[name], 0)

        if name in metrics:
            psnr_val = metrics[name]['psnr']
            ssim_val = metrics[name]['ssim']
            edge_pres = edge_analysis[name]['preservation_ratio']

            print(f"{name.upper():<20} {time_ms:<12.2f} {psnr_val:<12.2f} {ssim_val:<10.4f} {edge_pres:<12.4f}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    print(f"\n1. BOX FILTER (FASTEST)")
    print(f"   - Fastest filter, suitable for real-time applications")
    print(f"   - Good noise suppression with minimal edge blurring")
    print(f"   - Use when: Speed is critical, slight blur acceptable")

    print(f"\n2. GAUSSIAN FILTER (BALANCED)")
    print(f"   - Good balance between speed and quality")
    print(f"   - Smooth, natural-looking results")
    print(f"   - Use when: General-purpose smoothing needed")

    print(f"\n3. MEDIAN FILTER (EDGE PRESERVATION)")
    print(f"   - Excellent for salt-and-pepper noise")
    print(f"   - Very good edge preservation")
    print(f"   - Use when: Preserving sharp edges is important")

    print(f"\n4. BILATERAL FILTER (SLOWEST, BEST QUALITY)")
    print(f"   - Slowest but best overall quality")
    print(f"   - Excellent edge preservation + noise suppression")
    print(f"   - Use when: Quality is more important than speed")

    # Save report to file
    report_path = os.path.join(output_dir, 'benchmarking_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("OPENCV FILTER BENCHMARKING REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"{'FILTER':<20} {'TIME (ms)':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'EDGE PRES.':<12}\n")
        f.write("-"*70 + "\n")

        for name in ['box', 'gaussian', 'median', 'bilateral']:
            time_ms = times.get(filter_names[name], 0)

            if name in metrics:
                psnr_val = metrics[name]['psnr']
                ssim_val = metrics[name]['ssim']
                edge_pres = edge_analysis[name]['preservation_ratio']

                f.write(f"{name.upper():<20} {time_ms:<12.2f} {psnr_val:<12.2f} {ssim_val:<10.4f} {edge_pres:<12.4f}\n")

    print(f"\n✓ Report saved: {report_path}")


def main():
    """
    Execute comprehensive filter benchmarking exercises.

    Orchestrates the complete workflow:
    1. Loads or creates clean and noisy test images
    2. Applies all 4 smoothing filters
    3. Benchmarks execution speed (100 iterations each)
    4. Benchmarks quality using PSNR and SSIM metrics
    5. Analyzes edge preservation capability
    6. Displays visual comparisons
    7. Generates comprehensive report

    Prints:
        - Progress headers for each benchmarking section
        - Detailed metrics and statistics
        - Recommendations for filter selection
        - Report file path
    """
    print("\n" + "="*70)
    print("OPENCV FILTER BENCHMARKING - HANDS-ON EXERCISES")
    print("="*70)

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'test_image')
    output_dir = os.path.join(current_dir, 'output_benchmarking')

    # 1. Load test images
    clean, noisy = load_test_images(image_path)

    # 2. Benchmark speed
    filtered_results, times = benchmark_speed(noisy, kernel_size=5, iterations=100)

    # 3. Benchmark quality
    metrics = benchmark_quality(clean, filtered_results, output_dir)

    # 4. Analyze edge preservation
    edge_analysis = analyze_edge_preservation(clean, noisy, filtered_results, output_dir)

    # 5. Display visual comparison
    print("\n" + "="*70)
    print("VISUAL COMPARISON")
    print("="*70)
    display_comparison(clean, noisy, filtered_results)

    # 6. Create comprehensive report
    create_comparison_report(times, metrics, edge_analysis, output_dir)

    print("\n" + "="*70)
    print("✓ ALL BENCHMARKING EXERCISES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == "__main__":
    main()
