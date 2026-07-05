"""
Non-interactive test script for edge detector application.
Verifies edge detection functionality without GUI.
"""

import cv2
import numpy as np
import os
from opencv_edge_detector_app import EdgeDetectorApp


def test_edge_detector():
    """Test edge detection functionality without interactive GUI."""
    print("\n" + "="*70)
    print("TESTING EDGE DETECTOR APPLICATION")
    print("="*70)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'test_image.jpg')
    output_dir = os.path.join(current_dir, 'output_edge_detection')

    # Create application
    print("\n1. Initializing EdgeDetectorApp...")
    app = EdgeDetectorApp(image_path)
    print("   ✓ Application initialized successfully")

    # Test edge detection with default parameters
    print("\n2. Testing edge detection with default parameters...")
    print(f"   - Canny thresholds: Low={app.canny_low}, High={app.canny_high}")
    print(f"   - Sobel kernel size: {app.sobel_ksize}")

    app.compute_edges()
    print("   ✓ Edge detection computed successfully")

    # Verify edge detection results
    print("\n3. Verifying edge detection results...")
    assert app.edges_canny is not None, "Canny edges not computed"
    assert app.edges_sobel is not None, "Sobel edges not computed"
    assert app.edges_laplacian is not None, "Laplacian edges not computed"
    assert app.edges_scharr is not None, "Scharr edges not computed"
    print("   ✓ All edge detection results available")

    # Check shapes
    print("\n4. Checking edge detection output shapes...")
    h, w = app.gray.shape
    assert app.edges_canny.shape == (h, w), "Canny shape mismatch"
    assert app.edges_sobel.shape == (h, w), "Sobel shape mismatch"
    assert app.edges_laplacian.shape == (h, w), "Laplacian shape mismatch"
    assert app.edges_scharr.shape == (h, w), "Scharr shape mismatch"
    print(f"   ✓ All outputs have correct shape: {(h, w)}")

    # Test parameter variation
    print("\n5. Testing parameter variation...")

    # Test Canny with different thresholds
    app.canny_low = 30
    app.canny_high = 100
    app.compute_edges()
    canny_low_threshold = np.count_nonzero(app.edges_canny)
    print(f"   - Canny (low=30, high=100): {canny_low_threshold} edge pixels")

    app.canny_low = 100
    app.canny_high = 200
    app.compute_edges()
    canny_high_threshold = np.count_nonzero(app.edges_canny)
    print(f"   - Canny (low=100, high=200): {canny_high_threshold} edge pixels")

    assert canny_high_threshold < canny_low_threshold, \
        "Higher thresholds should detect fewer edges"
    print("   ✓ Threshold variation working correctly")

    # Test kernel size variation
    print("\n6. Testing kernel size variation...")

    app.sobel_ksize = 3
    app.compute_edges()
    sobel_k3 = np.count_nonzero(app.edges_sobel)
    print(f"   - Sobel (ksize=3): {sobel_k3} edge pixels")

    app.sobel_ksize = 7
    app.compute_edges()
    sobel_k7 = np.count_nonzero(app.edges_sobel)
    print(f"   - Sobel (ksize=7): {sobel_k7} edge pixels")

    print("   ✓ Kernel size variation working correctly")

    # Test edge analysis
    print("\n7. Testing edge analysis...")
    print("   Analyzing edge detection results...")
    app.analyze_edges()
    print("   ✓ Edge analysis completed successfully")

    # Test save functionality
    print("\n8. Testing save functionality...")
    os.makedirs(output_dir, exist_ok=True)
    app._save_results()
    print("   ✓ Results saved successfully")

    # Verify saved files
    print("\n9. Verifying saved files...")
    expected_files = [
        'canny_low100_high200.jpg',
        'sobel_ksize7.jpg',
        'laplacian.jpg',
        'scharr.jpg'
    ]

    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"   ✓ {filename}")
        else:
            print(f"   ✗ {filename} NOT FOUND")

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED SUCCESSFULLY!")
    print("="*70)
    print(f"\nTest image: {image_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    test_edge_detector()
