import cv2
import os
from roi_extraction_utility import ROIExtractionUtility


def test_roi_generation_and_extraction():
    """Test ROI generation and extraction with synthetic image."""
    print("\n" + "="*70)
    print("TEST 1: ROI Generation and Extraction")
    print("="*70)

    extractor = ROIExtractionUtility('./test_output_roi')
    face_roi, circular_roi, polygon_roi = extractor.run_analysis()

    assert face_roi is not None, "Face ROI extraction failed"
    assert circular_roi is not None, "Circular ROI extraction failed"
    assert polygon_roi is not None, "Polygon ROI extraction failed"

    assert face_roi.shape[0] > 0 and face_roi.shape[1] > 0, "Face ROI has invalid dimensions"
    assert circular_roi.shape[0] > 0 and circular_roi.shape[1] > 0, "Circular ROI has invalid dimensions"
    assert polygon_roi.shape[0] > 0 and polygon_roi.shape[1] > 0, "Polygon ROI has invalid dimensions"

    print("\n✓ Test 1 PASSED: ROI extraction works correctly")
    print(f"  - Face ROI shape: {face_roi.shape}")
    print(f"  - Circular ROI shape: {circular_roi.shape}")
    print(f"  - Polygon ROI shape: {polygon_roi.shape}")


def test_output_files_creation():
    """Test that all output files are created."""
    print("\n" + "="*70)
    print("TEST 2: Output Files Creation")
    print("="*70)

    extractor = ROIExtractionUtility('./test_output_roi_files')
    extractor.run_analysis()

    required_files = [
        'original_image.png',
        'face_roi.png',
        'face_mask.png',
        'circular_roi.png',
        'circular_mask.png',
        'polygon_roi.png',
        'polygon_mask.png',
        'roi_extraction_dashboard.png',
        'roi_extraction_report.txt',
        'generated_test_image.png'
    ]

    output_dir = './test_output_roi_files'
    for filename in required_files:
        filepath = os.path.join(output_dir, filename)
        assert os.path.exists(filepath), f"Missing output file: {filename}"

    print("\n✓ Test 2 PASSED: All output files created successfully")
    print(f"  - Generated {len(required_files)} output files")


def test_custom_image_loading():
    """Test custom image loading capability."""
    print("\n" + "="*70)
    print("TEST 3: Custom Image Loading")
    print("="*70)

    # Create a simple test image
    test_img = cv2.imread(os.path.join('./test_output_roi', 'generated_test_image.png'))
    if test_img is not None:
        test_img_path = './test_custom_roi.jpg'
        cv2.imwrite(test_img_path, test_img)

        try:
            extractor = ROIExtractionUtility('./test_output_roi_custom')
            face_roi, _, _ = extractor.run_analysis(test_img_path)

            assert face_roi is not None, "Custom image ROI extraction failed"
            print("\n✓ Test 3 PASSED: Custom image loading works correctly")
            print(f"  - Successfully loaded and processed custom image")

        finally:
            if os.path.exists(test_img_path):
                os.remove(test_img_path)


def test_roi_dimensions():
    """Test ROI dimension validity."""
    print("\n" + "="*70)
    print("TEST 4: ROI Dimension Validation")
    print("="*70)

    extractor = ROIExtractionUtility('./test_output_roi_dims')
    img = extractor.generate_test_image()

    # Test face ROI with custom coordinates
    face_x, face_y, face_w, face_h = 100, 100, 150, 200
    face_roi = extractor.extract_face_roi(img, face_x, face_y, face_w, face_h)

    assert face_roi.shape[1] == face_w, "Face ROI width mismatch"
    assert face_roi.shape[0] == face_h, "Face ROI height mismatch"

    # Test circular ROI
    center_x, center_y, radius = 300, 200, 100
    circular_roi, circular_mask = extractor.extract_circular_roi(img, center_x, center_y, radius)

    assert circular_mask.shape == img.shape[:2], "Circular mask dimensions mismatch"
    assert circular_roi.shape == img.shape, "Circular ROI dimensions mismatch"

    # Test polygon ROI
    polygon_vertices = [[50, 50], [200, 50], [200, 200], [50, 200]]
    polygon_roi, polygon_mask = extractor.extract_polygon_roi(img, polygon_vertices)

    assert polygon_mask.shape == img.shape[:2], "Polygon mask dimensions mismatch"
    assert polygon_roi.shape == img.shape, "Polygon ROI dimensions mismatch"

    print("\n✓ Test 4 PASSED: All ROI dimensions are valid")
    print(f"  - Face ROI: {face_roi.shape}")
    print(f"  - Circular ROI: {circular_roi.shape}")
    print(f"  - Polygon ROI: {polygon_roi.shape}")


def test_mask_operations():
    """Test mask operations and bitwise AND."""
    print("\n" + "="*70)
    print("TEST 5: Mask Operations")
    print("="*70)

    extractor = ROIExtractionUtility('./test_output_roi_masks')
    img = extractor.generate_test_image()

    # Test circular mask
    _, circular_mask = extractor.extract_circular_roi(img, 300, 200, 100)
    assert circular_mask.dtype == 'uint8', "Circular mask has incorrect dtype"
    assert circular_mask.min() >= 0 and circular_mask.max() <= 255, "Circular mask values out of range"

    # Test polygon mask
    polygon_vertices = [[50, 50], [200, 50], [200, 200], [50, 200]]
    _, polygon_mask = extractor.extract_polygon_roi(img, polygon_vertices)
    assert polygon_mask.dtype == 'uint8', "Polygon mask has incorrect dtype"
    assert polygon_mask.min() >= 0 and polygon_mask.max() <= 255, "Polygon mask values out of range"

    # Test that masks are binary (0 or 255)
    unique_vals = set(polygon_mask.flatten())
    for val in unique_vals:
        assert val in [0, 255], f"Polygon mask contains non-binary value: {val}"

    print("\n✓ Test 5 PASSED: Mask operations work correctly")
    print(f"  - Circular mask unique values: {set(circular_mask.flatten())}")
    print(f"  - Polygon mask unique values: {set(polygon_mask.flatten())}")


def cleanup_test_outputs():
    """Remove test output directories."""
    import shutil
    test_dirs = [
        './test_output_roi',
        './test_output_roi_files',
        './test_output_roi_custom',
        './test_output_roi_dims',
        './test_output_roi_masks'
    ]

    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ROI EXTRACTION UTILITY - TEST SUITE")
    print("="*70)

    try:
        test_roi_generation_and_extraction()
        test_output_files_creation()
        test_custom_image_loading()
        test_roi_dimensions()
        test_mask_operations()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")

    finally:
        cleanup_test_outputs()
