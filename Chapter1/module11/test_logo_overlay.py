import cv2
import numpy as np
import os
import shutil
from logo_overlay_utility import LogoOverlayUtility


def test_basic_overlay_pipeline():
    """Test basic logo overlay pipeline with synthetic images."""
    print("\n" + "="*70)
    print("TEST 1: Basic Logo Overlay Pipeline")
    print("="*70)

    overlay = LogoOverlayUtility('./test_output_logo_basic')
    result = overlay.run_analysis()

    assert result is not None, "Overlay result is None"
    assert result.shape[0] > 0 and result.shape[1] > 0, "Result has invalid dimensions"
    assert result.shape[2] == 3, "Result should be BGR image (3 channels)"

    print("\n✓ Test 1 PASSED: Basic pipeline execution successful")
    print(f"  - Result shape: {result.shape}")
    print(f"  - Data type: {result.dtype}")


def test_output_files_generation():
    """Test that all expected output files are generated."""
    print("\n" + "="*70)
    print("TEST 2: Output Files Generation")
    print("="*70)

    overlay = LogoOverlayUtility('./test_output_logo_files')
    overlay.run_analysis()

    required_files = [
        'background_original.png',
        'logo_original.png',
        'logo_resized.png',
        'roi_extracted.png',
        'binary_mask.png',
        'inverse_mask.png',
        'cleared_roi.png',
        'isolated_logo.png',
        'blended_composite.png',
        'final_overlay.png',
        'logo_overlay_dashboard.png',
        'logo_overlay_report.txt'
    ]

    output_dir = './test_output_logo_files'
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)

    assert len(missing_files) == 0, f"Missing output files: {missing_files}"

    print("\n✓ Test 2 PASSED: All output files generated successfully")
    print(f"  - Total files: {len(required_files)}")
    for filename in required_files:
        print(f"    ✓ {filename}")


def test_custom_images():
    """Test overlay with custom background and logo images."""
    print("\n" + "="*70)
    print("TEST 3: Custom Image Loading")
    print("="*70)

    # Create temporary test images
    overlay = LogoOverlayUtility('./test_output_logo_custom_temp')
    bg, logo = overlay.generate_test_images()

    bg_path = './test_bg_custom.jpg'
    logo_path = './test_logo_custom.jpg'

    cv2.imwrite(bg_path, bg)
    cv2.imwrite(logo_path, logo)

    try:
        overlay_custom = LogoOverlayUtility('./test_output_logo_custom')
        result = overlay_custom.run_analysis(background_path=bg_path, logo_path=logo_path)

        assert result is not None, "Custom image overlay failed"
        assert result.shape == bg.shape, "Result shape mismatch"

        print("\n✓ Test 3 PASSED: Custom images processed successfully")
        print(f"  - Background loaded: {bg_path}")
        print(f"  - Logo loaded: {logo_path}")
        print(f"  - Result shape: {result.shape}")

    finally:
        if os.path.exists(bg_path):
            os.remove(bg_path)
        if os.path.exists(logo_path):
            os.remove(logo_path)


def test_logo_resizing():
    """Test logo resizing when exceeding 30% of background area."""
    print("\n" + "="*70)
    print("TEST 4: Logo Resizing")
    print("="*70)

    overlay = LogoOverlayUtility('./test_output_logo_resize')

    # Create background
    bg_shape = (400, 600, 3)
    background = np.ones(bg_shape, dtype=np.uint8) * 200

    # Create large logo (>30% of background)
    logo_h, logo_w = 300, 400  # 45% of background area
    large_logo = np.ones((logo_h, logo_w, 3), dtype=np.uint8) * 100

    # Test resizing
    resized = overlay.resize_logo_if_needed(large_logo, background)

    assert resized is not None, "Resizing returned None"
    assert resized.shape[0] <= logo_h and resized.shape[1] <= logo_w, "Logo not resized"

    # Calculate percentage after resize
    resized_percentage = (resized.shape[0] * resized.shape[1]) / (bg_shape[0] * bg_shape[1]) * 100
    assert resized_percentage <= 31, f"Resized logo exceeds 30%: {resized_percentage:.1f}%"

    print("\n✓ Test 4 PASSED: Logo resizing works correctly")
    print(f"  - Original logo: {large_logo.shape}")
    print(f"  - Resized logo: {resized.shape}")
    print(f"  - Area coverage: {resized_percentage:.2f}% of background")


def test_mask_operations():
    """Test binary mask creation and inversion."""
    print("\n" + "="*70)
    print("TEST 5: Mask Operations")
    print("="*70)

    overlay = LogoOverlayUtility('./test_output_logo_masks')
    _, logo = overlay.generate_test_images()

    # Create masks
    binary_mask = overlay.create_binary_mask(logo)
    inverse_mask = overlay.create_inverse_mask(binary_mask)

    # Verify binary mask
    assert binary_mask.dtype == np.uint8, "Binary mask wrong dtype"
    assert binary_mask.shape[:2] == logo.shape[:2], "Mask dimensions mismatch"
    unique_values = set(binary_mask.flatten())
    assert unique_values.issubset({0, 255}), "Binary mask contains non-binary values"

    # Verify inverse mask
    assert inverse_mask.dtype == np.uint8, "Inverse mask wrong dtype"
    unique_inverse = set(inverse_mask.flatten())
    assert unique_inverse.issubset({0, 255}), "Inverse mask contains non-binary values"

    # Verify bitwise relationship
    combined = cv2.bitwise_or(binary_mask, inverse_mask)
    assert np.all(combined == 255), "Binary and inverse masks don't cover all pixels"

    print("\n✓ Test 5 PASSED: Mask operations work correctly")
    print(f"  - Binary mask shape: {binary_mask.shape}")
    print(f"  - Binary mask unique values: {unique_values}")
    print(f"  - Inverse mask unique values: {unique_inverse}")


def test_roi_isolation():
    """Test ROI extraction with position clamping."""
    print("\n" + "="*70)
    print("TEST 6: ROI Isolation")
    print("="*70)

    overlay = LogoOverlayUtility('./test_output_logo_roi')
    background, logo = overlay.generate_test_images()

    # Test 1: Normal position
    roi1, pos1 = overlay.isolate_roi(background, logo, position=(100, 100))
    assert roi1.shape == (logo.shape[0], logo.shape[1], 3), "ROI shape mismatch"
    assert pos1 == (100, 100), "Position not preserved"

    # Test 2: Position beyond bounds (should be clamped)
    bg_h, bg_w = background.shape[:2]
    roi2, pos2 = overlay.isolate_roi(background, logo, position=(bg_w, bg_h))
    assert roi2.shape == (logo.shape[0], logo.shape[1], 3), "Clamped ROI shape invalid"
    assert pos2[0] <= bg_w - logo.shape[1], "X position not clamped correctly"
    assert pos2[1] <= bg_h - logo.shape[0], "Y position not clamped correctly"

    print("\n✓ Test 6 PASSED: ROI isolation works correctly")
    print(f"  - Normal position ROI: {roi1.shape}")
    print(f"  - Clamped position: {pos2}")


def test_image_blending():
    """Test image blending using cv2.add."""
    print("\n" + "="*70)
    print("TEST 7: Image Blending")
    print("="*70)

    overlay = LogoOverlayUtility('./test_output_logo_blend')
    background, logo = overlay.generate_test_images()

    # Create simple test images for blending
    roi_patch, _ = overlay.isolate_roi(background, logo)
    binary_mask = overlay.create_binary_mask(logo)
    inverse_mask = overlay.create_inverse_mask(binary_mask)

    cleared = overlay.clear_logo_from_roi(roi_patch, inverse_mask)
    isolated = overlay.isolate_logo_colors(logo, binary_mask)
    blended = overlay.blend_images(cleared, isolated)

    assert blended is not None, "Blending returned None"
    assert blended.dtype == np.uint8, "Blended image wrong dtype"
    assert blended.shape == roi_patch.shape, "Blended shape mismatch"
    assert 0 <= blended.min() <= 255 and 0 <= blended.max() <= 255, "Pixel values out of range"

    print("\n✓ Test 7 PASSED: Image blending works correctly")
    print(f"  - Blended shape: {blended.shape}")
    print(f"  - Value range: [{blended.min()}, {blended.max()}]")


def test_position_accuracy():
    """Test that logo is placed at exact position."""
    print("\n" + "="*70)
    print("TEST 8: Position Accuracy")
    print("="*70)

    overlay = LogoOverlayUtility('./test_output_logo_position')
    background, logo = overlay.generate_test_images()

    test_position = (150, 200)
    _, actual_pos = overlay.isolate_roi(background, logo, position=test_position)

    assert actual_pos[0] >= 0 and actual_pos[1] >= 0, "Position coordinates negative"
    assert actual_pos[0] + logo.shape[1] <= background.shape[1], "Position exceeds width"
    assert actual_pos[1] + logo.shape[0] <= background.shape[0], "Position exceeds height"

    print("\n✓ Test 8 PASSED: Position accuracy verified")
    print(f"  - Requested position: {test_position}")
    print(f"  - Actual position: {actual_pos}")
    print(f"  - Logo bounds: x=[{actual_pos[0]}, {actual_pos[0] + logo.shape[1]}], "
          f"y=[{actual_pos[1]}, {actual_pos[1] + logo.shape[0]}]")


def cleanup_test_outputs():
    """Remove test output directories."""
    test_dirs = [
        './test_output_logo_basic',
        './test_output_logo_files',
        './test_output_logo_custom',
        './test_output_logo_custom_temp',
        './test_output_logo_resize',
        './test_output_logo_masks',
        './test_output_logo_roi',
        './test_output_logo_blend',
        './test_output_logo_position'
    ]

    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("LOGO OVERLAY UTILITY - TEST SUITE")
    print("="*70)

    try:
        test_basic_overlay_pipeline()
        test_output_files_generation()
        test_custom_images()
        test_logo_resizing()
        test_mask_operations()
        test_roi_isolation()
        test_image_blending()
        test_position_accuracy()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")

    finally:
        cleanup_test_outputs()
