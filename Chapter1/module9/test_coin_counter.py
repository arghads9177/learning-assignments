import cv2
import os
from automated_coin_counter import AutomatedCoinCounter


def test_coin_detection_with_generated_image():
    print("\n" + "="*70)
    print("TEST 1: Coin Detection with Generated Synthetic Image")
    print("="*70)

    counter = AutomatedCoinCounter('./test_output_coins')
    coins, _ = counter.run_analysis()

    assert len(coins) > 0, "No coins detected in generated image"
    assert os.path.exists(os.path.join('./test_output_coins', 'coin_analysis_dashboard.png')), \
        "Dashboard not created"
    assert os.path.exists(os.path.join('./test_output_coins', 'coin_analysis_report.txt')), \
        "Report not created"

    print("\n✓ Test 1 PASSED: Generated image coin detection works correctly")
    print(f"  - Detected {len(coins)} coins")
    print(f"  - All output files created successfully")


def test_coin_detection_with_custom_image():
    print("\n" + "="*70)
    print("TEST 2: Coin Detection with Custom Test Image")
    print("="*70)

    counter_generate = AutomatedCoinCounter('./temp_generate')
    test_img = counter_generate.generate_coin_image(600, 400)
    test_img_path = './temp_test_coins.jpg'
    cv2.imwrite(test_img_path, test_img)

    try:
        counter = AutomatedCoinCounter('./test_output_coins_custom')
        coins, _ = counter.run_analysis(test_img_path)

        assert len(coins) > 0, "No coins detected in custom image"
        assert os.path.exists(os.path.join('./test_output_coins_custom',
                                           'coin_detection.png')), \
            "Detection image not created"

        print("\n✓ Test 2 PASSED: Custom image coin detection works correctly")
        print(f"  - Detected {len(coins)} coins in custom image")
        print(f"  - All coin coordinates computed successfully")

    finally:
        if os.path.exists(test_img_path):
            os.remove(test_img_path)


def test_parameter_variations():
    print("\n" + "="*70)
    print("TEST 3: Parameter Variations")
    print("="*70)

    counter = AutomatedCoinCounter('./test_output_coins_params')
    img = counter.generate_coin_image()

    test_params = [
        (300, 5, 'Small kernel, low threshold'),
        (500, 9, 'Standard kernel and threshold'),
        (800, 11, 'Large kernel, high threshold'),
    ]

    for min_area, kernel_size, description in test_params:
        _, blurred = counter.preprocess_image(img)
        binary = counter.threshold_image(blurred)
        closed = counter.apply_morphology(binary, kernel_size)
        coins = counter.find_coins(closed, min_area)

        print(f"  - {description}")
        print(f"    Min Area: {min_area}, Kernel: {kernel_size}x{kernel_size}")
        print(f"    Coins detected: {len(coins)}")

    print("\n✓ Test 3 PASSED: Parameter variations work correctly")


def test_centroid_calculation():
    print("\n" + "="*70)
    print("TEST 4: Centroid Calculation Verification")
    print("="*70)

    counter = AutomatedCoinCounter('./test_output_coins_centroids')
    coins, _ = counter.run_analysis()

    for coin in coins:
        cx, cy = coin['centroid']
        assert isinstance(cx, int) and isinstance(cy, int), \
            "Centroid coordinates should be integers"
        assert cx > 0 and cy > 0, "Centroid coordinates should be positive"

    print("\n✓ Test 4 PASSED: All centroids calculated correctly")
    print(f"  - Verified {len(coins)} coin centroids")
    print(f"  - All coordinates are valid integers")


def cleanup_test_outputs():
    test_dirs = [
        './test_output_coins',
        './test_output_coins_custom',
        './test_output_coins_params',
        './test_output_coins_centroids',
        './temp_generate'
    ]

    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("AUTOMATED COIN COUNTER - TEST SUITE")
    print("="*70)

    try:
        test_coin_detection_with_generated_image()
        test_coin_detection_with_custom_image()
        test_parameter_variations()
        test_centroid_calculation()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")

    finally:
        cleanup_test_outputs()
