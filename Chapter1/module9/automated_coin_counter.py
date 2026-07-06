import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt


class AutomatedCoinCounter:
    def __init__(self, output_dir='./output_coins'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_or_generate_image(self, image_path=None, width=800, height=600):
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unable to load image from {image_path}")
            return img
        return self.generate_coin_image(width, height)

    def generate_coin_image(self, width=800, height=600):
        img = np.ones((height, width, 3), dtype=np.uint8) * 180

        np.random.seed(42)
        num_coins = np.random.randint(5, 12)

        for _ in range(num_coins):
            x = np.random.randint(80, width - 80)
            y = np.random.randint(80, height - 80)
            radius = np.random.randint(25, 50)

            color = (np.random.randint(50, 100), np.random.randint(100, 150),
                    np.random.randint(100, 180))
            cv2.circle(img, (x, y), radius, color, -1)

            highlight_x = x - radius // 3
            highlight_y = y - radius // 3
            cv2.circle(img, (highlight_x, highlight_y), radius // 4,
                      (255, 255, 200), -1)

            shadow_offset = int(radius * 0.7)
            cv2.ellipse(img, (x + 5, y + 10), (shadow_offset, shadow_offset // 3),
                       0, 0, 180, (0, 0, 0), -1)

        cv2.imwrite(os.path.join(self.output_dir, 'generated_coins.png'), img)
        return img

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        return gray, blurred

    def threshold_image(self, blurred):
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def apply_morphology(self, binary, kernel_size=9):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return closed

    def find_coins(self, binary, min_area=500):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        coins = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    radius = int(radius)

                    coins.append({
                        'contour': contour,
                        'area': area,
                        'centroid': (cx, cy),
                        'radius': radius,
                        'circle_center': (int(x), int(y))
                    })

        return coins

    def visualize_coins(self, img, coins, title_suffix=''):
        result = img.copy()

        for coin in coins:
            cx, cy = coin['centroid']
            radius = coin['radius']

            cv2.circle(result, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(result, f"({cx}, {cy})", (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        coin_count = len(coins)
        cv2.putText(result, f"Coins Detected: {coin_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        output_path = os.path.join(self.output_dir, f'coin_detection{title_suffix}.png')
        cv2.imwrite(output_path, result)

        return result

    def create_analysis_dashboard(self, img, binary, closed, result, coins):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Coin Counter: Complete Analysis Pipeline', fontsize=16, fontweight='bold')

        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title(f'Otsu Threshold (Binary Inverted)')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(closed, cmap='gray')
        axes[1, 0].set_title('After Morphological Closing')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Detected Coins: {len(coins)}')
        axes[1, 1].axis('off')

        plt.tight_layout()
        dashboard_path = os.path.join(self.output_dir, 'coin_analysis_dashboard.png')
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()

    def print_statistics(self, coins, img_shape):
        print("\n" + "="*70)
        print("AUTOMATED COIN COUNTER - ANALYSIS REPORT")
        print("="*70)
        print(f"Image Resolution: {img_shape[1]}x{img_shape[0]} pixels")
        print(f"Total Coins Detected: {len(coins)}")
        print("-"*70)

        if coins:
            areas = [c['area'] for c in coins]
            radii = [c['radius'] for c in coins]

            print(f"\nCOIN STATISTICS:")
            print(f"  Average Area: {np.mean(areas):.2f} px²")
            print(f"  Min Area: {np.min(areas):.2f} px²")
            print(f"  Max Area: {np.max(areas):.2f} px²")
            print(f"\n  Average Radius: {np.mean(radii):.2f} px")
            print(f"  Min Radius: {np.min(radii):.2f} px")
            print(f"  Max Radius: {np.max(radii):.2f} px")

            print(f"\nCOIN CENTROIDS (Spatial Coordinates):")
            for i, coin in enumerate(coins, 1):
                cx, cy = coin['centroid']
                area = coin['area']
                radius = coin['radius']
                print(f"  Coin {i:2d}: ({cx:4d}, {cy:4d}) | "
                      f"Area: {area:6.0f} px² | Radius: {radius:3d} px")

        print("-"*70)
        print(f"Output Directory: {os.path.abspath(self.output_dir)}")
        print("="*70 + "\n")

    def save_report(self, coins, img_shape, processing_time):
        report_path = os.path.join(self.output_dir, 'coin_analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("AUTOMATED COIN COUNTER - DETAILED ANALYSIS REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image Resolution: {img_shape[1]}x{img_shape[0]} pixels\n")
            f.write(f"Processing Time: {processing_time:.2f} seconds\n")
            f.write(f"Total Coins Detected: {len(coins)}\n")
            f.write("-"*70 + "\n")

            if coins:
                areas = [c['area'] for c in coins]
                radii = [c['radius'] for c in coins]

                f.write("\nCOIN STATISTICS:\n")
                f.write(f"  Average Area: {np.mean(areas):.2f} px²\n")
                f.write(f"  Std Dev Area: {np.std(areas):.2f} px²\n")
                f.write(f"  Min Area: {np.min(areas):.2f} px²\n")
                f.write(f"  Max Area: {np.max(areas):.2f} px²\n")
                f.write(f"\n  Average Radius: {np.mean(radii):.2f} px\n")
                f.write(f"  Std Dev Radius: {np.std(radii):.2f} px\n")
                f.write(f"  Min Radius: {np.min(radii):.2f} px\n")
                f.write(f"  Max Radius: {np.max(radii):.2f} px\n")

                f.write(f"\nDETECTED COIN LOCATIONS:\n")
                for i, coin in enumerate(coins, 1):
                    cx, cy = coin['centroid']
                    area = coin['area']
                    radius = coin['radius']
                    f.write(f"  Coin {i:2d}: Position=({cx:4d}, {cy:4d}) | "
                           f"Area={area:6.0f} px² | Radius={radius:3d} px\n")

                f.write(f"\nCOORDINATE BOUNDS:\n")
                x_coords = [c['centroid'][0] for c in coins]
                y_coords = [c['centroid'][1] for c in coins]
                f.write(f"  X Range: {min(x_coords)} - {max(x_coords)} px\n")
                f.write(f"  Y Range: {min(y_coords)} - {max(y_coords)} px\n")

            f.write("-"*70 + "\n")

    def run_analysis(self, image_path=None, min_area=500, kernel_size=9):
        import time
        start_time = time.time()

        img = self.load_or_generate_image(image_path)
        _, blurred = self.preprocess_image(img)
        binary = self.threshold_image(blurred)
        closed = self.apply_morphology(binary, kernel_size)
        coins = self.find_coins(closed, min_area)
        result = self.visualize_coins(img, coins)

        self.create_analysis_dashboard(img, binary, closed, result, coins)
        self.print_statistics(coins, img.shape)

        processing_time = time.time() - start_time
        self.save_report(coins, img.shape, processing_time)

        cv2.imwrite(os.path.join(self.output_dir, 'original_image.png'), img)
        cv2.imwrite(os.path.join(self.output_dir, 'binary_threshold.png'), binary)
        cv2.imwrite(os.path.join(self.output_dir, 'morphological_closing.png'), closed)

        return coins, result


def main():
    parser = argparse.ArgumentParser(
        description='Automated Coin Counter - Detect and count coins in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate synthetic coin image and detect coins
  python automated_coin_counter.py

  # Use custom image with default settings
  python automated_coin_counter.py --custom coins.jpg

  # Use custom image with optimized parameters
  python automated_coin_counter.py --custom coins.jpg --min-area 300 --kernel-size 7

  # Save results to specific directory
  python automated_coin_counter.py --custom coins.jpg --output-dir ./results
        '''
    )

    parser.add_argument('--custom', type=str, default=None,
                       help='Path to custom coin image (JPG, PNG, etc.)')
    parser.add_argument('--min-area', type=int, default=500,
                       help='Minimum coin area threshold in pixels (default: 500)')
    parser.add_argument('--kernel-size', type=int, default=9,
                       help='Kernel size for morphological operations (default: 9)')
    parser.add_argument('--output-dir', type=str, default='./output_coins',
                       help='Output directory for results (default: ./output_coins)')

    args = parser.parse_args()

    counter = AutomatedCoinCounter(args.output_dir)
    counter.run_analysis(args.custom, args.min_area, args.kernel_size)


if __name__ == '__main__':
    main()
