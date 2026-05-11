import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import pad_binary_image, convert_to_grayscale, ensure_binary

def visualize_padding(original, padded, pad_h, pad_w, title="Padding Result"):
    """Visualize original vs padded images with boundary indicators"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'Original Image\nShape: {original.shape}', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Add grid and pixel values for small images
    if original.size <= 100:
        for i in range(original.shape[0] + 1):
            axes[0].axhline(y=i - 0.5, color='blue', linewidth=1, alpha=0.3)
            axes[0].axvline(x=i - 0.5, color='blue', linewidth=1, alpha=0.3)
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                color = 'white' if original[i, j] < 128 else 'black'
                axes[0].text(j, i, str(original[i, j]), ha='center', va='center', 
                           color=color, fontsize=8, fontweight='bold')
    
    # Padded image
    axes[1].imshow(padded, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Padded Image\nShape: {padded.shape} | pad_h={pad_h}, pad_w={pad_w}', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Highlight the original image region with a rectangle
    rect = Rectangle((pad_w - 0.5, pad_h - 0.5), 
                     original.shape[1], original.shape[0], 
                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    axes[1].add_patch(rect)
    
    # Add grid for small images
    if padded.size <= 400:
        for i in range(padded.shape[0] + 1):
            axes[1].axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.2)
            axes[1].axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.2)
    
    # Add padding zone labels
    if pad_h > 0:
        # Top padding
        axes[1].text(padded.shape[1]/2, pad_h/2, f'Top Padding\n({pad_h} px)', 
                    ha='center', va='center', fontsize=9, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        # Bottom padding
        axes[1].text(padded.shape[1]/2, padded.shape[0] - pad_h/2, f'Bottom Padding\n({pad_h} px)', 
                    ha='center', va='center', fontsize=9, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    if pad_w > 0:
        # Left padding
        axes[1].text(pad_w/2, padded.shape[0]/2, f'Left\n({pad_w})', 
                    ha='center', va='center', fontsize=9, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        # Right padding
        axes[1].text(padded.shape[1] - pad_w/2, padded.shape[0]/2, f'Right\n({pad_w})', 
                    ha='center', va='center', fontsize=9, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_with_custom_image():
    """Load and pad a custom image"""
    
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("No file selected.")
        return
    
    try:
        # Load the image
        pil_image = Image.open(file_path)
        print(f"\nLoaded: {os.path.basename(file_path)}")
        print(f"Original mode: {pil_image.mode}")
        print(f"Original size: {pil_image.size}")
        
        # Convert to numpy array
        original_array = np.array(pil_image)
        
        # Convert to grayscale and ensure binary if color
        if len(original_array.shape) == 3:
            print("Converting to grayscale and binarizing...")
            grayscale = convert_to_grayscale(original_array)
            binary = ensure_binary(grayscale)
        else:
            binary = ensure_binary(original_array)
        
        print(f"Binary image shape: {binary.shape}")
        
        # Get padding values
        while True:
            try:
                print("\nEnter padding values (number of pixels):")
                pad_h = int(input("  Height padding (top & bottom): ").strip())
                pad_w = int(input("  Width padding (left & right): ").strip())
                
                if pad_h >= 0 and pad_w >= 0:
                    break
                else:
                    print("Padding values must be non-negative")
            except ValueError:
                print("Please enter valid numbers")
        
        # Apply padding
        padded = pad_binary_image(binary, pad_h, pad_w)
        
        print(f"\nPadding Results:")
        print(f"  Original shape: {binary.shape}")
        print(f"  Padded shape: {padded.shape}")
        print(f"  Total padding: {np.sum(padded == 0) - np.sum(binary == 0)} background pixels added")
        print(f"  Original pixels: {np.sum(padded == 255)}")
        
        # Visualize
        visualize_padding(binary, padded, pad_h, pad_w,
                         f"Padding: {os.path.basename(file_path)}")
        
        # Save option
        save_option = input("\nDo you want to save the padded image? (y/n): ").lower().strip()
        if save_option == 'y':
            save_path = filedialog.asksaveasfilename(
                title="Save padded image",
                defaultextension=".png",
                filetypes=[("PNG file", "*.png"), ("JPEG file", "*.jpg"), ("BMP file", "*.bmp")]
            )
            if save_path:
                Image.fromarray(padded).save(save_path)
                print(f"Image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_with_test_patterns():
    """Demonstrate padding with various test patterns"""
    
    while True:
        print("\n" + "="*60)
        print("PAD BINARY IMAGE VISUALIZER")
        print("Using: binary_morphology.pad_binary_image(image, pad_h, pad_w)")
        print("="*60)
        print("1. Upload and pad your own image")
        print("2. Test with checkerboard pattern")
        print("3. Test with letter pattern (F shape)")
        print("4. Interactive padding explorer")
        print("5. Compare different padding amounts")
        print("6. Asymmetric padding demo (using different h/w)")
        print("7. Show function source code")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            test_with_custom_image()
        
        elif choice == '2':
            # Checkerboard pattern
            print("\nCreating checkerboard pattern...")
            size = 4
            checkerboard = np.zeros((size, size), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        checkerboard[i, j] = 255
            
            print("Original checkerboard:")
            print(checkerboard)
            print(f"Shape: {checkerboard.shape}")
            
            # Get padding
            while True:
                try:
                    pad_h = int(input("Enter height padding (default=2): ").strip() or "2")
                    pad_w = int(input("Enter width padding (default=2): ").strip() or "2")
                    if pad_h >= 0 and pad_w >= 0:
                        break
                    else:
                        print("Padding must be non-negative")
                except ValueError:
                    print("Please enter a valid number")
            
            padded = pad_binary_image(checkerboard, pad_h, pad_w)
            
            print(f"\nOriginal shape: {checkerboard.shape}")
            print(f"Padded shape: {padded.shape}")
            print(f"pad_h={pad_h}, pad_w={pad_w}")
            print("\nPadded image matrix:")
            print(padded)
            
            visualize_padding(checkerboard, padded, pad_h, pad_w,
                            f"Checkerboard with pad_h={pad_h}, pad_w={pad_w}")
        
        elif choice == '3':
            # Letter F pattern
            print("\nCreating letter 'F' pattern...")
            F_pattern = np.array([
                [255, 255, 255, 255],
                [255, 0, 0, 0],
                [255, 255, 255, 0],
                [255, 0, 0, 0],
                [255, 0, 0, 0],
            ], dtype=np.uint8)
            
            print("Original F pattern:")
            print(F_pattern)
            print(f"Shape: {F_pattern.shape}")
            
            # Get padding
            while True:
                try:
                    pad_h = int(input("Height padding (default=2): ").strip() or "2")
                    pad_w = int(input("Width padding (default=3): ").strip() or "3")
                    
                    if pad_h >= 0 and pad_w >= 0:
                        break
                    else:
                        print("Padding values must be non-negative")
                except ValueError:
                    print("Please enter valid numbers")
            
            padded = pad_binary_image(F_pattern, pad_h, pad_w)
            
            print(f"\nOriginal shape: {F_pattern.shape}")
            print(f"Padded shape: {padded.shape}")
            print(f"pad_h={pad_h}, pad_w={pad_w}")
            print("\nPadded image matrix:")
            print(padded)
            
            visualize_padding(F_pattern, padded, pad_h, pad_w,
                            f"Letter F with pad_h={pad_h}, pad_w={pad_w}")
        
        elif choice == '4':
            # Interactive padding explorer
            print("\n" + "-"*40)
            print("INTERACTIVE PADDING EXPLORER")
            print("-"*40)
            
            # Create a simple test pattern
            test_pattern = np.array([
                [255, 0, 255],
                [0, 255, 0],
                [255, 0, 255]
            ], dtype=np.uint8)
            
            print("Test pattern (3x3 cross):")
            print(test_pattern)
            print(f"Shape: {test_pattern.shape}")
            
            while True:
                print("\nEnter padding values (-1 to quit):")
                try:
                    pad_h = int(input("  Height padding (pad_h): ").strip())
                    if pad_h == -1:
                        break
                    pad_w = int(input("  Width padding (pad_w): ").strip())
                    if pad_w == -1:
                        break
                    
                    if 0 <= pad_h <= 5 and 0 <= pad_w <= 5:
                        padded = pad_binary_image(test_pattern, pad_h, pad_w)
                        
                        print(f"\npad_h={pad_h}, pad_w={pad_w}")
                        print(f"Original shape: {test_pattern.shape}")
                        print(f"Padded shape: {padded.shape}")
                        print("Padded matrix:")
                        print(padded)
                        
                        visualize_padding(test_pattern, padded, pad_h, pad_w,
                                        f"pad_h={pad_h}, pad_w={pad_w}")
                    else:
                        print("Padding must be between 0 and 5")
                except ValueError:
                    print("Please enter valid numbers")
        
        elif choice == '5':
            # Compare different padding amounts
            print("\nComparing different padding amounts...")
            
            # Create test pattern
            test = np.ones((3, 3), dtype=np.uint8) * 255
            test[1, 1] = 0  # Center is black
            
            # Different (pad_h, pad_w) combinations
            padding_combos = [
                (0, 0, "No padding"),
                (1, 1, "pad_h=1, pad_w=1"),
                (2, 2, "pad_h=2, pad_w=2"),
                (1, 3, "pad_h=1, pad_w=3"),
                (3, 1, "pad_h=3, pad_w=1"),
            ]
            
            fig, axes = plt.subplots(1, len(padding_combos), figsize=(18, 3.5))
            
            for idx, (pad_h, pad_w, label) in enumerate(padding_combos):
                padded = pad_binary_image(test, pad_h, pad_w)
                axes[idx].imshow(padded, cmap='gray', vmin=0, vmax=255)
                axes[idx].set_title(f'{label}\nShape: {padded.shape}', fontsize=10)
                axes[idx].axis('off')
                
                # Add red rectangle for original image boundary
                if pad_h > 0 or pad_w > 0:
                    rect = Rectangle((pad_w - 0.5, pad_h - 0.5), test.shape[1], test.shape[0],
                                   linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                    axes[idx].add_patch(rect)
            
            plt.suptitle('Padding Comparison - Different (pad_h, pad_w) Combinations', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print shapes
            print("\nPadding Comparison:")
            print(f"  Original shape: {test.shape}")
            for pad_h, pad_w, label in padding_combos:
                padded = pad_binary_image(test, pad_h, pad_w)
                print(f"  {label:20s}: {test.shape} → {padded.shape}")
        
        elif choice == '6':
            # Asymmetric padding demo
            print("\n" + "-"*40)
            print("ASYMMETRIC PADDING DEMO")
            print("Using different pad_h and pad_w values")
            print("-"*40)
            
            # Create a test pattern
            pattern = np.array([
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [0, 0, 255, 255],
                [0, 0, 255, 255],
            ], dtype=np.uint8)
            
            print("Original pattern (4x4):")
            print(pattern)
            print(f"Shape: {pattern.shape}")
            
            # Asymmetric examples
            examples = [
                (1, 3, "Wider borders (h=1, w=3)"),
                (3, 1, "Taller borders (h=3, w=1)"),
                (2, 5, "Very wide borders (h=2, w=5)"),
                (5, 2, "Very tall borders (h=5, w=2)"),
            ]
            
            fig, axes = plt.subplots(1, len(examples) + 1, figsize=(18, 4))
            
            # Original
            axes[0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\n{pattern.shape}', fontsize=10)
            axes[0].axis('off')
            
            for idx, (pad_h, pad_w, label) in enumerate(examples):
                padded = pad_binary_image(pattern, pad_h, pad_w)
                axes[idx + 1].imshow(padded, cmap='gray', vmin=0, vmax=255)
                axes[idx + 1].set_title(f'{label}\nShape: {padded.shape}', fontsize=9)
                axes[idx + 1].axis('off')
                
                # Original boundary
                rect = Rectangle((pad_w - 0.5, pad_h - 0.5), pattern.shape[1], pattern.shape[0],
                               linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
                axes[idx + 1].add_patch(rect)
            
            plt.suptitle('Asymmetric Padding: Different Height vs Width Padding', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print shapes
            print("\nShape transformations:")
            print(f"  Original: {pattern.shape}")
            for pad_h, pad_w, label in examples:
                padded = pad_binary_image(pattern, pad_h, pad_w)
                print(f"  {label:35s}: {pattern.shape} → {padded.shape}")
        
        elif choice == '7':
            # Show source code
            import inspect
            print("\n" + "="*60)
            print("Function source code:")
            print("="*60)
            print(inspect.getsource(pad_binary_image))
            print("="*60)
            print(f"Function location: {inspect.getfile(pad_binary_image)}")
            print(f"Function signature: pad_binary_image(binary_image, pad_h, pad_w)")
        
        elif choice == '8':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


def quick_demo():
    """Quick demonstration of padding"""
    
    print("\n" + "="*60)
    print("QUICK DEMO - pad_binary_image(binary_image, pad_h, pad_w)")
    print("="*60)
    
    # Create test patterns
    patterns = {
        'Single Pixel': np.array([[255]], dtype=np.uint8),
        '2x2 Checker': np.array([[255, 0], [0, 255]], dtype=np.uint8),
        '3x3 Cross': np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]], dtype=np.uint8),
    }
    
    # Test with different pad_h and pad_w values
    pad_combos = [(1, 1), (2, 1), (1, 2)]
    
    for pad_h, pad_w in pad_combos:
        print(f"\n--- Testing with pad_h={pad_h}, pad_w={pad_w} ---")
        for name, pattern in patterns.items():
            print(f"\n{name}:")
            print(f"  Original shape: {pattern.shape}")
            
            padded = pad_binary_image(pattern, pad_h, pad_w)
            print(f"  Padded shape: {padded.shape}")
    
    # Visual demo with pad_h=2, pad_w=2
    print("\n\nVisual Demo:")
    pad_h, pad_w = 2, 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, pattern) in enumerate(patterns.items()):
        padded = pad_binary_image(pattern, pad_h, pad_w)
        
        # Create combined visualization
        combined_h = max(pattern.shape[0] + 2, padded.shape[0])
        combined_w = pattern.shape[1] + padded.shape[1] + 10
        combined = np.zeros((combined_h, combined_w), dtype=np.uint8)
        
        # Place original at left (centered vertically)
        start_row_orig = (combined_h - pattern.shape[0]) // 2
        combined[start_row_orig:start_row_orig+pattern.shape[0], :pattern.shape[1]] = pattern
        
        # Place padded at right (centered vertically)
        start_col = pattern.shape[1] + 10
        start_row_pad = (combined_h - padded.shape[0]) // 2
        combined[start_row_pad:start_row_pad+padded.shape[0], start_col:start_col+padded.shape[1]] = padded
        
        axes[idx].imshow(combined, cmap='gray', vmin=0, vmax=255)
        axes[idx].set_title(f'{name}\n{pattern.shape} → {padded.shape}', fontsize=10)
        axes[idx].axis('off')
        
        # Add separator line
        axes[idx].axvline(x=pattern.shape[1] + 4.5, color='red', linewidth=2)
        axes[idx].text(pattern.shape[1]/2, combined_h - 0.5, 'Original', 
                      ha='center', fontsize=8)
        axes[idx].text(start_col + padded.shape[1]/2, combined_h - 0.5, f'Padded\n(pad_h={pad_h}, pad_w={pad_w})', 
                      ha='center', fontsize=8)
    
    plt.suptitle(f'Quick Demo: pad_binary_image(image, {pad_h}, {pad_w})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Verify the import works
    try:
        import inspect
        print(f"Successfully imported from:")
        print(f"  pad_binary_image: {inspect.getfile(pad_binary_image)}")
        if 'convert_to_grayscale' in dir():
            print(f"  convert_to_grayscale: {inspect.getfile(convert_to_grayscale)}")
        if 'ensure_binary' in dir():
            print(f"  ensure_binary: {inspect.getfile(ensure_binary)}")
        print(f"\nFunction signature: pad_binary_image(binary_image, pad_h, pad_w)")
    except ImportError as e:
        print(f"Error importing function: {e}")
        print("Make sure the processing module is in your Python path")
        sys.exit(1)
    
    # Show quick demo
    quick_demo()
    
    # Launch interactive mode
    demo_with_test_patterns()