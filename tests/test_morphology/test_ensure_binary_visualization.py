import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import ensure_binary, convert_to_grayscale


def visualize_binarization(original, binary, title="Binarization Result"):
    """Visualize original vs binarized images with pixel value comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    im1 = axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Original histogram
    axes[0, 1].hist(original.flatten(), bins=50, color='gray', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Original Pixel Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].axvline(x=255, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    # Binary image
    im2 = axes[1, 0].imshow(binary, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('After ensure_binary()', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Binary histogram
    axes[1, 1].hist(binary.flatten(), bins=2, color=['black', 'white'], edgecolor='black', alpha=0.7)
    axes[1, 1].set_title(f'Binary Pixel Distribution\n(Only 0 and 255)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xticks([0, 255])
    
    # Add statistics
    zero_count = np.sum(binary == 0)
    non_zero_count = np.sum(binary > 0)
    total = binary.size
    
    stats_text = f"Statistics:\n"
    stats_text += f"  Zeros (0): {zero_count} ({100*zero_count/total:.1f}%)\n"
    stats_text += f"  Non-zeros (255): {non_zero_count} ({100*non_zero_count/total:.1f}%)"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_with_custom_image():
    """Load and binarize a custom image"""
    
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
        # Load the image using PIL
        pil_image = Image.open(file_path)
        print(f"\nLoaded: {os.path.basename(file_path)}")
        print(f"Original mode: {pil_image.mode}")
        print(f"Original size: {pil_image.size}")
        
        # Convert PIL image to numpy array
        original_array = np.array(pil_image)
        print(f"Numpy array shape: {original_array.shape}")
        print(f"Numpy array dtype: {original_array.dtype}")
        
        # Convert to grayscale if color
        if len(original_array.shape) == 3:
            print("Converting color image to grayscale first...")
            grayscale_array = convert_to_grayscale(original_array)
        else:
            grayscale_array = original_array
        
        # Apply ensure_binary
        binary_array = ensure_binary(grayscale_array)
        
        # Show original pixel values statistics
        unique_before = np.unique(grayscale_array)
        unique_after = np.unique(binary_array)
        
        print(f"\nBefore ensure_binary:")
        print(f"  Unique values: {len(unique_before)}")
        print(f"  Min: {grayscale_array.min()}, Max: {grayscale_array.max()}")
        print(f"  Mean: {grayscale_array.mean():.1f}")
        
        print(f"\nAfter ensure_binary:")
        print(f"  Unique values: {unique_after}")
        print(f"  All zeros: {np.all(binary_array == 0)}")
        print(f"  All 255s: {np.all(binary_array == 255)}")
        
        # Show sample of pixel transformations
        print("\nSample pixel transformations (first 5 non-zero pixels):")
        count = 0
        for i in range(min(5, grayscale_array.shape[0])):
            for j in range(min(5, grayscale_array.shape[1])):
                before = grayscale_array[i, j]
                after = binary_array[i, j]
                print(f"  Position ({i},{j}): {before} → {after}")
        
        # Visualize
        visualize_binarization(grayscale_array, binary_array, 
                              f"ensure_binary() on {os.path.basename(file_path)}")
        
        # Save option
        save_option = input("\nDo you want to save the binarized image? (y/n): ").lower().strip()
        if save_option == 'y':
            save_path = filedialog.asksaveasfilename(
                title="Save binary image",
                defaultextension=".png",
                filetypes=[
                    ("PNG file", "*.png"),
                    ("JPEG file", "*.jpg"),
                    ("BMP file", "*.bmp"),
                ]
            )
            if save_path:
                Image.fromarray(binary_array).save(save_path)
                print(f"Image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_with_test_patterns():
    """Demonstrate ensure_binary with various test patterns"""
    
    while True:
        print("\n" + "="*60)
        print("ENSURE_BINARY VISUALIZER")
        print("Using: processing.morphology.binary_morphology.ensure_binary")
        print("="*60)
        print("1. Upload and binarize your own image")
        print("2. Test with gradient pattern")
        print("3. Test with random noise image")
        print("4. Test with specific value transformations")
        print("5. Side-by-side comparison of different images")
        print("6. Interactive pixel value explorer")
        print("7. Show function source code")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            test_with_custom_image()
        
        elif choice == '2':
            # Gradient test pattern
            print("\nCreating gradient test pattern (0 to 255)...")
            gradient = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
            
            result = ensure_binary(gradient)
            
            # Show with a line profile
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            
            # Original gradient
            axes[0, 0].imshow(gradient, cmap='gray', vmin=0, vmax=255)
            axes[0, 0].set_title('Original Gradient (0-255)', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Line profile of original
            line_y = 50
            axes[0, 1].plot(gradient[line_y, :], 'b-', linewidth=1)
            axes[0, 1].set_title(f'Original Line Profile (y={line_y})', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('X Position')
            axes[0, 1].set_ylabel('Pixel Value')
            axes[0, 1].set_ylim(-10, 265)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Binary result
            axes[1, 0].imshow(result, cmap='gray', vmin=0, vmax=255)
            axes[1, 0].set_title('After ensure_binary()', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            # Line profile of binary
            axes[1, 1].plot(result[line_y, :], 'r-', linewidth=2)
            axes[1, 1].set_title(f'Binary Line Profile (y={line_y})', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('X Position')
            axes[1, 1].set_ylabel('Pixel Value')
            axes[1, 1].set_ylim(-10, 265)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add transformation indicator
            axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.3, label='x=0 (stays 0)')
            axes[0, 1].legend()
            
            plt.suptitle('Gradient Binarization - All Non-Zero Values Become 255', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print some transformations
            print("\nSample transformations:")
            for x in [0, 1, 128, 254, 255]:
                print(f"  Value {x:3d} → {result[50, x]}")
        
        elif choice == '3':
            # Random noise
            print("\nCreating random noise image (values 0-255)...")
            np.random.seed(42)
            noise = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
            
            result = ensure_binary(noise)
            
            visualize_binarization(noise, result, "Random Noise Binarization")
            
            # Show transformation ratio
            zero_before = np.sum(noise == 0)
            zero_after = np.sum(result == 0)
            print(f"\nTransformation stats:")
            print(f"  Zeros before: {zero_before} ({100*zero_before/noise.size:.2f}%)")
            print(f"  Zeros after: {zero_after} ({100*zero_after/result.size:.2f}%)")
            print(f"  Only true zeros remain 0, everything else → 255")
        
        elif choice == '4':
            # Specific value transformations
            print("\n" + "-"*40)
            print("SPECIFIC VALUE TRANSFORMATIONS")
            print("-"*40)
            
            # Create test values
            test_values = [0, 1, 2, 10, 50, 100, 127, 128, 129, 200, 254, 255]
            
            # Create a 12x1 test image
            test_image = np.array(test_values, dtype=np.uint8).reshape(1, -1)
            result = ensure_binary(test_image)
            
            # Visualize
            fig, axes = plt.subplots(2, 1, figsize=(14, 6))
            
            # Original values as bar chart
            colors_before = ['black' if v == 0 else 'gray' for v in test_values]
            axes[0].bar(range(len(test_values)), test_values, color=colors_before)
            axes[0].set_title('Original Values', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Test Case')
            axes[0].set_ylabel('Pixel Value')
            axes[0].set_xticks(range(len(test_values)))
            axes[0].set_xticklabels(test_values)
            axes[0].set_ylim(-5, 270)
            axes[0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(test_values):
                axes[0].text(i, v + 8, str(v), ha='center', fontsize=9)
            
            # Binary result
            colors_after = ['black' if v == 0 else 'white' for v in result[0]]
            axes[1].bar(range(len(result[0])), result[0], color=colors_after, edgecolor='black')
            axes[1].set_title('After ensure_binary() - All Non-Zero → 255', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Test Case')
            axes[1].set_ylabel('Pixel Value')
            axes[1].set_xticks(range(len(result[0])))
            axes[1].set_xticklabels([f'{v}→{r}' for v, r in zip(test_values, result[0])])
            axes[1].set_ylim(-5, 270)
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(result[0]):
                axes[1].text(i, v + 8, str(v), ha='center', fontsize=9, fontweight='bold')
            
            plt.suptitle('ensure_binary() Value Transformations', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print summary
            print("\nTransformation Summary:")
            print("-" * 40)
            for original, binary in zip(test_values, result[0]):
                print(f"  {original:3d} → {binary:3d}  {'(Only 0 stays 0)' if original == 0 else '(Non-zero becomes 255)'}")
        
        elif choice == '5':
            # Side-by-side comparison
            print("\nCreating different test patterns for comparison...")
            
            # Create different patterns
            patterns = {
                'Checkerboard': np.array([[0, 255, 0, 255], [255, 0, 255, 0]] * 4, dtype=np.uint8),
                'Gradient': np.tile(np.arange(0, 256, 32, dtype=np.uint8), (8, 1)),
                'Random': np.random.randint(0, 256, (8, 8), dtype=np.uint8),
            }
            
            fig, axes = plt.subplots(len(patterns), 2, figsize=(10, 12))
            
            for idx, (name, pattern) in enumerate(patterns.items()):
                result = ensure_binary(pattern)
                
                # Original
                axes[idx, 0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
                axes[idx, 0].set_title(f'{name} - Original', fontsize=11, fontweight='bold')
                axes[idx, 0].axis('off')
                
                # Add pixel values
                for i in range(pattern.shape[0]):
                    for j in range(pattern.shape[1]):
                        color = 'white' if pattern[i, j] < 128 else 'black'
                        axes[idx, 0].text(j, i, str(pattern[i, j]), ha='center', va='center', 
                                         color=color, fontsize=8)
                
                # Binary
                axes[idx, 1].imshow(result, cmap='gray', vmin=0, vmax=255)
                axes[idx, 1].set_title(f'{name} - Binary', fontsize=11, fontweight='bold')
                axes[idx, 1].axis('off')
                
                # Add pixel values
                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        color = 'white' if result[i, j] < 128 else 'black'
                        axes[idx, 1].text(j, i, str(result[i, j]), ha='center', va='center', 
                                         color=color, fontsize=8)
            
            plt.suptitle('Side-by-Side Comparison - ensure_binary()', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print transformations
            print("\nDetailed Transformations:")
            for name, pattern in patterns.items():
                print(f"\n{name}:")
                print("  Original → Binary")
                unique_orig = np.unique(pattern)
                for val in unique_orig:
                    binary_val = 0 if val == 0 else 255
                    print(f"    {val:3d} → {binary_val}")
        
        elif choice == '6':
            # Interactive pixel value explorer
            print("\n" + "-"*40)
            print("INTERACTIVE PIXEL VALUE EXPLORER")
            print("-"*40)
            print("Enter pixel values to see how they transform")
            print("(Enter 'q' to quit)")
            
            while True:
                value_input = input("\nEnter pixel value (0-255): ").strip()
                
                if value_input.lower() == 'q':
                    break
                
                try:
                    value = int(value_input)
                    if 0 <= value <= 255:
                        test = np.array([[value]], dtype=np.uint8)
                        result = ensure_binary(test)
                        
                        print(f"  Input:  {value:3d}")
                        print(f"  Output: {result[0, 0]:3d}")
                        
                        if value == 0:
                            print("  ✓ Zero stays zero")
                        else:
                            print("  ✓ Non-zero becomes 255")
                        
                        # Visual
                        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                        
                        axes[0].imshow(test, cmap='gray', vmin=0, vmax=255)
                        axes[0].set_title(f'Original: {value}', fontsize=12)
                        axes[0].axis('off')
                        
                        axes[1].imshow(result, cmap='gray', vmin=0, vmax=255)
                        axes[1].set_title(f'Binary: {result[0, 0]}', fontsize=12)
                        axes[1].axis('off')
                        
                        plt.suptitle('ensure_binary() Transformation', fontsize=14, fontweight='bold')
                        plt.show()
                    else:
                        print("Value must be between 0 and 255")
                except ValueError:
                    print("Please enter a valid number")
        
        elif choice == '7':
            # Show source code
            import inspect
            print("\n" + "="*60)
            print("Function source code:")
            print("="*60)
            print(inspect.getsource(ensure_binary))
            print("="*60)
            print(f"Function location: {inspect.getfile(ensure_binary)}")
        
        elif choice == '8':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


def quick_demo():
    """Quick demonstration of ensure_binary"""
    
    print("\n" + "="*60)
    print("QUICK DEMO - ensure_binary() Transformations")
    print("="*60)
    
    # Create test image with known values
    test_image = np.array([
        [0, 1, 50, 100],
        [127, 128, 200, 254],
        [255, 10, 75, 150]
    ], dtype=np.uint8)
    
    result = ensure_binary(test_image)
    
    # Print transformation table
    print("\nInput  →  Output")
    print("-" * 25)
    for i in range(test_image.shape[0]):
        for j in range(test_image.shape[1]):
            before = test_image[i, j]
            after = result[i, j]
            print(f"  {before:3d}  →  {after:3d}")
    
    print("\nRule: Only 0 stays 0, everything else becomes 255")
    
    # Visual with pixel values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original with values
    axes[0].imshow(test_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    for i in range(test_image.shape[0]):
        for j in range(test_image.shape[1]):
            color = 'white' if test_image[i, j] < 128 else 'black'
            axes[0].text(j, i, str(test_image[i, j]), ha='center', va='center', 
                        color=color, fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Result with values
    axes[1].imshow(result, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('After ensure_binary()', fontsize=12, fontweight='bold')
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            color = 'white' if result[i, j] < 128 else 'black'
            axes[1].text(j, i, str(result[i, j]), ha='center', va='center', 
                        color=color, fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('ensure_binary() Quick Demo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Verify the import works
    try:
        import inspect
        print(f"Successfully imported from:")
        print(f"  ensure_binary: {inspect.getfile(ensure_binary)}")
        if 'convert_to_grayscale' in dir():
            print(f"  convert_to_grayscale: {inspect.getfile(convert_to_grayscale)}")
    except ImportError as e:
        print(f"Error importing function: {e}")
        print("Make sure the processing module is in your Python path")
        sys.exit(1)
    
    # Show quick demo
    quick_demo()
    
    # Launch interactive mode
    demo_with_test_patterns()