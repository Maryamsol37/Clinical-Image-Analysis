import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import global_threshold, convert_to_grayscale


def load_and_threshold_image():
    """Open file dialog, load image, apply thresholding, and display results"""
    
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
        
        # Convert to grayscale first if it's a color image
        if len(original_array.shape) == 3:
            print("Converting color image to grayscale first...")
            grayscale_array = convert_to_grayscale(original_array)
        else:
            grayscale_array = original_array
        
        # Get threshold value from user
        print(f"\nImage value range: {grayscale_array.min()} to {grayscale_array.max()}")
        print(f"Image mean value: {grayscale_array.mean():.1f}")
        
        while True:
            try:
                threshold_input = input(f"Enter threshold value (0-255, or 'auto' for mean={grayscale_array.mean():.0f}): ").strip()
                
                if threshold_input.lower() == 'auto':
                    threshold = int(grayscale_array.mean())
                    print(f"Using mean value as threshold: {threshold}")
                    break
                else:
                    threshold = int(threshold_input)
                    if 0 <= threshold <= 255:
                        break
                    else:
                        print("Threshold must be between 0 and 255")
            except ValueError:
                print("Please enter a valid number or 'auto'")
        
        # Apply thresholding
        binary_array = global_threshold(grayscale_array, threshold)
        print(f"Binary image created with threshold = {threshold}")
        print(f"Binary shape: {binary_array.shape}")
        
        # Count black and white pixels
        white_pixels = np.sum(binary_array == 255)
        black_pixels = np.sum(binary_array == 0)
        total_pixels = binary_array.size
        print(f"White pixels: {white_pixels} ({100*white_pixels/total_pixels:.1f}%)")
        print(f"Black pixels: {black_pixels} ({100*black_pixels/total_pixels:.1f}%)")
        
        # Display original, grayscale, and binary images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Show original image
        if len(original_array.shape) == 3 and original_array.shape[2] in [3, 4]:
            axes[0].imshow(original_array)
        else:
            axes[0].imshow(original_array, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Show grayscale image
        axes[1].imshow(grayscale_array, cmap='gray')
        axes[1].set_title(f'Grayscale Image\nRange: [{grayscale_array.min()}, {grayscale_array.max()}]', fontsize=12)
        axes[1].axis('off')
        
        # Show binary image
        axes[2].imshow(binary_array, cmap='gray')
        axes[2].set_title(f'Binary Image (Threshold={threshold})\nWhite: {100*white_pixels/total_pixels:.1f}% | Black: {100*black_pixels/total_pixels:.1f}%', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Interactive threshold adjustment
        adjust = input("\nDo you want to try different threshold values? (y/n): ").lower().strip()
        while adjust == 'y':
            try:
                new_threshold = int(input(f"Enter new threshold value (0-255): ").strip())
                if 0 <= new_threshold <= 255:
                    # Apply new threshold
                    binary_array = global_threshold(grayscale_array, new_threshold)
                    
                    # Update statistics
                    white_pixels = np.sum(binary_array == 255)
                    black_pixels = np.sum(binary_array == 0)
                    
                    # Update display
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    axes[0].imshow(grayscale_array, cmap='gray')
                    axes[0].set_title(f'Grayscale Image', fontsize=12)
                    axes[0].axis('off')
                    
                    axes[1].imshow(binary_array, cmap='gray')
                    axes[1].set_title(f'Binary Image (Threshold={new_threshold})\nWhite: {100*white_pixels/total_pixels:.1f}% | Black: {100*black_pixels/total_pixels:.1f}%', fontsize=12)
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"\nNew stats with threshold={new_threshold}:")
                    print(f"  White pixels: {white_pixels} ({100*white_pixels/total_pixels:.1f}%)")
                    print(f"  Black pixels: {black_pixels} ({100*black_pixels/total_pixels:.1f}%)")
                    
                    adjust = input("\nTry another threshold? (y/n): ").lower().strip()
                else:
                    print("Threshold must be between 0 and 255")
            except ValueError:
                print("Please enter a valid number")
        
        # Ask if user wants to save the binary image
        save_option = input("\nDo you want to save the binary image? (y/n): ").lower().strip()
        if save_option == 'y':
            save_path = filedialog.asksaveasfilename(
                title="Save binary image",
                defaultextension=".png",
                filetypes=[
                    ("PNG file", "*.png"),
                    ("JPEG file", "*.jpg"),
                    ("BMP file", "*.bmp"),
                    ("TIFF file", "*.tiff")
                ]
            )
            if save_path:
                Image.fromarray(binary_array).save(save_path)
                print(f"Image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_with_sample_patterns():
    """Demonstrate thresholding with built-in test patterns"""
    
    while True:
        print("\n" + "="*60)
        print("THRESHOLDING DEMO")
        print("Using: binary_morphology.global_threshold")
        print("="*60)
        print("1. Upload and threshold your own image")
        print("2. Test with gradient pattern")
        print("3. Test with checkerboard pattern")
        print("4. Test with varying threshold values")
        print("5. Show function source code")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            load_and_threshold_image()
        
        elif choice == '2':
            # Create gradient test pattern
            gradient = np.zeros((300, 400), dtype=np.uint8)
            for i in range(400):
                val = int(i * 255 / 399)
                gradient[:, i] = val
            
            print("\nGradient image: values from 0 (left) to 255 (right)")
            
            # Get threshold from user
            while True:
                try:
                    threshold = int(input("Enter threshold value (0-255, default=128): ").strip() or "128")
                    if 0 <= threshold <= 255:
                        break
                    else:
                        print("Threshold must be between 0 and 255")
                except ValueError:
                    print("Please enter a valid number")
            
            result = global_threshold(gradient, threshold)
            
            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(gradient, cmap='gray')
            axes[0].set_title('Original Gradient\n(0 to 255)', fontsize=12)
            axes[0].axvline(x=threshold * 400 / 255, color='r', linestyle='--', linewidth=2)
            axes[0].text(threshold * 400 / 255, 150, f'  Threshold={threshold}', 
                        color='red', fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            # Show histogram
            axes[1].hist(gradient.flatten(), bins=50, color='gray', edgecolor='black')
            axes[1].axvline(x=threshold, color='r', linestyle='--', linewidth=2)
            axes[1].set_title(f'Histogram\nThreshold={threshold}', fontsize=12)
            axes[1].set_xlabel('Pixel Value')
            axes[1].set_ylabel('Frequency')
            
            axes[2].imshow(result, cmap='gray')
            axes[2].set_title(f'Binary Result\n(Threshold={threshold})', fontsize=12)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        elif choice == '3':
            # Create checkerboard pattern with different intensities
            size = 100
            grid_size = 5
            pattern = np.zeros((grid_size * size, grid_size * size), dtype=np.uint8)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    intensity = int((i * grid_size + j) * 255 / (grid_size * grid_size - 1))
                    pattern[i*size:(i+1)*size, j*size:(j+1)*size] = intensity
            
            print("\nCheckerboard with intensities from 0 to 255")
            
            # Get threshold
            while True:
                try:
                    threshold = int(input("Enter threshold value (0-255, default=128): ").strip() or "128")
                    if 0 <= threshold <= 255:
                        break
                    else:
                        print("Threshold must be between 0 and 255")
                except ValueError:
                    print("Please enter a valid number")
            
            result = global_threshold(pattern, threshold)
            
            # Create a mask showing which squares are white/black
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original with labels
            im1 = axes[0].imshow(pattern, cmap='gray')
            axes[0].set_title('Original Checkerboard\n(Values 0-255)', fontsize=12)
            for i in range(grid_size):
                for j in range(grid_size):
                    intensity = int((i * grid_size + j) * 255 / (grid_size * grid_size - 1))
                    color = 'white' if intensity < 128 else 'black'
                    axes[0].text(j*size + size/2, i*size + size/2, str(intensity), 
                               ha='center', va='center', color=color, fontsize=10)
            axes[0].axis('off')
            
            # Plot intensity values
            square_values = [int((i * grid_size + j) * 255 / (grid_size * grid_size - 1)) 
                           for i in range(grid_size) for j in range(grid_size)]
            colors = ['white' if val >= threshold else 'black' for val in square_values]
            axes[1].bar(range(len(square_values)), square_values, color='gray', edgecolor='black')
            axes[1].axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
            axes[1].set_title('Square Intensities', fontsize=12)
            axes[1].set_xlabel('Square Number')
            axes[1].set_ylabel('Intensity')
            axes[1].legend()
            
            # Binary result with labels
            axes[2].imshow(result, cmap='gray')
            axes[2].set_title(f'Binary Result\n(≥{threshold}=White)', fontsize=12)
            for i in range(grid_size):
                for j in range(grid_size):
                    intensity = int((i * grid_size + j) * 255 / (grid_size * grid_size - 1))
                    binary_val = '255' if intensity >= threshold else '0'
                    color = 'black' if intensity >= threshold else 'white'
                    axes[2].text(j*size + size/2, i*size + size/2, binary_val, 
                               ha='center', va='center', color=color, fontsize=10)
            axes[2].axis('off')
            
            plt.suptitle(f'Threshold Demo (Threshold = {threshold})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        elif choice == '4':
            # Show effect of different thresholds on a single image
            print("\nCreating a complex test pattern...")
            
            # Create a test pattern with circles of different intensities
            pattern = np.zeros((200, 600), dtype=np.uint8)
            
            # Add 5 circles with different intensities
            for idx, intensity in enumerate([51, 102, 128, 153, 204]):
                center_x = 100 + idx * 100
                center_y = 100
                
                for i in range(pattern.shape[0]):
                    for j in range(pattern.shape[1]):
                        distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                        if distance < 80:
                            pattern[i, j] = max(pattern[i, j], intensity)
            
            # Display with different thresholds
            thresholds = [50, 80, 110, 140, 170, 200]
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Original image
            axes[0, 0].imshow(pattern, cmap='gray')
            axes[0, 0].set_title('Original Pattern', fontsize=10)
            axes[0, 0].axis('off')
            
            # Histogram
            axes[0, 1].hist(pattern.flatten(), bins=50, color='gray', edgecolor='black')
            axes[0, 1].set_title('Histogram', fontsize=10)
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Count')
            
            # Apply different thresholds
            for idx, threshold in enumerate(thresholds):
                if idx < 6:
                    result = global_threshold(pattern, threshold)
                    row = (idx + 2) // 4
                    col = (idx + 2) % 4
                    
                    axes[row, col].imshow(result, cmap='gray')
                    white_pct = 100 * np.sum(result == 255) / result.size
                    axes[row, col].set_title(f'T={threshold} | White: {white_pct:.0f}%', fontsize=10)
                    axes[row, col].axis('off')
            
            plt.suptitle('Effect of Different Threshold Values', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        elif choice == '5':
            # Show the source code of the imported function
            import inspect
            print("\n" + "="*60)
            print("Function source code:")
            print("="*60)
            print(inspect.getsource(global_threshold))
            print("="*60)
            print(f"Function location: {inspect.getfile(global_threshold)}")
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


if __name__ == '__main__':
    # Verify the import works
    try:
        import inspect
        print(f"Successfully imported from:")
        print(f"  global_threshold: {inspect.getfile(global_threshold)}")
        if 'convert_to_grayscale' in dir():
            print(f"  convert_to_grayscale: {inspect.getfile(convert_to_grayscale)}")
    except ImportError as e:
        print(f"Error importing function: {e}")
        print("Make sure the processing module is in your Python path")
        sys.exit(1)
    
    test_with_sample_patterns()