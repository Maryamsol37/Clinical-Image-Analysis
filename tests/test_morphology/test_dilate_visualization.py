import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys
import os

# Add the project root to the Python path to import from processing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the functions from your existing module
from processing.morphology.binary_morphology import dilate, create_structuring_element, convert_to_grayscale, ensure_binary


def visualize_dilation(original, dilated, structuring_element, title="Dilation Result"):
    """Visualize original vs dilated images with SE display"""
    
    fig = plt.figure(figsize=(16, 5))
    
    # Original image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    
    white_before = np.sum(original == 255)
    ax1.set_title(f'Original Image\nShape: {original.shape}\nWhite px: {white_before}', 
                 fontsize=11, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add grid and pixel values for small images
    if original.size <= 100:
        for i in range(original.shape[0] + 1):
            ax1.axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
            ax1.axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                if original[i, j] == 255:
                    ax1.text(j, i, '255', ha='center', va='center', 
                            color='black', fontsize=7, fontweight='bold')
    
    # Structuring Element
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(structuring_element, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'Structuring Element\nShape: {structuring_element.shape}\nActive: {np.sum(structuring_element)} px', 
                 fontsize=11, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add grid for SE
    for i in range(structuring_element.shape[0] + 1):
        ax2.axhline(y=i - 0.5, color='green', linewidth=1, alpha=0.5)
        ax2.axvline(x=i - 0.5, color='green', linewidth=1, alpha=0.5)
    for i in range(structuring_element.shape[0]):
        for j in range(structuring_element.shape[1]):
            color = 'white' if structuring_element[i, j] < 0.5 else 'black'
            ax2.text(j, i, str(structuring_element[i, j]), ha='center', va='center', 
                    color=color, fontsize=9, fontweight='bold')
    
    # Dilated image
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(dilated, cmap='gray', vmin=0, vmax=255)
    
    white_after = np.sum(dilated == 255)
    white_added = white_after - white_before
    
    ax3.set_title(f'Dilated Image\nShape: {dilated.shape}\nWhite px: {white_before} → {white_after} (+{white_added})', 
                 fontsize=11, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Highlight newly added white pixels
    if original.size <= 400:
        new_white = (dilated == 255) & (original == 0)
        for i in range(dilated.shape[0]):
            for j in range(dilated.shape[1]):
                if new_white[i, j]:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                   linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
                    ax3.add_patch(rect)
    
    # Add grid for small images
    if dilated.size <= 100:
        for i in range(dilated.shape[0] + 1):
            ax3.axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
            ax3.axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
        for i in range(dilated.shape[0]):
            for j in range(dilated.shape[1]):
                if dilated[i, j] == 255:
                    is_new = original[i, j] == 0
                    color = 'red' if is_new else 'black'
                    ax3.text(j, i, '255', ha='center', va='center', 
                            color=color, fontsize=7, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_with_custom_image():
    """Load and dilate a custom image"""
    
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
        
        # Convert to numpy array and ensure binary
        original_array = np.array(pil_image)
        if len(original_array.shape) == 3:
            print("Converting to grayscale and binarizing...")
            grayscale = convert_to_grayscale(original_array)
            binary = ensure_binary(grayscale)
        else:
            binary = ensure_binary(original_array)
        
        print(f"Binary image shape: {binary.shape}")
        
        # Choose structuring element
        print("\nChoose structuring element:")
        print("1. 3x3 Square")
        print("2. 3x3 Cross")
        print("3. 5x5 Square")
        print("4. 5x5 Cross")
        print("5. Custom size")
        
        se_choice = input("Select (1-5): ").strip()
        
        if se_choice == '1':
            se = create_structuring_element(3, "square")
        elif se_choice == '2':
            se = create_structuring_element(3, "cross")
        elif se_choice == '3':
            se = create_structuring_element(5, "square")
        elif se_choice == '4':
            se = create_structuring_element(5, "cross")
        elif se_choice == '5':
            while True:
                try:
                    size = int(input("Enter SE size (odd number ≥ 3): ").strip())
                    if size >= 3 and size % 2 == 1:
                        break
                    else:
                        print("Size must be an odd number ≥ 3")
                except ValueError:
                    print("Please enter a valid number")
            
            shape = input("Shape (square/cross): ").strip().lower()
            if shape not in ['square', 'cross']:
                print("Invalid shape, using 'square'")
                shape = 'square'
            se = create_structuring_element(size, shape)
        else:
            print("Invalid choice, using 3x3 square")
            se = create_structuring_element(3, "square")
        
        # Apply dilation
        dilated = dilate(binary, se)
        
        print(f"\nDilation Results:")
        print(f"  SE shape: {se.shape}")
        print(f"  White pixels before: {np.sum(binary == 255)}")
        print(f"  White pixels after: {np.sum(dilated == 255)}")
        print(f"  Pixels added: {np.sum(dilated == 255) - np.sum(binary == 255)}")
        
        # Visualize
        visualize_dilation(binary, dilated, se, f"Dilation of {os.path.basename(file_path)}")
        
        # Save option
        save_option = input("\nDo you want to save the dilated image? (y/n): ").lower().strip()
        if save_option == 'y':
            save_path = filedialog.asksaveasfilename(
                title="Save dilated image",
                defaultextension=".png",
                filetypes=[("PNG file", "*.png"), ("JPEG file", "*.jpg"), ("BMP file", "*.bmp")]
            )
            if save_path:
                Image.fromarray(dilated).save(save_path)
                print(f"Image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_with_test_patterns():
    """Demonstrate dilation with various test patterns"""
    
    while True:
        print("\n" + "="*60)
        print("DILATION VISUALIZER")
        print("Using: processing.morphology.binary_morphology.dilate")
        print("="*60)
        print("1. Upload and dilate your own image")
        print("2. Test single pixel dilation (see SE shape emerge)")
        print("3. Test expanding boundaries")
        print("4. Test connecting objects")
        print("5. Test filling gaps")
        print("6. Compare different SE sizes")
        print("7. Compare square vs cross SE")
        print("8. Interactive dilation explorer")
        print("9. Erosion vs Dilation comparison")
        print("10. Show function source code")
        print("11. Exit")
        
        choice = input("\nSelect option (1-11): ").strip()
        
        if choice == '1':
            test_with_custom_image()
        
        elif choice == '2':
            # Single pixel dilation
            print("\n" + "-"*40)
            print("SINGLE PIXEL DILATION")
            print("Dilation of a single pixel reveals the SE shape!")
            print("-"*40)
            
            # Create single pixel
            size = 9
            single = np.zeros((size, size), dtype=np.uint8)
            single[size//2, size//2] = 255
            
            print(f"Original: Single white pixel in {size}x{size} image")
            print(single)
            
            # Try different SEs
            se_configs = [
                ("3x3 Square", create_structuring_element(3, "square")),
                ("3x3 Cross", create_structuring_element(3, "cross")),
                ("5x5 Square", create_structuring_element(5, "square")),
                ("5x5 Cross", create_structuring_element(5, "cross")),
            ]
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Original
            axes[0, 0].imshow(single, cmap='gray', vmin=0, vmax=255)
            axes[0, 0].set_title('Original\nSingle Pixel', fontsize=10)
            axes[0, 0].axis('off')
            axes[0, 0].plot(size//2, size//2, 'r.', markersize=20)
            
            for idx, (name, se) in enumerate(se_configs):
                row = idx // 2
                col = (idx % 2) + 1
                
                dilated = dilate(single, se)
                
                axes[row, col].imshow(dilated, cmap='gray', vmin=0, vmax=255)
                axes[row, col].set_title(f'{name}\nWhite: {np.sum(dilated==255)} px', fontsize=10)
                axes[row, col].axis('off')
                
                # Show SE in overlay
                axes[row + 1, col].imshow(se, cmap='gray', vmin=0, vmax=1)
                axes[row + 1, col].set_title(f'{name} SE\nActive: {np.sum(se)} px', fontsize=10)
                axes[row + 1, col].axis('off')
                for i in range(se.shape[0] + 1):
                    axes[row + 1, col].axhline(y=i-0.5, color='green', linewidth=1, alpha=0.5)
                    axes[row + 1, col].axvline(x=i-0.5, color='green', linewidth=1, alpha=0.5)
            
            plt.suptitle('Single Pixel Dilation → SE Shape Emerges!', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nKey Insight: Dilation of a single pixel produces the SE shape!")
            print("  Square SE → produces a filled square")
            print("  Cross SE → produces a cross pattern")
        
        elif choice == '3':
            # Expanding boundaries
            print("\n" + "-"*40)
            print("EXPANDING BOUNDARIES")
            print("-"*40)
            
            # Create white square
            img_size = 9
            square_size = 3
            image = np.zeros((img_size, img_size), dtype=np.uint8)
            start = (img_size - square_size) // 2
            image[start:start+square_size, start:start+square_size] = 255
            
            print(f"Original {square_size}x{square_size} white square in {img_size}x{img_size} image")
            print(image)
            
            # Try different dilation amounts
            se_sizes = [3, 5, 7]
            
            fig, axes = plt.subplots(1, len(se_sizes) + 1, figsize=(16, 4))
            
            # Original
            axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\n{square_size}x{square_size} Square\nWhite: {np.sum(image==255)} px', fontsize=10)
            axes[0].axis('off')
            
            for idx, se_size in enumerate(se_sizes):
                se = create_structuring_element(se_size, "square")
                dilated = dilate(image, se)
                
                white_after = np.sum(dilated == 255)
                if white_after > 0:
                    new_size = int(np.sqrt(white_after))
                    axes[idx + 1].set_title(f'{se_size}x{se_size} SE\nExpands to ~{new_size}x{new_size}\nWhite: {white_after} px', fontsize=10)
                else:
                    axes[idx + 1].set_title(f'{se_size}x{se_size} SE\nWhite: {white_after} px', fontsize=10)
                
                axes[idx + 1].imshow(dilated, cmap='gray', vmin=0, vmax=255)
                axes[idx + 1].axis('off')
            
            plt.suptitle('Boundary Expansion with Different SE Sizes', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nBoundary Expansion:")
            print(f"  Original: {square_size}x{square_size} = {square_size**2} white pixels")
            for se_size in se_sizes:
                se = create_structuring_element(se_size, "square")
                dilated = dilate(image, se)
                white = np.sum(dilated == 255)
                new_size = int(np.sqrt(white)) if white > 0 else 0
                expansion = (se_size - 1) // 2
                print(f"  {se_size}x{se_size} SE: {white} white pixels (~{new_size}x{new_size})")
                print(f"    Expands by {expansion} pixels on each side")
        
        elif choice == '4':
            # Connecting objects
            print("\n" + "-"*40)
            print("CONNECTING OBJECTS")
            print("-"*40)
            
            # Create two separate squares
            image = np.zeros((10, 15), dtype=np.uint8)
            image[3, 2] = 255  # Left pixel
            image[3, 12] = 255  # Right pixel
            
            # Also add some small groups
            image[5:7, 2] = 255  # Left vertical
            image[5:7, 12] = 255  # Right vertical
            
            print("Original (two separate groups, 10 pixels apart):")
            print(image)
            
            # Try different SE sizes to connect them
            se_sizes = [3, 5, 7, 9]
            
            fig, axes = plt.subplots(1, len(se_sizes) + 1, figsize=(18, 4))
            
            # Original
            axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\nTwo Groups\nWhite: {np.sum(image==255)} px', fontsize=10)
            axes[0].axis('off')
            
            for idx, se_size in enumerate(se_sizes):
                se = create_structuring_element(se_size, "square")
                dilated = dilate(image, se)
                
                # Check if connected
                # Simple check: is there a continuous path of white between left and right groups?
                middle_col = dilated.shape[1] // 2
                connected = np.any(dilated[:, middle_col] == 255)
                
                axes[idx + 1].imshow(dilated, cmap='gray', vmin=0, vmax=255)
                status = "CONNECTED!" if connected else "Still Separate"
                axes[idx + 1].set_title(f'{se_size}x{se_size} SE\n{status}\nWhite: {np.sum(dilated==255)} px', fontsize=10)
                axes[idx + 1].axis('off')
                
                if connected:
                    # Highlight connection
                    axes[idx + 1].axvline(x=middle_col - 0.5, color='red', linewidth=2, linestyle='--')
            
            plt.suptitle('Connecting Objects with Increasing Dilation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nConnection Analysis:")
            for se_size in se_sizes:
                se = create_structuring_element(se_size, "square")
                dilated = dilate(image, se)
                middle_col = dilated.shape[1] // 2
                connected = np.any(dilated[:, middle_col] == 255)
                expansion = (se_size - 1) // 2
                print(f"  {se_size}x{se_size} SE (expansion={expansion}): ", end="")
                print(f"{'CONNECTED' if connected else 'Separate'} | White: {np.sum(dilated==255)} px")
        
        elif choice == '5':
            # Filling gaps
            print("\n" + "-"*40)
            print("FILLING GAPS")
            print("-"*40)
            
            # Create image with gaps
            image = np.zeros((7, 10), dtype=np.uint8)
            image[2:5, :3] = 255  # Left block (3 columns)
            image[2:5, 7:] = 255  # Right block (3 columns)
            # Gap at columns 3-6 (4 columns)
            
            # Add small holes
            image[3, 1] = 0  # Small hole in left block
            
            print("Original (two blocks with gap and hole):")
            print(image)
            
            # Apply dilation with different SEs
            se_3x3 = create_structuring_element(3, "square")
            se_5x5 = create_structuring_element(5, "square")
            
            dilated_3x3 = dilate(image, se_3x3)
            dilated_5x5 = dilate(image, se_5x5)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Original
            axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\nGap + Hole\nWhite: {np.sum(image==255)} px', fontsize=10)
            axes[0].axis('off')
            
            # 3x3 dilation
            axes[1].imshow(dilated_3x3, cmap='gray', vmin=0, vmax=255)
            gap_filled_3 = np.all(dilated_3x3[2:5, 3:7] == 255)
            hole_filled_3 = dilated_3x3[3, 1] == 255
            axes[1].set_title(f'3x3 SE\nHole filled: {hole_filled_3}\nGap filled: {gap_filled_3}\nWhite: {np.sum(dilated_3x3==255)} px', fontsize=10)
            axes[1].axis('off')
            
            # 5x5 dilation
            axes[2].imshow(dilated_5x5, cmap='gray', vmin=0, vmax=255)
            gap_filled_5 = np.all(dilated_5x5[2:5, 3:7] == 255)
            hole_filled_5 = dilated_5x5[3, 1] == 255
            axes[2].set_title(f'5x5 SE\nHole filled: {hole_filled_5}\nGap filled: {gap_filled_5}\nWhite: {np.sum(dilated_5x5==255)} px', fontsize=10)
            axes[2].axis('off')
            
            plt.suptitle('Filling Gaps and Holes with Dilation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nGap/Hole Filling Results:")
            print(f"  3x3 SE: Hole filled={hole_filled_3}, Gap filled={gap_filled_3}")
            print(f"  5x5 SE: Hole filled={hole_filled_5}, Gap filled={gap_filled_5}")
            print(f"\nNote: Dilation fills small holes and narrow gaps!")
        
        elif choice == '6':
            # Compare different SE sizes
            print("\n" + "-"*40)
            print("COMPARING SE SIZES")
            print("-"*40)
            
            # Create test pattern with various features
            pattern = np.zeros((7, 7), dtype=np.uint8)
            pattern[3, 3] = 255  # Single pixel
            
            se_sizes = [3, 5, 7]
            
            fig, axes = plt.subplots(1, len(se_sizes) + 1, figsize=(16, 4))
            
            # Original
            axes[0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\nSingle Pixel\nWhite: {np.sum(pattern==255)} px', fontsize=10)
            axes[0].axis('off')
            axes[0].plot(3, 3, 'r.', markersize=20)
            
            for idx, se_size in enumerate(se_sizes):
                se = create_structuring_element(se_size, "square")
                dilated = dilate(pattern, se)
                
                white = np.sum(dilated == 255)
                new_size = int(np.sqrt(white)) if white > 0 else 0
                
                axes[idx + 1].imshow(dilated, cmap='gray', vmin=0, vmax=255)
                axes[idx + 1].set_title(f'{se_size}x{se_size} SE\nWhite: {white} px\n~{new_size}x{new_size} square', fontsize=10)
                axes[idx + 1].axis('off')
            
            plt.suptitle('Effect of SE Size on Dilation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nSize Comparison:")
            for se_size in se_sizes:
                se = create_structuring_element(se_size, "square")
                dilated = dilate(pattern, se)
                white = np.sum(dilated == 255)
                print(f"  {se_size}x{se_size} SE: 1 pixel → {white} pixels ({int(np.sqrt(white))}x{int(np.sqrt(white))} square)")
        
        elif choice == '7':
            # Compare square vs cross
            print("\n" + "-"*40)
            print("SQUARE VS CROSS SE")
            print("-"*40)
            
            # Test patterns
            patterns = {
                'Single Pixel': np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,255,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.uint8),
                'Two Pixels': np.array([[0,0,0,0,0],[0,0,0,0,0],[0,255,0,255,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.uint8),
                'Corner Pixel': np.array([[255,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.uint8),
            }
            
            se_square = create_structuring_element(3, "square")
            se_cross = create_structuring_element(3, "cross")
            
            fig, axes = plt.subplots(len(patterns), 3, figsize=(12, 12))
            
            for idx, (name, pattern) in enumerate(patterns.items()):
                dilated_sq = dilate(pattern, se_square)
                dilated_cr = dilate(pattern, se_cross)
                
                # Original
                axes[idx, 0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
                axes[idx, 0].set_title(f'{name}\nOriginal\nWhite: {np.sum(pattern==255)}', fontsize=9)
                axes[idx, 0].axis('off')
                
                # Square SE result
                axes[idx, 1].imshow(dilated_sq, cmap='gray', vmin=0, vmax=255)
                axes[idx, 1].set_title(f'Square SE\nWhite: {np.sum(dilated_sq==255)}', fontsize=9)
                axes[idx, 1].axis('off')
                
                # Cross SE result
                axes[idx, 2].imshow(dilated_cr, cmap='gray', vmin=0, vmax=255)
                axes[idx, 2].set_title(f'Cross SE\nWhite: {np.sum(dilated_cr==255)}', fontsize=9)
                axes[idx, 2].axis('off')
            
            plt.suptitle('Square vs Cross SE: Dilation Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nSquare SE (9 active pixels) vs Cross SE (5 active pixels):")
            print("  Square SE: More aggressive expansion")
            print("  Cross SE: Preserves more structure")
            for name, pattern in patterns.items():
                dilated_sq = dilate(pattern, se_square)
                dilated_cr = dilate(pattern, se_cross)
                print(f"\n  {name}:")
                print(f"    Square: {np.sum(pattern==255)} → {np.sum(dilated_sq==255)} (+{np.sum(dilated_sq==255)-np.sum(pattern==255)})")
                print(f"    Cross:  {np.sum(pattern==255)} → {np.sum(dilated_cr==255)} (+{np.sum(dilated_cr==255)-np.sum(pattern==255)})")
        
        elif choice == '8':
            # Interactive dilation explorer
            print("\n" + "-"*40)
            print("INTERACTIVE DILATION EXPLORER")
            print("-"*40)
            
            # Create a test pattern
            pattern = np.zeros((9, 9), dtype=np.uint8)
            pattern[4, 4] = 255  # Center pixel
            pattern[2, 2] = 255  # Top-left pixel
            pattern[6, 6] = 255  # Bottom-right pixel
            
            print("Original pattern:")
            print(pattern)
            
            while True:
                print("\nCurrent pattern shown above")
                print("Enter SE parameters (-1 to quit):")
                try:
                    se_size = int(input("  SE size (odd, ≥ 3): ").strip())
                    if se_size == -1:
                        break
                    se_shape = input("  SE shape (square/cross): ").strip().lower()
                    
                    if se_size >= 3 and se_size % 2 == 1 and se_shape in ['square', 'cross']:
                        se = create_structuring_element(se_size, se_shape)
                        dilated = dilate(pattern, se)
                        
                        white_added = np.sum(dilated == 255) - np.sum(pattern == 255)
                        
                        print(f"\n{se_size}x{se_size} {se_shape} SE:")
                        print(f"  White pixels before: {np.sum(pattern==255)}")
                        print(f"  White pixels after: {np.sum(dilated==255)}")
                        print(f"  Added: {white_added}")
                        
                        visualize_dilation(pattern, dilated, se, 
                                         f"Interactive: {se_size}x{se_size} {se_shape} SE")
                    else:
                        print("Invalid parameters. Size must be odd ≥ 3, shape: square/cross")
                except ValueError:
                    print("Please enter valid numbers")
        
        elif choice == '9':
            # Erosion vs Dilation comparison
            print("\n" + "-"*40)
            print("EROSION VS DILATION COMPARISON")
            print("-"*40)
            
            # Import erode for comparison
            from processing.morphology.binary_morphology import erode
            
            # Create test patterns
            patterns = {
                'Single Pixel': np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,255,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.uint8),
                '3x3 Square': np.array([[0,0,0,0,0],[0,255,255,255,0],[0,255,255,255,0],[0,255,255,255,0],[0,0,0,0,0]], dtype=np.uint8),
                'Thin Line': np.zeros((5, 5), dtype=np.uint8),
            }
            patterns['Thin Line'][2, :] = 255
            
            se = create_structuring_element(3, "square")
            
            fig, axes = plt.subplots(len(patterns), 3, figsize=(12, 12))
            
            for idx, (name, pattern) in enumerate(patterns.items()):
                eroded = erode(pattern, se)
                dilated = dilate(pattern, se)
                
                # Original
                axes[idx, 0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
                axes[idx, 0].set_title(f'{name}\nOriginal\nWhite: {np.sum(pattern==255)}', fontsize=10)
                axes[idx, 0].axis('off')
                
                # Eroded
                axes[idx, 1].imshow(eroded, cmap='gray', vmin=0, vmax=255)
                white_e = np.sum(eroded == 255)
                axes[idx, 1].set_title(f'Erosion\nWhite: {white_e}\n(-{np.sum(pattern==255)-white_e})', fontsize=10)
                axes[idx, 1].axis('off')
                
                # Dilated
                axes[idx, 2].imshow(dilated, cmap='gray', vmin=0, vmax=255)
                white_d = np.sum(dilated == 255)
                axes[idx, 2].set_title(f'Dilation\nWhite: {white_d}\n(+{white_d-np.sum(pattern==255)})', fontsize=10)
                axes[idx, 2].axis('off')
            
            plt.suptitle('Erosion (Shrinks) vs Dilation (Expands)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nErosion vs Dilation Summary:")
            print("  Erosion: Removes pixels from boundaries → Shrinks objects")
            print("  Dilation: Adds pixels to boundaries → Expands objects")
            print("  Erosion removes noise but can disconnect objects")
            print("  Dilation fills gaps but can merge objects")
            print("  They are dual operations: eroding foreground = dilating background")
        
        elif choice == '10':
            # Show source code
            import inspect
            print("\n" + "="*60)
            print("Function source code:")
            print("="*60)
            print(inspect.getsource(dilate))
            print("="*60)
            print(f"Function location: {inspect.getfile(dilate)}")
        
        elif choice == '11':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


def quick_demo():
    """Quick demonstration of dilation"""
    
    print("\n" + "="*60)
    print("QUICK DEMO - dilate()")
    print("="*60)
    
    # Single pixel
    single = np.zeros((5, 5), dtype=np.uint8)
    single[2, 2] = 255
    
    se_square = create_structuring_element(3, "square")
    se_cross = create_structuring_element(3, "cross")
    
    dilated_sq = dilate(single, se_square)
    dilated_cr = dilate(single, se_cross)
    
    print("\nOriginal single pixel:")
    print(single)
    
    print("\n3x3 Square SE:")
    print(se_square)
    print("\nAfter dilation (Square SE):")
    print(dilated_sq)
    print(f"White pixels: 1 → {np.sum(dilated_sq==255)}")
    
    print("\n3x3 Cross SE:")
    print(se_cross)
    print("\nAfter dilation (Cross SE):")
    print(dilated_cr)
    print(f"White pixels: 1 → {np.sum(dilated_cr==255)}")
    
    print("\nKey Insight: Dilation 'grows' the shape of the SE from each white pixel!")
    
    # Visual
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(single, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original\nSingle Pixel', fontsize=11)
    axes[0].axis('off')
    axes[0].plot(2, 2, 'r.', markersize=20)
    
    axes[1].imshow(dilated_sq, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Dilated (Square SE)\n→ 3x3 Square', fontsize=11)
    axes[1].axis('off')
    
    axes[2].imshow(dilated_cr, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Dilated (Cross SE)\n→ Cross Shape', fontsize=11)
    axes[2].axis('off')
    
    plt.suptitle('Dilation of Single Pixel Shows SE Shape', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Verify the import works
    try:
        import inspect
        print(f"Successfully imported from:")
        print(f"  dilate: {inspect.getfile(dilate)}")
        print(f"  create_structuring_element: {inspect.getfile(create_structuring_element)}")
        print(f"  convert_to_grayscale: {inspect.getfile(convert_to_grayscale)}")
        print(f"  ensure_binary: {inspect.getfile(ensure_binary)}")
    except ImportError as e:
        print(f"Error importing function: {e}")
        print("Make sure the processing module is in your Python path")
        sys.exit(1)
    
    # Show quick demo
    quick_demo()
    
    # Launch interactive mode
    demo_with_test_patterns()