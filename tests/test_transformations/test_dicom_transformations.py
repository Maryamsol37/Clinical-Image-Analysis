import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from processing.geometry.transformations import rotate, shear, scale
import matplotlib.pyplot as plt
import pydicom

# Load a real DICOM image
dicom_folder = r"C:\Users\DELL\.cache\kagglehub\datasets\kmader\siim-medical-images\versions\6" 
dicom_path=None
for root, dirs, files in os.walk(dicom_folder):
    for file in files:
        if file.endswith(".dcm"):
            dicom_path = os.path.join(root, file)
            break
    if dicom_path:
        break
print("Loading:", dicom_path)
ds= pydicom.dcmread(dicom_path)
img= ds.pixel_array.astype(np.float32)



img= ((img- img.min())/(img.max()-img.min()) * 255).astype(np.uint8)

print("Image shape:", img.shape)
print("Min/Max pixel values:", img.min(), img.max()) 


rotated_img= rotate(img,angle_degrees= 30)
sheared_img= shear(img, shear_x=0.2, shear_y=0.0)
scaled_up_img= scale(img, scale_factor=1.5)
scaled_down_img= scale(img, scale_factor=0.5)

fig, axes= plt.subplots(1,5, figsize=(20,5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Image") 
axes[1].imshow(rotated_img, cmap='gray')
axes[1].set_title("Rotated Image (30°)")
axes[2].imshow(sheared_img, cmap='gray')
axes[2].set_title("Sheared Image (shear_x=0.2)")
axes[3].imshow(scaled_up_img, cmap='gray')
axes[3].set_title("Scaled Up Image (1.5x)")
axes[4].imshow(scaled_down_img, cmap='gray')
axes[4].set_title("Scaled Down Image (0.5x)")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig("test_dicom_transformations.png")
plt.show()
