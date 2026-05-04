import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from processing.geometry.transformations import rotate, shear, scale    
import matplotlib.pyplot as plt

img= np.zeros((100,100), dtype=np.uint8)
img[45:55, :]=255
img[:, 45:55]=255

rotated_img= rotate(img,angle_degrees= 45)
sheared_img= shear(img, shear_x=0.3, shear_y=0.0)
scaled_up_img= scale(img, scale_factor=1.5)
scaled_down_img= scale(img, scale_factor=0.5)

flig, axes= plt.subplots(1,5, figsize=(15,8))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Image")  
axes[1].imshow(rotated_img, cmap='gray')
axes[1].set_title("Rotated Image (45°)")
axes[2].imshow(sheared_img, cmap='gray')    
axes[2].set_title("Sheared Image (shear_x=0.3)")
axes[3].imshow(scaled_up_img, cmap='gray')  
axes[3].set_title("Scaled Up Image (1.5x)")
axes[4].imshow(scaled_down_img, cmap='gray')
axes[4].set_title("Scaled Down Image (0.5x)")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig("test_transformations.png") 
plt.show()