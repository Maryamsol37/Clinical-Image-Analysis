import numpy as np 
from processing.interpolation.zoom import zoom_image

def rotate(image, angle_degrees):
    angle_radian=np.radians(angle_degrees)
    h,w=image.shape[:2]
    cx, cy= w/2, h/2
    output= np.zeros_like(image)

    for y_out in range(h):
        for x_out in range(w):
            dx, dy= x_out - cx, y_out - cy
            x_src= dx * np.cos(angle_radian) + dy * np.sin(angle_radian) + cx
            y_src= -dx * np.sin(angle_radian) + dy * np.cos(angle_radian) + cy
            output[y_out, x_out]= zoom_image(image, x_src, y_src)
    return output


def shear(image,shear_x=0.0, shear_y=0.0):
    h,w=image.shape[:2]
    output= np.zeros_like(image)

    for y_out in range(h):
        for x_out in range(w):
            x_src= x_out - shear_x * y_out
            y_src= y_out - shear_y * x_out
            output[y_out, x_out]= zoom_image(image, x_src, y_src)
    return output