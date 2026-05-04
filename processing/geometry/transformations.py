import numpy as np 
from processing.interpolation.zoom import zoom_image, bilinear_interpolation_pixel

def rotate(image, angle_degrees):
    angle_radian=np.radians(angle_degrees)
    h,w=image.shape[:2]
    cx, cy= w/2, h/2

    cos_a=abs(np.cos(angle_radian))
    sin_a=abs(np.sin(angle_radian))
    new_w= int(h*sin_a + w*cos_a)
    new_h= int(h*cos_a + w*sin_a)
   

    new_cx, new_cy= new_w/2, new_h/2

    if image.ndim == 3:
        output= np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8)
    else:
        output= np.zeros((new_h, new_w), dtype=np.uint8)

    for y_out in range(new_h):
        for x_out in range(new_w):
            dx, dy= x_out - new_cx, y_out - new_cy
            x_src= dx * np.cos(angle_radian) + dy * np.sin(angle_radian) + cx
            y_src= -dx * np.sin(angle_radian) + dy * np.cos(angle_radian) + cy
            output[y_out, x_out]=np.clip(bilinear_interpolation_pixel(image, x_src, y_src),0,255)
    return output


def shear(image,shear_x=0.0, shear_y=0.0):
    h,w=image.shape[:2]
    output= np.zeros_like(image)

    for y_out in range(h):
        for x_out in range(w):
            x_src= x_out - shear_x * y_out
            y_src= y_out - shear_y * x_out
            output[y_out, x_out]=np.clip(bilinear_interpolation_pixel(image, x_src, y_src),0,255)
    return output

def scale(image, scale_factor, method="bilinear"):
    return zoom_image(image, scale_factor, method)