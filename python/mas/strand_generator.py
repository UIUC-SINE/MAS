import numpy as np
import cv2, skimage
from scipy.sparse import coo_matrix

def strand(theta, x, thickness=7, intensity=1, image_width=160):
    """Generate single strand image"""

    pt1 = (x + int(image_width * np.tan(np.deg2rad(theta))), 0) # point where it crosses y=0 line
    pt2 = (x, image_width) # point where it crosses y=image_width line

    img = np.zeros((image_width, image_width))
    return cv2.line(img, pt1, pt2, intensity, thickness=thickness)

def strands(num_strands=100, thickness=5, max_angle=30,
            image_width=160, initial_width=512):
    """Generate nanoflare image

    Args:
        num_strands (int): number of strands in image
        thickness (int): pixel width of one strand
        max_angle (float): maximum offset from vertical of strands in degrees
        image_width (int): width of returned image
        initial_width (int): width of computed image before downscaling

    Returns:
        ndarray: (image_width, image_width) image containing strands"""

    img = np.zeros((image_width,image_width))
    for _ in range(num_strands):
        theta = np.random.uniform(-max_angle, max_angle)
        x  = np.random.randint(
            0 if theta < 0 else image_width * -np.tan(np.deg2rad(theta)),
            image_width - image_width * np.tan(np.deg2rad(theta)) if theta < 0 else image_width
        )
        intensity = np.random.rand()

        img += strand(theta, x, thickness, intensity, image_width)

    imgr = skimage.transform.resize(img, (image_width, image_width))
    imgr = imgr / imgr.max()
    return imgr
