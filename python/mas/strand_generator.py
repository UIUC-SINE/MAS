import numpy as np
import cv2, skimage

def strand(theta, x, thickness=7, intensity=1, image_width=160):
    """Generate single strand image"""

    pt1 = (x - int(image_width * np.tan(theta)), 0) # point where it crosses y=0 line
    pt2 = (x, image_width) # point where it crosses y=image_width line

    img = np.zeros((image_width, image_width))
    return cv2.line(img, pt1, pt2, intensity, thickness=thickness)

def strands(numstrands=100, thickness=7, max_angle=20,
            image_width=160, initial_width=512):
    """Generate nanoflare image

    Args:
        numstrands (int): number of strands in image
        thickness (int): pixel width of one strand
        max_angle (float): maximum offset from vertical of strands in degrees
        image_width (int): width of returned image
        initial_width (int): width of computed image before downscaling

    Returns:
        ndarray: (image_width, image_width) image containing strands"""

    img = np.zeros((image_width,image_width))
    for _ in range(numstrands):
        theta = np.random.uniform(-max_angle, max_angle) / 180 * np.pi
        x  = np.random.randint(
            0 if theta > 0 else image_width * np.tan(theta),
            image_width + image_width * np.tan(theta) if theta > 0 else image_width
        )
        intensity = np.random.rand()

        img += strand(theta, x, thickness, intensity, image_width)

    imgr = skimage.transform.resize(img, (image_width, image_width))
    imgr = imgr / imgr.max()
    return imgr
