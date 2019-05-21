from skimage.measure import compare_ssim as skimage_ssim
import numpy as np

def compare_ssim(x, y):
    return skimage_ssim(x, y, data_range=np.max(y) - np.min(y))
