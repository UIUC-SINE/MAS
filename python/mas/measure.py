from skimage.measure import compare_ssim as skimage_ssim
from skimage.measure import compare_psnr as skimage_psnr
import numpy as np

def compare_ssim(x, y):
    return skimage_ssim(x, y, data_range=np.max(y) - np.min(y))

def compare_psnr(x, y):
    return skimage_psnr(x, y, data_range=np.max(y) - np.min(y))
