from skimage.measure import compare_ssim as skimage_ssim
from mas.decorators import _vectorize
import numpy as np

@_vectorize(signature='(a,b),(c,d)->()', included=[0, 1])
def compare_ssim(x, y):
    return skimage_ssim(x, y, data_range=np.max(y) - np.min(y))
