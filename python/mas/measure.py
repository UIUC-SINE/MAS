from skimage.measure import compare_ssim as skimage_ssim
from skimage.measure import compare_psnr as skimage_psnr
from mas.decorators import _vectorize
import numpy as np

@_vectorize(signature='(a,b),(c,d)->()', included=[0, 1])
def compare_ssim(*, truth, estimate):
    return skimage_ssim(estimate, truth, data_range=np.max(truth) - np.min(truth))

def compare_psnr(*, truth, estimate):
    return skimage_psnr(estimate, truth, data_range=np.max(truth) - np.min(truth))
