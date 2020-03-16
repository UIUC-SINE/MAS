from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from mas.decorators import _vectorize
import numpy as np

@_vectorize(signature='(a,b),(c,d)->()', included=['truth', 'estimate'])
def compare_ssim(*, truth, estimate):
    return skimage_ssim(estimate, truth, data_range=np.max(truth) - np.min(truth))

def compare_psnr(*, truth, estimate):
    return skimage_psnr(estimate, truth, data_range=np.max(truth) - np.min(truth))
