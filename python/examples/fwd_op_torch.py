from mas.psf_generator import PSFs, PhotonSieve
from mas.strand_generator import strands
from mas.forward_model import size_equalizer, get_fwd_op_torch
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from mas.tracking import mb_kernel
from skimage.transform import resize, rescale
from torch.nn.functional import conv2d

x = strands(image_width=128)
fwd_op, psf_final = get_fwd_op_torch()

y = fwd_op(x)

plt.figure()
plt.imshow(x, cmap='hot')
plt.colorbar()
plt.title('Input Image')
plt.show()

plt.figure()
plt.imshow(y.squeeze().cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title('Blurred Image')
plt.show()

plt.figure()
plt.imshow(psf_final[0,0].cpu().numpy(), cmap='gray')
plt.colorbar()
plt.title('Combined PSF')
plt.show()