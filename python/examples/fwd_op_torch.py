from mas.psf_generator import PSFs, PhotonSieve
from mas.strand_generator import strands
from mas.forward_model import size_equalizer, get_fwd_op_torch
from slitless.forward import add_noise
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from mas.tracking import mb_kernel
from skimage.transform import resize, rescale
from torch.nn.functional import conv2d

x = strands(image_width=128)

diameter=75e-3 # meters
smallest_hole_diameter=16e-6 # meters
wavelengths=np.array([30.4e-9]) # meters
plane_offset=15e-3 # meters
drift_angle=-45 # degrees
drift_velocity=0.10e-3 # meters / s
jitter_rms=3e-6 # meters
frame_rate=7.5 # Hz
pixel_size=7e-6 # meters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fwd_op, psf_final = get_fwd_op_torch(
    diameter=diameter,
    smallest_hole_diameter=smallest_hole_diameter,
    wavelengths=wavelengths,
    plane_offset=plane_offset,
    drift_angle=drift_angle,
    drift_velocity=drift_velocity,
    jitter_rms=jitter_rms,
    frame_rate=frame_rate,
    pixel_size=pixel_size,
    device=device
)

meas_noiseless = fwd_op(x)

meas = add_noise(meas_noiseless, dbsnr=10, noise_model='Poisson')

plt.figure()
plt.imshow(x, cmap='hot')
plt.colorbar()
plt.title('Input Image')
plt.show()

plt.figure()
plt.imshow(meas_noiseless.squeeze().cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title('Blurred Image')
plt.show()

plt.figure()
plt.imshow(meas.squeeze().cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title('Blurred Noisy Image')
plt.show()

plt.figure()
plt.imshow(psf_final[0,0].cpu().numpy(), cmap='gray')
plt.colorbar()
plt.title('Combined PSF')
plt.show()