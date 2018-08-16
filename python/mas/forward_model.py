from mas.psf_generator import generate_measurements
from mas.sse_cost import block_mul, block_inv, block_herm
import scipy.misc
import numpy as np
from mas.sse_cost import init
from matplotlib import pyplot as plt
from mas.plotting import fourier_slices


sources = np.array([scipy.misc.face(gray=True)])[:, np.newaxis, :, :]
num_sources = sources.shape[0]
source_wavelengths = np.array([33.4]) * 1e-9

diameter = 2.5e-2
smallest_zone_width = 5e-6
# focal length and depth of focus for each wavelength
focal_lengths = diameter * smallest_zone_width / source_wavelengths

measurements = generate_measurements(source_wavelengths=source_wavelengths,
                                     planes=focal_lengths)

# find the paddings
pad_x1 = int(np.ceil((sources[0].shape[1] - measurements.image_width) / 2))
pad_x2 = (sources[0].shape[1] - measurements.image_width) // 2
pad_y1 = int(np.ceil((sources[0].shape[2] - measurements.image_width) / 2))
pad_y2 = (sources[0].shape[2] - measurements.image_width) // 2

measurements.psfs = np.pad(measurements.psfs,
                           ((0, 0), (0, 0), (pad_x1, pad_x2), (pad_y1, pad_y2)),
                           mode='constant')

init(measurements)

dbSNR = 5

measurements.psfs = np.fft.ifftshift(measurements.psfs)
measured_fft = block_mul(np.fft.fft2(measurements.psfs), np.fft.fft2(sources))
measured = np.fft.ifft2(measured_fft)
noise_var = np.var(measured) / 10**(dbSNR / 10)
measured += np.random.normal(loc=0, scale=np.sqrt(noise_var), size=measured.shape)

lam = 0.01
SIG_e_dft = measurements.initialized_data['GAM'] + lam *  np.einsum('ij,kl', np.eye(num_sources), measurements.initialized_data['LAM'])
SIG_e_dft2 = measurements.initialized_data['GAM']
reconstruction = np.fft.ifft2(block_mul(block_inv(SIG_e_dft),
                                        block_mul(block_herm(np.fft.fft2(measurements.psfs)),
                                                  np.fft.fft2(measured)
                                        )
))
reconstruction2 = np.fft.ifft2(block_mul(block_inv(SIG_e_dft2),
                                        block_mul(block_herm(np.fft.fft2(measurements.psfs)),
                                                  np.fft.fft2(measured)
                                        )
))

plt.subplot(2, 2, 1)
plt.imshow(np.abs(reconstruction[0, 0]))
plt.title('lam={}'.format(lam))
plt.subplot(2, 2, 2)
plt.imshow(np.abs(reconstruction2[0, 0]))
plt.title('lam=0')
plt.subplot(2, 2, 3)
plt.imshow(np.abs(sources[0, 0]))
plt.title('source')
plt.subplot(2, 2, 4)
plt.imshow(np.abs(measured[0, 0]))
plt.title('measured')

plt.show()
