from csbs import csbs
import sse_cost
from psf_generator import generate_measurements
from plotting import fourier_slices
import numpy as np

measurements = generate_measurements(source_wavelengths=np.array([33.4,  33.7, 33.8]) * 1e-9,
                                        image_width=51,
                                        num_copies=10)

csbs(measurements, sse_cost, 290, lam=20)

plt = fourier_slices(measurements)
plt.savefig('csbs_fourier_slices.png')
