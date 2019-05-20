from mas.psf_generator import PhotonSieve, PSFs
from mas.forward_model import add_noise, get_measurements
from mas.data import strands
from skimage.measure import compare_ssim
from bayes_opt import BayesianOptimization
from mas.deconvolution import tikhonov
import numpy as np
from matplotlib import pyplot as plt

truth = strands[0:1]
ps = PhotonSieve()

wavelengths = np.array([33.4e-9])

psfs = PSFs(
    ps,
    source_wavelengths=wavelengths,
    measurement_wavelengths=wavelengths
)
measured = get_measurements(psfs=psfs, sources=truth, real=True)
measured = add_noise(measured, model='poisson', max_count=10)

# Bounded region of parameter space
pbounds = {'tikhonov_lam': (5e-20, 5e-1)}

def cost(tikhonov_lam):

    recon = tikhonov(
            sources=measured,
            psfs=psfs,
            measurements=measured,
            tikhonov_lam=tikhonov_lam
    )[0]

    plt.imshow(recon)
    plt.show()
    plt.pause(.05)
    return compare_ssim(
        truth[0],
        recon
    )

optimizer = BayesianOptimization(
    cost,
    pbounds=pbounds,
    random_state=1,
)

# %% optimize -----

np.seterr(all='ignore')
optimizer.maximize(
    init_points=2,
    n_iter=30,
)
