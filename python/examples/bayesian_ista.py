from mas.psf_generator import PhotonSieve, PSFs
from mas.forward_model import add_noise, get_measurements
from mas.data import strands
from skimage.measure import compare_ssim
from bayes_opt import BayesianOptimization
from mas.deconvolution import ista
import numpy as np
from matplotlib import pyplot as plt

truth = strands[0:1]
ps = PhotonSieve()

wavelengths = np.array([33.4e-9])

psfs = PSFs(
    ps,
    source_wavelengths=wavelengths,
    measurement_wavelengths=wavelengths,
    num_copies=1
)
measured = get_measurements(psfs=psfs, sources=truth, real=True)
measured = add_noise(measured, model='poisson', max_count=10)

def cost(lam, time_step):

    recon = ista(
        psfs=psfs,
        measurements=measured,
        lam=lam,
        time_step=time_step,
        iterations=100
    )[0]

    plt.imshow(recon)
    plt.show()
    plt.pause(.05)
    return compare_ssim(
        truth[0],
        recon
    )


# Bounded region of parameter space
pbounds = {'lam': (1e-5, 1e-1), 'time_step':(1e-5, 1e-1)}
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
