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

def cost(lam_exp, time_step_exp):

    recon = ista(
        psfs=psfs,
        measurements=measured,
        lam=10**lam_exp,
        time_step=10**time_step_exp,
        iterations=100
    )[0]

    plt.imshow(recon)
    plt.show()
    plt.pause(.05)

    cost = compare_ssim(
        truth[0],
        recon
    )

    return cost if cost > 0 else 0


# Bounded region of parameter space
pbounds = {'lam_exp': (-5, -1), 'time_step_exp':(-5, -1)}
optimizer = BayesianOptimization(
    cost,
    pbounds=pbounds,
    random_state=1,
)

# %% optimize -----

np.seterr(all='ignore')
optimizer.maximize(
    acq='ucb',
    kappa=0.1,
    init_points=2,
    n_iter=10,
)
