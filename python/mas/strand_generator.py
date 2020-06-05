import numpy as np
import cv2, skimage
from scipy.sparse import coo_matrix
from scipy.ndimage.filters import gaussian_filter
from numpy.random import normal
from mas.decorators import store_kwargs
from tqdm import tqdm
from cachalot import Cache

def strand(theta, x, thickness=22, intensity=1, image_width=512):
    """Generate single strand image"""

    pt1 = (x + int(image_width * np.tan(np.deg2rad(theta))), 0) # point where it crosses y=0 line
    pt2 = (x, image_width) # point where it crosses y=image_width line

    img = np.zeros((image_width, image_width))
    return cv2.line(img, pt1, pt2, intensity, thickness=thickness)

def strands(num_strands=100, thickness=22, min_angle=-20, max_angle=20,
            image_width=160, initial_width=512):
    """Generate nanoflare image

    Args:
        num_strands (int): number of strands in image
        thickness (int): pixel width of one strand
        max_angle (float): maximum offset from vertical of strands in degrees
        image_width (int): width of returned image
        initial_width (int): width of computed image before downscaling

    Returns:
        ndarray: (image_width, image_width) image containing strands"""

    img = np.zeros((initial_width,initial_width))
    for s in tqdm(range(num_strands), desc='Strands', leave=None):
        theta = np.random.uniform(min_angle, max_angle)
        x  = np.random.randint(
            0 if theta < 0 else initial_width * -np.tan(np.deg2rad(theta)),
            initial_width - initial_width * np.tan(np.deg2rad(theta)) if theta < 0 else initial_width
        )
        intensity = np.random.rand()

        img += strand(theta, x, thickness, intensity, initial_width)

    imgr = gaussian_filter(img, sigma=2)
    imgr = skimage.transform.resize(imgr, (image_width, image_width))
    imgr = imgr / imgr.max()
    return imgr

def get_visors_noise(dark_current=0.8, background=2, read_noise=1):
    """
    Return a noise function which generates noisy frames out of noiseless ones
    according to the given parameters

    Args:
        dark_current (float): [dark current counts / pixel / s]
        background (float): [background photon counts / pixel / s]
        read_noise (float): standard deviation of the Gaussian readout noise [1/read]
    """
    def func(x, frame_rate):
        x = np.random.poisson(x + (dark_current + background) / frame_rate)
        return normal(loc=x, scale=read_noise)
    return func


class StrandVideo(object):

    """A class for holding strand video state

    Args:
        exp_time (int): total experiment time (s)
        drift_angle (float): angle of drift (degrees)
        drift_velocity (float): velocity of drift (m / s)
        max_count (int): maximum photon rate
        noise_model (function): function which takes 'frames', 'frame_rate',
            'dark_current', 'background', 'read_noise' and returns noisy frames

        ccd_size (tuple): resolution of physical CCD
        pixel_size (float): width of CCD pixel (m)

        resolution_ratio (int): subpixel scale factor
            (ccd pixel size / simulation pixel size)
        fov_ratio (int): (simulation FOV / CCD FOV)

        num_strands (int): number of nanoflares

        diameter (float): sieve diameter (m)
        smallest_hole_diameter (float): smallest sieve hole diam (m)
    """

    @store_kwargs
    def __init__(
            self,
            # experiment parameters
            exp_time=10, # s
            drift_angle=-45, # degrees
            drift_velocity=0.2e-3, # meters / s
            angle_velocity=0, # deg / s
            max_count=20,
            noise_model=get_visors_noise(),
            wavelengths=np.array([30.4e-9]),
            # CCD parameters
            frame_rate=4, # Hz
            ccd_size=np.array((750, 750)),
            start=(1500, 1500),
            pixel_size=14e-6, # meters
            # simulation subpixel parameters
            resolution_ratio=2, # CCD pixel_size / simulation pixel_size
            fov_ratio=2, # simulated FOV / CCD FOV
            # strand parameters
            scene=None, # pregenerated scene
            num_strands=100, # num strands per CCD FOV
            # sieve parameters
            diameter=75e-3, # meters
            smallest_hole_diameter=17e-6, # meters
    ):
        from mas.psf_generator import PSFs, PhotonSieve
        from mas.forward_model import get_measurements
        from mas.tracking import video

        self.ps = PhotonSieve(diameter=diameter, smallest_hole_diameter=smallest_hole_diameter)
        self.psfs = PSFs(
            self.ps,
            sampling_interval=pixel_size / resolution_ratio,
            source_wavelengths=wavelengths,
            measurement_wavelengths=wavelengths
        )

        # load common high resolution scene from file
        if scene is not None:
            scene = np.copy(scene)
        elif (num_strands, fov_ratio, resolution_ratio, ccd_size[0]) == (100, 2, 5, 160):
            from mas.data import strand_highres
            self.scene = strand_highres
        elif (num_strands, fov_ratio, resolution_ratio, ccd_size[0]) == (100, 2, 2, 750):
            from mas.data import strand_highres2
            self.scene = strand_highres2
        else:
            self.scene = strands(
                num_strands=int(num_strands*fov_ratio*ccd_size[0]/160),
                thickness=22 * resolution_ratio,
                image_width=ccd_size[0] * resolution_ratio * fov_ratio,
                initial_width=3 * ccd_size[0] * resolution_ratio * fov_ratio
            )

        self.scene = get_measurements(sources=self.scene[np.newaxis, :, :], psfs=self.psfs)[0]

        self.scene *= max_count / np.max(self.scene)

        self.frames_clean, self.midpoint_coords = video(
            scene=self.scene,
            frame_rate=frame_rate,
            exp_time=exp_time,
            drift_angle=drift_angle,
            drift_velocity=drift_velocity,
            angle_velocity=angle_velocity,
            ccd_size=ccd_size,
            resolution_ratio=resolution_ratio,
            pixel_size=pixel_size,
            start=start
        )

        # add noise to the frames
        if noise_model is not None:
            self.frames = noise_model(self.frames_clean, frame_rate)
        else:
            self.frames = self.frames_clean

        self.true_drift = drift_velocity / frame_rate * np.array([
            -np.sin(np.deg2rad(drift_angle)), # use image coordinate system
            np.cos(np.deg2rad(drift_angle))
        ]) / pixel_size
