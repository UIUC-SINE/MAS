import numpy as np
import cv2, skimage
from scipy.sparse import coo_matrix
from scipy.ndimage.filters import gaussian_filter
from numpy.random import normal
from scipy.stats import poisson

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
    for s in range(num_strands):
        print('Strand {}/{}\r'.format(s, num_strands), end='')
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


def noise_model(x, frame_rate):

    # scale scene to max photon count
    x = poisson.rvs((x + 8 + 2) / frame_rate)
    return normal(loc=x, scale=10)

def strand_video(
        # experiment parameters
        exp_time=10, # s
        drift_angle=np.deg2rad(10), # radians
        drift_velocity=0.2e-3, # meters / s
        max_count=20,
        noise_model=noise_model,
        wavelengths=np.array([30.4e-9]),
        # CCD parameters
        frame_rate=8, # Hz
        ccd_size=(750, 750),
        start=(400,0),
        pixel_size=14e-6, # meters
        # simulation subpixel parameters
        resolution_ratio=2, # CCD pixel_size / simulation pixel_size
        fov_ratio=2, # simulated FOV / CCD FOV
        # strand parameters
        num_strands=100, # num strands per CCD FOV
        # sieve parameters
        diameter=75e-3, # meters
        smallest_hole_diameter=17e-6, # meters
):
    from mas.psf_generator import PSFs, PhotonSieve
    from mas.forward_model import get_measurements
    from mas.tracking import video

    ps = PhotonSieve(diameter=17e-6, smallest_hole_diameter=17e-6)
    psfs = PSFs(
        ps,
        sampling_interval=pixel_size / resolution_ratio,
        source_wavelengths=wavelengths,
        measurement_wavelengths=wavelengths
    )

    # load common high resolution scene from file
    if (num_strands, fov_ratio, resolution_ratio, ccd_size[0]) == (100, 2, 5, 160):
        from mas.data import strand_highres
        scene = strand_highres
    elif (num_strands, fov_ratio, resolution_ratio, ccd_size[0]) == (100, 2, 2, 750):
        from mas.data import strand_highres2
        scene = strand_highres2
    else:
        scene = strands(
            num_strands=int(num_strands*fov_ratio*ccd_size[0]/160),
            thickness=22 * resolution_ratio,
            image_width=ccd_size[0] * resolution_ratio * fov_ratio,
            initial_width=3 * ccd_size[0] * resolution_ratio * fov_ratio
        )

    scene = get_measurements(sources=scene[np.newaxis, :, :], psfs=psfs)[0]

    frames_clean, topleft_coords = video(
        scene=scene,
        frame_rate=frame_rate,
        exp_time=exp_time,
        drift_angle=drift_angle,
        drift_velocity=drift_velocity,
        ccd_size=ccd_size,
        resolution_ratio=resolution_ratio,
        pixel_size=pixel_size,
        start=start
    )

    frames_clean /= np.max(frames_clean)
    frames_clean *= max_count
    # add noise to the frames
    if noise_model is not None:
        frames = noise_model(frames_clean, frame_rate)
    else:
        frames = frames_clean

    return frames, frames_clean, scene, topleft_coords
