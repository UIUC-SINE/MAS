#!/usr/bin/env python3
# Ulas Kamaci 2018-08-27

import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from mas.sse_cost import block_mul
# from psf_generator import photon_sieve, incoherent_psf
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.stats import poisson

def image_numpyer(*, inpath, outpath, size, upperleft):
    """
    Read png/jpg image, crop it with the specified size and location, and save
    it as an npy file

    Args:
        inpath (string): Path to the input png/jpg file
        outpath (string): Path to the output npy file
        size (tuple): Integer or tuple of 2 integers indicating the height and width
        of the cropped image, respectively.
        upperleft (tuple): Tuple of 2 integers indicating the coordinate of
        the upper left corner of the cropped image
    """
    if type(size) is int:
        size = (size,size)
    x = np.array(Image.open(inpath).convert('L'))
    x = x[upperleft[0]:upperleft[0]+size[0], upperleft[1]:upperleft[1]+size[1]]
    np.save(outpath, x)

def image_selector(*, inpath, size, upperleft):
    """
    Read png/jpg image, crop it with the specified coordinates and size, and
    show the full image vs cropped image.
    Puts a box around the cropped part for visualization. Box code might need
    adjustment for non-SDO images.

    Args:
        inpath (string): Path to the input png/jpg file
        size (tuple): Integer or tuple of 2 integers indicating the height and
        width of the cropped image, respectively.
        upperleft (tuple): Tuple of 2 integers indicating the coordinate of
        the upper left corner of the cropped image
    """
    if type(size) is int:
        size = (size,size)
    x = np.array(Image.open(inpath).convert('L'))
    x_c = x[upperleft[0]:upperleft[0]+size[0], upperleft[1]:upperleft[1]+size[1]]
    xmax = np.max(x[: int(x.shape[0]*0.94), :]) # to preserve the dynamic range

    # put a frame around the selected part
    t = 10
    x[upperleft[0] - t:upperleft[0],
    upperleft[1]: upperleft[1] + size[1]] = xmax

    x[upperleft[0] + size[0]+1: upperleft[0] + size[0] + t,
    upperleft[1]: upperleft[1] + size[1]] = xmax

    x[upperleft[0] - t: upperleft[0] + size[0] + t,
    upperleft[1] - t: upperleft[1]] = xmax

    x[upperleft[0] - t: upperleft[0] + size[0] + t,
    upperleft[1] + size[1] + 1: upperleft[1] + size[1] + t] = xmax

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(x[: int(x.shape[0]*0.94), :]) # crop the text to preserve dynamic range (for SDO)
    ax2.imshow(x_c)

def read_image(filename):
    """
    Read the spectral image-cube that is in the hdf5 file. Output it and
    its attributes (like wavelengths).

    Args:
        filename (string): Path of the hdf5 file in string format

    Returns:
        data_cube (ndarray): 3D Numpy array consisting of the bands
        wavelengths (ndarray): Wavelengths of the bands (ordered short to long)
    """

    # open the file in reading mode
    data = h5py.File(filename, 'r')

    # get the dataset which contains the compressed images in binary format
    dset = data['binary_data']

    # get the number of bands
    p = dset.shape[0]

    # get the first band to extract the dimensions
    temp = cv2.imdecode(dset[0], cv2.IMREAD_GRAYSCALE)

    # extract the dimensions
    [aa, bb] = temp.shape

    # form the numpy data_cube
    data_cube = np.zeros((p,1,aa,bb))
    data_cube[0,0,:,:] = temp
    for i in np.arange(1,p):
        data_cube[i,0,:,:] = cv2.imdecode(dset[i], cv2.IMREAD_GRAYSCALE)

    # get the wavelegths attribute from dset
    wavelengths = dset.attrs['wavelengths']

    # close the hdf file
    data.close()

    return data_cube, wavelengths


def psf_clipper(psf, energy_percent):
    """
    Clip the psf based on the given energy percent. A binary search is applied
    to determine the clipping boundary.

    Args:
        psf (ndarray): psf
        energy_percent (float): minimum energy percetage ratio between the
        clipped and the full size psfs.

    Returns:
        clipped_psf (ndarray): the clipped psf
    """

    s = psf.shape[0]
    m = int((s - 1) / 2)
    required_energy = np.sum(psf * psf) * energy_percent / 100
    left = 0
    right = m
    while right - left > 1:
        mid = int((left + right) / 2)
        psf_ = psf[m - mid : m + mid + 1 , m - mid : m + mid + 1]
        if np.sum(psf_ * psf_) > required_energy:
            right = mid
        else:
            left = mid
    return psf[m - right : m + right + 1 , m - right : m + right + 1]


def get_psfarray(photon_sieve,incoherent_psf, *, psf_path,
    sieve_diameter = 160e-3, smallest_hole_diameter = 5e-6,
    hole_diameter_to_zone_width = 1.53096,
    open_area_ratio = 0.6,
    source_wavelengths = np.array([33.4, 33.5]) * 1e-9,
    plane_distances = np.array([3.7425,3.7313]),
    source_distance=float(150e9), temp_psf_size = 1001, comments = ""
):
    """
    Given the sieve design, source wavelengths, and measurement plane
    distances, generate the psfs and place them into a 4D ndarray. If
    there are k measurement planes and p sources, and the psf dimension
    is psfsize by psfsize, then the dimension of this 4D ndarray is
    (k, p, psfsize, psfsize).

    Args:
        photon_sieve (function): generates photon sieve hole locations
        and diameters
        incoherent_psf (function): generates an incoherent photon sieve PSF
        psf_path (string): the path to which the computed array is saved
        Its form is MAS_PSFS_Kx_Py_Dz_wt_v01.h5 where x: num of meas planes
        y: num of sources, z: diameter of sieve in mm, t:min fabr sturct in µm
        replace x,y,z,t with appropriate values
        sieve_diameter (float): photon sieve diameter
        smallest_hole_diameter (float): diameter of holes on outermost zone
        hole_diameter_to_zone_width (float): ratio of hole diameter to zone
        open_area_ratio (float): ratio of hole area to zone area (?)
        source_wavelengths (ndarray): wavelengths of sources
        plane_distances (ndarray): distances of measurement planes
        source_distance (float): distance of the source
        comment (string): comments, notes about the generated psf

    """
    if type(source_wavelengths) is float:
        source_wavelengths = np.array([source_wavelengths])

    if type(plane_distances) is float:
        plane_distances = np.array([plane_distances])

    # Generate sieve hole locations and diameters
    white_zones = photon_sieve(
        sieve_diameter=sieve_diameter,
        smallest_hole_diameter=smallest_hole_diameter,
        hole_diameter_to_zone_width=hole_diameter_to_zone_width,
        open_area_ratio=open_area_ratio
    )

    k = plane_distances.shape[0]
    p = source_wavelengths.shape[0]

    # Find the most out of focus psf
    # This psf will be used to determine the psf size
    focal_distances = sieve_diameter * smallest_hole_diameter / source_wavelengths

    if max(plane_distances) - min(focal_distances) > max(focal_distances) - min(plane_distances):
        plane_index = k - 1
        wavelength_index = p - 1
        check = 1
    else:
        plane_index = 0
        wavelength_index = 0
        check = 0

    # Find the (source, plane distance) that gives the most out of focus psf
    plane_distance = plane_distances[plane_index]
    source_wavelength = source_wavelengths[wavelength_index]

    pixel_pitch = 0.5 * min(plane_distances) / max(focal_distances) * smallest_hole_diameter

    psf_defocused = incoherent_psf(
                white_zones = white_zones,
                source_wavelength = float(source_wavelength),
                plane_distance = plane_distance,
                sampling_interval = pixel_pitch,
                sieve_diameter = float(sieve_diameter),
                smallest_hole_diameter = float(smallest_hole_diameter),
                psf_width = temp_psf_size
            )

    psf_defocused_clipped = psf_clipper(psf_defocused, 99.9)
    psf_width = psf_defocused_clipped.shape[0]
    psfs = np.zeros((k,p,psf_width,psf_width))

    # generate incoherent psfs for each wavelength and plane location
    for m, plane_distance in enumerate(plane_distances):

        for n, source_wavelength in enumerate(source_wavelengths):
            if (m == 0 and n == 0 and check == 0) or (
            m == k-1 and n == p-1 and check == 1):
                psfs[m,n,:,:] = psf_defocused_clipped
                print('source: %d/%d , plane: %d/%d' % (n+1,p,m+1,k))
                continue
            psf = incoherent_psf(
                white_zones = white_zones,
                source_wavelength = float(source_wavelength),
                plane_distance = plane_distance,
                sampling_interval = pixel_pitch,
                sieve_diameter = float(sieve_diameter),
                smallest_hole_diameter = float(smallest_hole_diameter),
                psf_width = temp_psf_size
            )
            t1 = int((temp_psf_size-1)/2)
            t2 = int((psf_width-1)/2)

            psf = psf[t1-t2:t1+t2+1 , t1-t2:t1+t2+1]
            psfs[m,n,:,:] = psf
            print('source: %d/%d , plane: %d/%d' % (n+1,p,m+1,k))

    # File name sample: MAS_PSFS_K2_P2_D160_w5_v01.h5
    with h5py.File(psf_path) as f:
        dset = f.create_dataset('psfs', data = psfs)

        # The attributes of the dataset
        dset.attrs['wavelengths']   = source_wavelengths
        dset.attrs['plane_distances']   = plane_distances
        dset.attrs['source_distance']   = source_distance
        dset.attrs['sieve_diameter']   = sieve_diameter
        dset.attrs['smallest_hole_diameter']   = smallest_hole_diameter
        dset.attrs['hole_diameter_to_zone_width']   = hole_diameter_to_zone_width
        dset.attrs['open_area_ratio']   = open_area_ratio
        dset.attrs['psf_size']   = psfs.shape[3]
        dset.attrs['comment']   = comments

    return 0

def get_meas(*, source_path, psf_path, meas_path, mode = 'Circular', **kwargs):
    """
    Circularly convolve the sources and psfs to obtain measurements.
    Then add Gaussian noise, or apply Poisson noise.
    TODO: Add linear convolution mode

    Args:
        source_path (list): list of paths (strings) to the source files
        psf_path (string): path to the psfs hdf5 file
        meas_path (string): path to the measurements to be stored as hdf5
        Its form is MAS_MEAS_Kx_Py_Dz_wt_v01.h5 where x: num of meas planes
        y: num of sources, z: diameter of sieve in mm, t:min fabr sturct in µm
        replace x,y,z,t with appropriate values
        kwargs: extra keyword arguments to pass to add_noise function

    Returns:
        ndarray that is the noisy version of the input
    """

    sources = np.zeros( (len(source_path),1) + np.load(source_path[0]).shape )

    for i, path in enumerate(source_path):
        sources[i,0,:,:] = np.load(path)

    # with h5py.File(psf_path) as f:
    f = h5py.File(psf_path, 'r')
    psfs = f['psfs']

    assert sources.shape[0] == psfs.shape[1], "source dimension of psf and 3D image do not match"
    [p,_,aa,bb] = sources.shape
    [k,p,ss,ss] = psfs.shape
    psfs2 = np.zeros((k,p,aa,bb))

    # reshape psfs
    for i in range(k):
        for j in range(p):
            psfs2[i,j,:,:] = size_equalizer(psfs[i,j,:,:], [aa,bb])

    meas = fftshift(
        ifft2(
            block_mul(
                fft2(ifftshift(psfs2,axes=(2,3))) ,
                fft2(ifftshift(sources,axes=(2,3)))
            )
        ), axes=(2,3)
    )

    meas = np.real(meas) # remove the imag(which is due to numerical errors)
    noisy_meas = np.zeros(meas.shape)

    for i in range(k):
        noisy_meas[i,0,:,:] = add_noise(signal = meas[i,0,:,:], **kwargs)

    with h5py.File(meas_path) as f2:
        dset = f2.create_dataset('measset', shape= (2,)+meas.shape, dtype= 'f')
        dset[0] = meas
        dset[1] = noisy_meas

        # Pass the attributes of the psfs
        for i in list(psfs.attrs.keys()):
            dset.attrs[i] = psfs.attrs[i]

        dset.attrs['meas_size'] = noisy_meas.shape[2:]
        dset.attrs['conv_mode'] = mode
        if 'snr_dB' in kwargs:
            dset.attrs['snr_dB'] = kwargs['snr_dB']
        if 'exp_time' in kwargs:
            dset.attrs['exp_time'] = kwargs['exp_time']
        dset.attrs['comment']   = "dset[0] is noiseless measurements, " +\
        "and dset[1] is noisy measurements"

    f.close()

    return 0

def add_noise(*, signal, snr_dB = None, exp_time = None, model = 'Gaussian'):
    """
    Add noise to the given signal at the specified level.

    Args:
        signal (ndarray): noise-free input signal
        snr_dB (float): 10*log_10(SNR) where SNR is defined as the ratio of
        variance of the input signal to the variance of the noise. Necessary
        input for Gaussian noise.
        exp_time (float): Exposure time in seconds. Necessary input for
        Poisson noiseself.
        model (string): String that specifies the noise model. The 2 options are
        `Gaussian` and `Poisson`

    Returns:
        ndarray that is the noisy version of the input
    """
    assert model == 'Gaussian' or 'Poisson', "select the model correctly"
    if model == 'Gaussian':
        assert snr_dB is not None, "please specify snr_dB"
        var_sig = np.var(signal)
        var_noise = var_sig/(10**(snr_dB/10))
        out = np.random.normal(loc = signal, scale = np.sqrt(var_noise))
    elif model == 'Poisson':
        assert exp_time is not None, "please specify exp_time in seconds"
        beta = 1
        out = poisson.rvs(signal*exp_time*beta)
    return out


def size_equalizer(x, ref_size):
    """
    Crop or zero-pad a 2D array so that it has the size `ref_size`.
    Both cropping and zero-padding are done such that the symmetry of the
    input signal is preserved.
    Args:
        x (ndarray): array which will be cropped/zero-padded
        ref_size (list): list containing the desired size of the array [r1,r2]
    Returns:
        ndarray that is the cropper/zero-padded version of the input
    """
    [i1, i2] = x.shape
    [r1, r2] = ref_size
    [f1, f2] = [r1 - i1, r2 - i2]
    m1 = int(i1/2)
    m2 = int(i2/2)
    down = int((r1 - 1) / 2)
    up = r1 - down - 1
    right = int((r2 - 1) / 2)
    left = r2 - right -1

    out = x

    for i,k in enumerate((f1,f2)):

        if k > 0:
            after = int(k/2)
            before = k - after
            if i == 0:
                out = np.pad(out, ((before, after), (0, 0)), mode = 'constant')
            else:
                out = np.pad(out, ((0, 0), (before, after)), mode = 'constant')

        elif k == 0:
            out = np.pad(out, ((0, 0), (0, 0)), mode = 'constant')

        elif k < 0:
            if i == 0:
                out = out[m1 - up : m1 + down + 1, :]
            else:
                out = out[:, m2 - left : m2 + right + 1]

    return out
