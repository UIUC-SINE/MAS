import numpy as np
from mas.strand_generator import strand_video
from mas.tracking import phase_correlate
from mas.forward_model import crop
from mas.reconstruction import coadd
import matplotlib.pyplot as plt

# %% forward

max_count = 20
wavelengths = np.array([30.4e-9])
# CCD parameters
frame_rate = 8 # Hz
ccd_size = np.array((750, 750))
start = (400, 0)
pixel_size = 14e-6 # meters
# simulation subpixel parameters
resolution_ratio = 2 # CCD pixel_size / simulation pixel_size
fov_ratio = 2 # simulated FOV / CCD FOV
# sieve parameters
diameter = 75e-3 # meters
smallest_hole_diameter = 17e-6 # meters
# drift parameters
exp_time = 10 # s
drift_angle = np.deg2rad(-45) # radians
drift_velocity = 0.2e-3 # meters / s
true_drift = drift_velocity / frame_rate * np.array(
    [np.cos(drift_angle), np.sin(drift_angle)]
) / pixel_size # true drift from one to the next frame. notation: (x,y)

frames, frames_clean, scene, topleft_coords = strand_video(
        # experiment parameters
        exp_time=exp_time, # s
        drift_angle=drift_angle, # radians
        drift_velocity=drift_velocity, # meters / s
        max_count=max_count,
        wavelengths=wavelengths,
        # CCD parameters
        frame_rate=frame_rate, # Hz
        ccd_size=ccd_size,
        start=start,
        pixel_size=pixel_size, # meters
        # simulation subpixel parameters
        resolution_ratio=resolution_ratio, # CCD pixel_size / simulation pixel_size
        fov_ratio=fov_ratio, # simulated FOV / CCD FOV
        # sieve parameters
        diameter=diameter, # meters
        smallest_hole_diameter=smallest_hole_diameter # meters
)

# %% slide1
def adjust_pc(pc):
    """Shift phase correlation to center and crop"""
    # FIXME
    pc[0, :] = pc[-1, :]

    return crop(np.fft.fftshift(pc), width=100).real

pc_noiseless = adjust_pc(phase_correlate(frames_clean[0], frames_clean[1]))
pc_noisy = adjust_pc(phase_correlate(frames[0], frames[1]))

pc_noisy_sum = np.zeros(frames[0].shape, dtype='complex128')
for x, y in zip(frames[:-1], frames[1:]):
    pc_noisy_sum += phase_correlate(x, y)
pc_noisy_sum = adjust_pc(pc_noisy_sum)

pc_noisy_01 = adjust_pc(phase_correlate(frames[0], frames[1]))
pc_noisy_12 = adjust_pc(phase_correlate(frames[1], frames[2]))
pc_noisy_23 = adjust_pc(phase_correlate(frames[2], frames[3]))
pc_noisy_7879 = adjust_pc(phase_correlate(frames[78], frames[79]))


plt.imsave('figures/1_frame0.png', frames_clean[0])
plt.imsave('figures/1_frame1.png', frames_clean[1])

plt.imsave('figures/1_frame0_noisy.png', frames[0])
plt.imsave('figures/1_frame1_noisy.png', frames[1])
plt.imsave('figures/1_frame2_noisy.png', frames[2])
plt.imsave('figures/1_frame3_noisy.png', frames[3])
plt.imsave('figures/1_frame79_noisy.png', frames[-1])
plt.imsave('figures/1_pc_noisy_01.png', pc_noisy_01)
plt.imsave('figures/1_pc_noisy_12.png', pc_noisy_12)
plt.imsave('figures/1_pc_noisy_23.png', pc_noisy_23)
plt.imsave('figures/1_pc_noisy_7879.png', pc_noisy_7879)

plt.imsave('figures/1_pc.png', pc_noiseless)
plt.imsave('figures/1_pc_noisy.png', pc_noisy)
plt.imsave('figures/1_pc_noisy_sum.png', pc_noisy_sum)

# # %% correlate

# pcs = []
# for x, y in zip(frames[[0, 1, 2, 3, -2]], frames[[1, 2, 3, 4, -1]]):
#     pcs.append(phase_correlate(x, y))
# pcs = np.array(pcs)
# pcs = np.roll(pcs, ccd_size // 2, (1, 2))


# %% coadd

coadded = coadd(frames, true_drift)

plt.imsave('figures/4_frame0.png', frames_clean[0])
plt.imsave('figures/4_frame20.png', frames_clean[20])
plt.imsave('figures/4_frame40.png', frames_clean[40])
plt.imsave('figures/4_frame60.png', frames_clean[60])
plt.imsave('figures/4_frame79.png', frames_clean[79])
plt.imsave('figures/4_coadded.png', coadded)
