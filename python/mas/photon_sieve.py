#!/bin/python
# Evan Widloski, Ulas Kamaci - 2018-08-16
# Photon sieve generator based on Tunc Alkanat's code

import numpy as np
from matplotlib import pyplot as plt

# ----- Photon sieve generation -----

focal_length = 3.7425 # (m)
target_wavelength = 33.4e-9 # (m)
smallest_hole_diameter = 30.24e-6 # (m)
hole_diameter_to_zone_width = 1.53096
open_area_ratio = 0.6

sieve_diameter = target_wavelength * focal_length / smallest_hole_diameter

num_white_zones = np.floor(sieve_diameter**2 / (8 * target_wavelength * focal_length))
zone_radii = np.sqrt(2 * target_wavelength * focal_length * np.arange(1, num_white_zones + 1))
zone_widths = target_wavelength * focal_length / (2 * zone_radii)
hole_diameters = hole_diameter_to_zone_width * zone_widths
hole_counts = np.round(8 * open_area_ratio * zone_widths * zone_radii / (hole_diameters**2))

white_zones = []
for hole_diameter, hole_count, zone_radius in zip(hole_diameters, hole_counts, zone_radii):
    white_zone = {}
    white_zone['hole_diameter'] = hole_diameter
    white_zone['hole_coordinates'] = []
    white_zone['inner_radius'] = zone_radius - hole_diameter / 2
    white_zone['outer_radius'] = zone_radius + hole_diameter / 2
    white_zone['hole_count'] = hole_count
    for theta in 2 * np.pi * np.arange(hole_count) / hole_count:
        white_zone['hole_coordinates'].append(
            (
            zone_radius * np.cos(theta),
            zone_radius * np.sin(theta)
            )
        )

    white_zones.append(white_zone)


# ----- PSF computation -----

def a_func(x, y):
    radius = np.sqrt(x**2 + y**2)
    # return True if point falls inside hole
    for white_zone in white_zones:
        if radius >= white_zone['inner_radius']:
            if radius < white_zone['outer_radius']:
                theta = np.arctan2(y, x)
                closest_hole = int(
                    np.round(white_zone['hole_count'] * theta / (2 * np.pi))
                )
                # check if point falls in hole
                if np.sqrt(
                        (x - white_zone['hole_coordinates'][closest_hole][0])**2 +
                        (y - white_zone['hole_coordinates'][closest_hole][1])**2
                ) < white_zone['hole_diameter'] / 2:
                    return True
                else:
                    return False
        else:
            return False
    return False

a = np.vectorize(a_func)

num_samples = 1001
DOF = 2 * smallest_hole_diameter**2 / target_wavelength # distance from photon sieve to source
source_distance = focal_length + DOF
sampling_interval = target_wavelength * source_distance / sieve_diameter
fxx = np.arange(-(num_samples - 1) / 2, (num_samples - 1) / 2) / (num_samples * sampling_interval)
fyy = np.arange(-(num_samples - 1) / 2, (num_samples - 1) / 2) / (num_samples * sampling_interval)

fx, fy = np.meshgrid(fxx, fyy)


G = (
    (target_wavelength * source_distance)**4 *
    a(target_wavelength * source_distance * fx, target_wavelength * source_distance * fy) *
    np.e**(1j * np.pi * sampling_interval * source_distance**2 * (fx**2 + fy**2))
)

g = np.fft.ifft2(G)

plt.imshow(
    np.abs(g)
)

plt.show()
