import numpy as np
from skimage.draw import line
from skimage.transform import rescale
from scipy.ndimage import rotate
from tqdm import tqdm
from mas.forward_model import crop

def video(*, scene, resolution_ratio, frame_rate, exp_time, drift_angle,
          drift_velocity, angle_velocity, pixel_size, ccd_size, start,):

    """
    Get strand video frames

    Args:
        scene (ndarray): high resolution input scene
        resolution_ratio (float): downsample factor to low resolution images
        frame_rate (float): video frame rate
        exp_time (float): experiment duration
        drift_angle (float): linear drift direction (deg)
        drift_velocity (float): linear drift velocity (pix / s)
        angle_velocity (float): camera rotation rate (deg / s)
        pixel_size (float):
        ccd_size (int): size of square detector ccd (pixels)
        start (tuple): start location of detector in scene
    """

    num_frames = exp_time * frame_rate

    def coord(k):
        return np.array((
            start[0] - k * drift_velocity * np.sin(np.deg2rad(drift_angle)) *
            resolution_ratio / (frame_rate * pixel_size),
            start[1] + k * drift_velocity * np.cos(np.deg2rad(drift_angle)) *
            resolution_ratio / (frame_rate * pixel_size)
        )).astype(int).T

    # FIXME check box bounds correctly
    assert (
        0 <= coord(0)[0] < scene.shape[0] and 0 <= coord(0)[1] < scene.shape[1] and
        0 <= coord(num_frames)[0] < scene.shape[0] and 0 <= coord(num_frames)[1] < scene.shape[1]
    ), f"Frames drift outside of scene bounds ({x_coord(0)}, {y_coord(0)}) -> ({x_coord(num_frames)}, {y_coord(num_frames)})"

    # calculate the middle points for all frames
    mid = coord(np.arange(num_frames + 1))

    # initialize frame images
    frames = np.zeros((num_frames, ccd_size[0], ccd_size[1]))

    # calculate each frame by integrating high resolution image along the drift
    # direction
    for frame in tqdm(range(num_frames), desc='Frames', leave=None, position=1):
        hr_size = np.array(ccd_size) * resolution_ratio
        hr_frame = np.zeros(hr_size)
        # calculate middle coordinates for the shortest line connecting the
        # middle coordinates of the consecutive frames
        path_rows, path_cols = line(
            mid[frame][0],
            mid[frame][1],
            mid[frame+1][0],
            mid[frame+1][1]
        )
        total_rotation = exp_time * angle_velocity
        angles = total_rotation * np.sqrt((path_rows - mid[0][0])**2 + (path_cols - mid[0][1])**2) / np.linalg.norm(mid[-1] - mid[0])
        if len(path_rows) > 1:
            path_rows, path_cols = path_rows[:-1], path_cols[:-1]
        for row, col, angle in zip(path_rows, path_cols, angles):
            # accelerate algorithm by not rotating if angle_velocity is 0
            if angle_velocity == 0:
                slice_x = slice(row - hr_size[0] // 2, row + (hr_size[0] + 1) // 2)
                slice_y = slice(col - hr_size[1] // 2, col + (hr_size[1] + 1) // 2)
                hr_frame += scene[slice_x, slice_y]
            else:
                # diameter of circumscribing circle
                circum_diam = int(np.ceil(np.linalg.norm(hr_size)))
                slice_x = slice(row - circum_diam // 2, row + (circum_diam + 1) // 2)
                slice_y = slice(row - circum_diam // 2, row + (circum_diam + 1) // 2)
                unrotated = scene[slice_x, slice_y]
                hr_frame += crop(rotate(unrotated, angle, reshape='largest'), width=hr_size)
        # scale collected energy of subframes
        hr_frame /= frame_rate * len(path_rows)
        frames[frame] = rescale(hr_frame, 1 / resolution_ratio, anti_aliasing=False)

    return frames, mid
