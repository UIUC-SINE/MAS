import numpy as np
from skimage.draw import line
from skimage.transform import rescale
from tqdm import tqdm

def video(*, scene, resolution_ratio, frame_rate, exp_time, drift_angle,
        drift_velocity, pixel_size, ccd_size, start,):

    """
    Get strand video frames

    Args:
        resolution_ratio
        frame_rate
        exp_time
        drift_vector*
        scene*
        start*
    """

    num_frames = exp_time * frame_rate

    def x_coord(k):
        return int(
            start[0] - k * drift_velocity * np.sin(np.deg2rad(drift_angle)) *
            resolution_ratio / (frame_rate * pixel_size)
        )
    def y_coord(k):
        return int(
            start[1] + k * drift_velocity * np.cos(np.deg2rad(drift_angle)) *
            resolution_ratio / (frame_rate * pixel_size)
        )
    assert (
        0 <= x_coord(0) < scene.shape[0] and 0 <= y_coord(0) < scene.shape[1] and
        0 <= x_coord(num_frames) < scene.shape[0] and 0 <= y_coord(num_frames) < scene.shape[1]
    ), f"Frames drift outside of scene bounds ({x_coord(0)}, {y_coord(0)}) -> ({x_coord(num_frames)}, {y_coord(num_frames)})"

    # calculate the topleft points for all frames
    topleft_coords = []
    for k in range(num_frames + 1):
        topleft_coords.append((x_coord(k), y_coord(k)))

    # initialize frame images
    frames = np.zeros((num_frames, ccd_size[0], ccd_size[1]))

    # calculate each frame by integrating high resolution image along the drift
    # direction
    for frame in tqdm(range(num_frames), desc='Frames', leave=None, position=1):
        temp = np.zeros((ccd_size[0]*resolution_ratio, ccd_size[1]*resolution_ratio))
        # calculate topleft coordinates for the shortest line connecting the
        # topleft coordinates of the consecutive frames
        path_rows, path_cols = line(
            topleft_coords[frame][0],
            topleft_coords[frame][1],
            topleft_coords[frame+1][0],
            topleft_coords[frame+1][1]
        )
        if len(path_rows) > 1:
            path_rows, path_cols = path_rows[:-1], path_cols[:-1]
        for row,col in zip(path_rows, path_cols):
            temp += scene[row:row+temp.shape[0], col:col+temp.shape[1]]
        frames[frame] = rescale(temp, 1/resolution_ratio, anti_aliasing=False)
    return frames, topleft_coords
