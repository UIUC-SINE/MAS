import numpy as np
from skimage.draw import line
from skimage.transform import rescale

def video(
        *,
        scene,
        resolution_ratio=5,
        frame_rate=4,
        exp_time=10,
        drift_angle=np.deg2rad(-45),
        drift_velocity=0.1e-3,
        pixel_size=14e-6,
        ccd_size=(160, 160),
        start=(400, 0),
        ):
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

    # calculate the topleft points for all frames
    topleft_coords = []
    for k in range(num_frames + 1):
        topleft_coords.append(
            (
                int(
                    start[0] -
                    k * drift_velocity * np.sin(drift_angle) * resolution_ratio /
                    (frame_rate * pixel_size)
                ),
                int(
                    start[1] +
                    k * drift_velocity * np.cos(drift_angle) * resolution_ratio /
                    (frame_rate * pixel_size)
                )
            )
        )

    # initialize frame images
    frames = np.zeros((num_frames, ccd_size[0], ccd_size[1]))

    # calculate each frame by integrating high resolution image along the drift
    # direction
    for frame in range(num_frames):
        print('Frame {}/{}\r'.format(frame, num_frames), end='')
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
