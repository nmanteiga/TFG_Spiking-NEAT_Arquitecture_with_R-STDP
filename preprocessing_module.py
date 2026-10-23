import cv2
import numpy as np

HUD_HEIGHT = 32         # score / coins / world / time text at the top of the screen
GRID = 13               # 13x13 = 169 binary inputs
CELL_THRESHOLD = 0.2    # a cell spikes if at least 20% of its pixels are not background


def preprocess_state(state):
    '''
    binary "something is here" encoding: a cell spikes when it contains objects (ground, pipes, enemies, mario...),
    and stays silent when it is only background. The background colour is the most common colour of the frame,
    so it works for both the blue sky of 1-1 and the black background of underground levels like 1-2
    '''
    play_area = state[HUD_HEIGHT:].astype(np.int32)
    colours = (play_area[..., 0] << 16) | (play_area[..., 1] << 8) | play_area[..., 2]

    # most common colour on a subsample of the screen (cheap) = background
    values, counts = np.unique(colours[::8, ::8], return_counts=True)
    background = values[np.argmax(counts)]

    foreground = (colours != background).astype(np.float32)
    cells = cv2.resize(foreground, (GRID, GRID), interpolation=cv2.INTER_AREA).flatten()
    return np.where(cells > CELL_THRESHOLD, 1.0, 0.0)
