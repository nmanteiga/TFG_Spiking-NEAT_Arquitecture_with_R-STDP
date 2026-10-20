import cv2
import numpy as np


def preprocess_state(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (13, 13))
    flattened = resized.flatten() / 255.0
    return np.where(flattened > 0.5, 1.0, 0.0)
