import math

import numpy as np

def scale(s):
    if isinstance(s, (int, float)):
        s = [s, s]
    return np.array([
        [s[0], 0,    0],
        [0,    s[1], 0],
        [0,    0,    1],
    ])

def translate(s):
    if isinstance(s, (int, float)):
        s = [s, s]
    return np.array([
        [1, 0, s[0]],
        [0, 1, s[1]],
        [0, 0,    1],
    ])

def rotation(rad):
    return np.array([
        [math.cos(rad), -math.sin(rad), 0],
        [math.sin(rad),  math.cos(rad), 0],
        [            0,              0, 1],
    ])

def scale_square(m: np.ndarray):
    orig_scale = m.diagonal()[:2]
    s = np.sqrt(np.prod(orig_scale))
    return scale(s / orig_scale)
