"""NumPy implementation of flat/velocity_feedback.py's encoder contract."""
import numpy as np


def encoder_velocity(position, previous, dt):
    delta = np.asarray(position) - previous
    delta = delta.copy()
    wheel = delta[..., 5:]
    delta[..., 5:] = np.where(np.abs(wheel) > np.pi, (wheel + np.pi) % (2 * np.pi) - np.pi, wheel)
    return delta / dt
