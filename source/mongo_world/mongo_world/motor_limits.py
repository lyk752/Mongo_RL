"""Backend-independent directional motor speed protection (output-shaft SI).

This is an engineering speed guard, not an identified torque-speed curve.
Only propulsive effort tapers; commanded braking keeps the full effort limit.
External forces can still drive a joint beyond its nominal no-load speed.
"""


def speed_limited_effort(effort, velocity, effort_limit, velocity_limit, taper_start_fraction, backend):
    """Clip effort using actual motor speed; supports NumPy and torch arrays.

    Full drive is available below ``fraction * limit``. Drive falls linearly
    to zero at the limit. Beyond the limit, only braking/zero effort is allowed.
    The caller validates positive limits and a fraction strictly in [0, 1).
    """
    width = velocity_limit * (1.0 - taper_start_fraction)
    positive = effort_limit * backend.clip((velocity_limit - velocity) / width, 0.0, 1.0)
    negative = effort_limit * backend.clip((velocity_limit + velocity) / width, 0.0, 1.0)
    return backend.minimum(backend.maximum(effort, -negative), positive)
