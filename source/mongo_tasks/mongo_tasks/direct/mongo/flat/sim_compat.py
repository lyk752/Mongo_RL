"""Optional PhysX settings across Isaac Lab releases."""

import inspect
import warnings


def external_force_iteration_kwargs(physx_cfg_type):
    """Enable per-iteration external forces only when the API supports it."""
    name = "enable_external_forces_every_iteration"
    if name in inspect.signature(physx_cfg_type).parameters:
        return {name: True}
    warnings.warn(
        "[Mongo_RL] This Isaac Lab PhysxCfg does not support "
        "enable_external_forces_every_iteration; using the installed version's "
        "external-force behavior. TGS and solver iteration counts are preserved, "
        "but physics may differ from the reference training environment. "
        "Use a matching Isaac Lab version for reproducible training.",
        RuntimeWarning,
        stacklevel=2,
    )
    return {}
