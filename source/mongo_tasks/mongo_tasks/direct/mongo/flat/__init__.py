"""Flat-ground Mongo0909 task registration.

Two gymnasium ids are registered, mirroring the wheelbipe convention:

* ``Mongo-Flat-v0`` -- training
* ``Mongo-Flat-Play-v0`` -- play / deployment checks (no done reset)

Both ids point at the same entry point and configuration.  ``play.py`` forces
play mode itself, so ``Mongo-Flat-v0`` works for training *and* for play; the
``-Play-`` id is only there for people who like the explicit name.
"""

import gymnasium as gym

from .. import agents  # noqa: F401

_ENTRY_POINT = f"{__name__}.env:MongoFlatEnv"


def _task_kwargs() -> dict:
    return {
        "env_cfg_entry_point": f"{__name__}.env_cfg:MongoFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MongoFlatPPORunnerCfg",
    }


for _task_id in ("Mongo-Flat-v0", "Mongo-Flat-Play-v0"):
    gym.register(
        id=_task_id,
        entry_point=_ENTRY_POINT,
        disable_env_checker=True,
        kwargs=_task_kwargs(),
    )
