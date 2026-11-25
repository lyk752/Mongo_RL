"""List all Mongo0909 environments registered with gymnasium."""

"""Launch Isaac Sim Simulator first."""

import os
import sys

# make ``source/mongo_world`` / ``source/mongo_tasks`` importable without an
# editable install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mongo_bootstrap  # noqa: E402

mongo_bootstrap.ensure_source_packages(verbose=True)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import mongo_world  # noqa: F401
import mongo_tasks  # noqa: F401


def main():
    """Print all environments registered by the Mongo_RL extension."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Mongo_RL"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if task_spec.id.startswith("Mongo-"):
            table.add_row(
                [
                    index + 1,
                    task_spec.id,
                    task_spec.entry_point,
                    task_spec.kwargs["env_cfg_entry_point"],
                ]
            )
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
