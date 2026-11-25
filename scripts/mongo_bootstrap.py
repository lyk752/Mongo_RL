"""Make the ``source`` packages importable without installing them.

The recommended workflow is still the editable install::

    python -m pip install -e source/mongo_world -e source/mongo_tasks

But running the scripts straight from a fresh checkout is common enough that
the scripts call :func:`ensure_source_packages` first.  It prepends this checkout to ``sys.path`` so scripts always use local code.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Package directories inside ``<repo>/source`` that must be importable.
_SOURCE_PACKAGES = ("mongo_world", "mongo_tasks")


def find_repo_root(start: str | None = None) -> str | None:
    """Walk upwards from ``start`` until a directory containing ``source`` is found."""
    current = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    for _ in range(8):
        if os.path.isdir(os.path.join(current, "source")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def _is_importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def ensure_source_packages(verbose: bool = False) -> list[str]:
    """Prepend this checkout's source packages to ``sys.path``.

    Returns the list of paths that were added.
    """
    repo_root = find_repo_root()
    if repo_root is None:
        return []

    added: list[str] = []
    for package in _SOURCE_PACKAGES:
        path = os.path.join(repo_root, "source", package)
        if not os.path.isdir(path):
            continue
        # only add when the package is not importable already
        if path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)

    if verbose and added:
        print(f"[Mongo_RL] Added source packages to sys.path: {added}")
    return added
