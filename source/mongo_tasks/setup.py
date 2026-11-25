"""Installation script for the 'mongo_tasks' python package."""

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    # generic
    "psutil",
    "numpy>=1.26.0",
    "scipy>=1.11.0",  # collision hulls for the optional fixed-velocity evaluation
    "torch>=2.7.0",  # compatible with Isaac Sim 5.1 and Isaac Lab 2.3
    "torchvision>=0.22.0",  # compatible with torch 2.7.0
    # headless-safe CSV/PNG curves exported by scripts/rsl_rl/play.py
    "matplotlib>=3.8.0",
    # basic logger
    "tensorboard",
    "prettytable",
]

setup(
    name="mongo_tasks",
    packages=find_packages(),
    author="Mongo_RL",
    maintainer="Mongo_RL",
    url="",
    version="0.1.0",
    description="Mongo0909 reinforcement learning tasks",
    keywords="Mongo0909, Reinforcement Learning, Isaac Lab",
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10,<3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
