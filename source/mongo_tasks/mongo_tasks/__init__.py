"""Package containing task implementations for the Mongo0909 environments."""

import os

# Importing the task module triggers the gymnasium registration of the tasks.
import mongo_tasks.direct.mongo.flat  # noqa: F401

# Parent path of the package: ``<repo>/source/mongo_tasks/mongo_tasks``
RootPath = os.path.dirname(os.path.abspath(__file__))
