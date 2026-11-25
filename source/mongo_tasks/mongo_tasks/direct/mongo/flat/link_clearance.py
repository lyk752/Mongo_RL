"""Evaluation-only rear-link collision-surface clearance on a horizontal plane.

Read the configured USD once at startup. Per-step evaluation uses only Torch
and true link poses; it does not use COM positions or reconstruct passive joints.
"""
from __future__ import annotations

import numpy as np
import torch


REAR_LINK_BODY_NAMES = (
    "left_thigh", "right_thigh", "left_calf", "right_calf",
    "left_parallel_1", "right_parallel_1",
    "left_parallel_2", "right_parallel_2",
    "left_parallel_3", "right_parallel_3",
    "left_parallel_4", "right_parallel_4",
)


def load_collision_hull_vertices(usd_path: str, body_names: list[str]) -> list[np.ndarray]:
    """Return collision mesh hull vertices in each rigid body's local frame.

    Instance proxies, mesh scales and nested transforms are included. The
    source meshes bound the surface; PhysX cooking/contact offsets can differ.
    Unsupported/missing geometry is an error rather than a silent zero cost.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError(f"Cannot read Mongo collision geometry from {usd_path}.")
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    result = []
    for name in body_names:
        body = root.GetChild(name)
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Missing rigid body {name!r} in {usd_path}.")
        collision_root = body.GetChild("collisions")
        if not collision_root:
            raise ValueError(f"Missing collision geometry for {name!r} in {usd_path}.")
        vertices = []
        for prim in Usd.PrimRange(collision_root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            ancestor = prim
            collidable = False
            while ancestor != body:
                if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                    enabled = UsdPhysics.CollisionAPI(ancestor).GetCollisionEnabledAttr().Get()
                    collidable = enabled is not False
                    break
                ancestor = ancestor.GetParent()
            if not collidable:
                continue
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError(f"Unsupported clearance collider {prim.GetPath()}: expected Mesh.")
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            # Gf matrices multiply row vectors. ComputeRelativeTransform also
            # handles instance-proxy geometry beneath the collision Xform.
            transform, _ = cache.ComputeRelativeTransform(prim, body)
            matrix = np.asarray(transform)
            vertices.append(points @ matrix[:3, :3] + matrix[3, :3])
        if not vertices:
            raise ValueError(f"No enabled collision mesh for {name!r} in {usd_path}.")
        points = np.unique(np.concatenate(vertices), axis=0)
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite collision vertices for {name!r}.")
        # A linear world-z projection reaches its minimum on the convex hull,
        # so removing interior vertices preserves the exact mesh minimum.
        result.append(points[ConvexHull(points).vertices].astype(np.float32))
    return result


def pad_collision_vertices(vertices: list[np.ndarray], device: str) -> torch.Tensor:
    """Pack (body, point, xyz); repeating a vertex leaves the minimum intact."""
    max_points = max(len(points) for points in vertices)
    packed = np.empty((len(vertices), max_points, 3), dtype=np.float32)
    for i, points in enumerate(vertices):
        packed[i] = points[0]
        packed[i, :len(points)] = points
    return torch.as_tensor(packed, device=device)


def compute_link_clearance_m(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    local_vertices: torch.Tensor,
    ground_height_m: float = 0.0,
) -> torch.Tensor:
    """Minimum surface z per (env, body), given unit (w,x,y,z) link quaternions.

    Only the world-z row of each rotation is needed; avoid allocating a world
    xyz vector for every vertex. Horizontal translation and yaw do not affect
    clearance above the flat plane. Ground height is a world-z coordinate.
    """
    w, x, y, z = body_quat_w.unbind(dim=-1)
    world_z_in_link = torch.stack((
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x.square() + y.square()),
    ), dim=-1)
    surface_z = torch.einsum("nbc,bpc->nbp", world_z_in_link, local_vertices).amin(dim=-1)
    return surface_z + body_pos_w[..., 2] - ground_height_m
