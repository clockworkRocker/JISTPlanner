from enum import Enum
from typing import Optional, Tuple

import numpy as np
import gpmp2 as gp
import gtsam as gs


class DOFType(Enum):
    LINEAR = 0
    ANGULAR = 1


class DOF:
    """
    Degree of freedom description
    """

    __slots__ = ["name", "type", "limits"]

    def __init__(
        self,
        name: str,
        type: DOFType,
        limits: Optional[Tuple[float]] = None,
    ):
        self.name = name
        self.type = type
        self.limits = limits


# ============================================================================ #


class Sphere:
    """
    Sphere description
    """

    __slots__ = ["center", "radius"]

    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = radius

    def x(self):
        return self.center[0]

    def y(self):
        return self.center[1]


# ============================================================================ #


def default_point_robot_geometry(radius: float, num_dof: int) -> gp.PointRobotModel:
    sphere_origin = gs.Point3(0.0, 0.0, 0.0)
    bodies = gp.BodySphereVector()
    bodies.push_back(gp.BodySphere(0, radius, sphere_origin))

    return gp.PointRobotModel(gp.PointRobot(num_dof, 1), bodies)


# ============================================================================ #


def point_robot_geometry(spheres: list[Sphere], num_dof: int) -> gp.PointRobotModel:
    bodies = gp.BodySphereVector()

    for sphere in spheres:
        origin = gs.Point3(sphere.x(), sphere.y(), 0.0)
        bodies.push_back(gp.BodySphere(0, sphere.radius, origin))

    return gp.PointRobotModel(gp.PointRobot(num_dof, 1), bodies)


def default_point_robot(
    radius: float,
    linvel_limit: float,
    angvel_limit: float,
    search_limit: float,
    **kwargs
) -> dict:
    DEFAULT_NUM_DOF = 3

    num_dof = kwargs.get("num_dof", DEFAULT_NUM_DOF)

    return {
        "dof": [
            DOF("x", DOFType.LINEAR, (-search_limit, search_limit)),
            DOF("y", DOFType.LINEAR, (-search_limit, search_limit)),
            DOF("yaw", DOFType.ANGULAR, (-2 * np.pi, 2 * np.pi)),
        ],
        "num_dof": num_dof,
        "dof_limits": [
            (-search_limit, search_limit),
            (-search_limit, search_limit),
            (-np.pi, np.pi),
        ],
        "num_controls": num_dof,
        "control_limits": np.asarray([linvel_limit, linvel_limit, angvel_limit]),
        "model": default_point_robot_geometry(radius, num_dof),
        "dynamics_factor": gp.VehicleDynamicsFactorVector,
        "movement_factor": gp.GaussianProcessPriorLinear,
        "obstacle_factor": gp.ObstaclePlanarSDFFactorPointRobot,
        "avoidance_factor": gp.ObstaclePlanarSDFFactorGPPointRobot,
        "avg_vel": 0.1,
    }


def make_point_robot(
    geometry,
    linvel_limit: float,
    angvel_limit: float,
    search_limit: float,
) -> dict:
    return {
        "dof": [
            DOF("x", DOFType.LINEAR, (-search_limit, search_limit)),
            DOF("y", DOFType.LINEAR, (-search_limit, search_limit)),
            DOF("yaw", DOFType.ANGULAR, (-2 * np.pi, 2 * np.pi)),
        ],
        "num_dof": geometry.dof(),
        "dof_limits": [
            (-search_limit, search_limit),
            (-search_limit, search_limit),
            (-np.pi, np.pi),
        ],
        "num_controls": geometry.dof(),
        "control_limits": np.asarray([linvel_limit, linvel_limit, angvel_limit]),
        "model": geometry,
        "prior_factor": gs.PriorFactorVector,
        "dynamics_factor": gp.VehicleDynamicsFactorVector,
        "movement_factor": gp.GaussianProcessPriorLinear,
        "obstacle_factor": gp.ObstaclePlanarSDFFactorPointRobot,
        "avoidance_factor": gp.ObstaclePlanarSDFFactorGPPointRobot,
        "avg_vel": 0.1,
    }

# ============================================================================ #

def mobile_base_geometry(spheres: list[Sphere]) -> gp.Pose2MobileBaseModel:
    bodies = gp.BodySphereVector()

    for sphere in spheres:
        origin = gs.Point3(sphere.x(), sphere.y(), 0.0)
        bodies.push_back(gp.BodySphere(0, sphere.radius, origin))

    return gp.Pose2MobileBaseModel(gp.Pose2MobileBase(), bodies)

def make_mobile_base(geometry,
    linvel_limit: float,
    angvel_limit: float,
    search_limit: float):
        return {
        "dof": [
            DOF("x", DOFType.LINEAR, (-search_limit, search_limit)),
            DOF("y", DOFType.LINEAR, (-search_limit, search_limit)),
            DOF("yaw", DOFType.ANGULAR, (-2 * np.pi, 2 * np.pi)),
        ],
        "num_dof": geometry.dof(),
        "dof_limits": [
            (-search_limit, search_limit),
            (-search_limit, search_limit),
            (-np.pi, np.pi),
        ],
        "num_controls": geometry.dof(),
        "control_limits": np.asarray([linvel_limit, linvel_limit, angvel_limit]),
        "model": geometry,
        "pose_prior_factor": gs.PriorFactorPose2,
        "vels_prior_factor": gs.PriorFactorVector,
        "dynamics_factor": gp.VehicleDynamicsFactorPose2,
        "movement_factor": gp.GaussianProcessPriorPose2,
        "obstacle_factor": gp.ObstaclePlanarSDFFactorPose2MobileBase,
        "avoidance_factor": gp.ObstaclePlanarSDFFactorGPPose2MobileBase,
        "avg_vel": 0.1,
    }
