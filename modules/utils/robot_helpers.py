import numpy as np
import gpmp2 as gp
import gtsam as gs

def make_point_robot(radius: float, num_dof: int) -> gp.PointRobotModel:
    sphere_origin = gs.Point3(0.0, 0.0, 0.0)
    bodies = gp.BodySphereVector()
    bodies.push_back(gp.BodySphere(0, radius, sphere_origin))

    return gp.PointRobotModel(gp.PointRobot(num_dof, 1), bodies)

def RobotDict(**kwargs) -> dict:
    DEFAULT_NUM_DOF = 3

    if "search_limit" not in kwargs:
        raise KeyError("Please provide a search limit for RRT (in meters)")
    if "radius" not in kwargs:
        raise KeyError("Please provide the radius of the robot")
    if "linvel_limit" not in kwargs:
        raise KeyError("Please provide a limit for robot linear velocity")
    if "angvel_limit" not in kwargs:
        raise KeyError("Please provide a limit for robot angular velocity")

    limit = kwargs["search_limit"]
    num_dof = DEFAULT_NUM_DOF if "num_dof" not in kwargs else kwargs["num_dof"]
    return {
        "num_dof": num_dof,
        "dof_limits": [(-limit, limit), (-limit, limit), (-np.pi, np.pi)],
        "num_controls": num_dof,
        "control_limits": np.asarray([
            kwargs["linvel_limit"],
            kwargs["linvel_limit"],
            kwargs["angvel_limit"]
        ]),
        "model": make_point_robot(
            kwargs["radius"], 
            num_dof
        ),
        "dynamics_factor": gp.VehicleDynamicsFactorVector,
        "movement_factor": gp.GaussianProcessPriorLinear,
        "obstacle_factor": gp.ObstaclePlanarSDFFactorPointRobot,
        "avg_vel": 0.1,
    }
