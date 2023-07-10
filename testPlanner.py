import numpy as np
import gpmp2 as gp
import gtsam as gs
from modules.debug import PlottingPlanner
from modules.datasets.generate2Ddataset import Dynamic2Ddataset

NUM_DOF = 3
NUM_LINKS = 1
ROBOT_RADIUS = 1.5


def make_robot():
    sphere_origin = gs.Point3(0.0, 0.0, 0.0)
    bodies = gp.BodySphereVector()
    bodies.push_back(gp.BodySphere(0, ROBOT_RADIUS, sphere_origin))

    return gp.PointRobotModel(gp.PointRobot(NUM_DOF, NUM_LINKS), bodies)


def print_graph(nodes):
    print("Graph:")
    for key in nodes:
        print(key, nodes[key].pose, nodes[key].vels, nodes[key].neighbours.keys())


def main():
    robot = {
        "num_dof": NUM_DOF,
        "dof_limits": [(0, 100), (0, 100), (0, 2 * np.pi)],
        "num_controls": NUM_DOF,
        "control_limits": np.asarray([3.0, 3.0, 0.6]),
        "model": make_robot(),
        "dynamics_factor": gp.VehicleDynamicsFactorVector,
        "movement_factor": gp.GaussianProcessPriorLinear,
        "obstacle_factor": gp.ObstaclePlanarSDFFactorPointRobot,
        "avg_vel": 0.3,
    }

    dataset = Dynamic2Ddataset()
    planner = PlottingPlanner(robot, sdf_side=15)
    start = np.asarray([45.0, 80.0, 0.0])
    target = np.asarray([45.0, 10.0, np.pi / 2])

    path, result = planner.plan(
        start,
        target,
        np.zeros(NUM_DOF),
        np.zeros((100, 100)),
        1,
        50,
    )

    print("Path:", path)
    print("Controls:", result)


if __name__ == "__main__":
    main()
