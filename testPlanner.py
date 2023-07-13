import numpy as np
import gpmp2 as gp
import gtsam as gs
from modules.debug import PlottingPlanner
from modules.datasets.generate2Ddataset import Dynamic2Ddataset

NUM_DOF = 3
NUM_LINKS = 1
ROBOT_RADIUS = 0.4


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
        "dof_limits": [(-8, 8), (-8, 8), (0, 2 * np.pi)],
        "num_controls": 3,
        "control_limits": np.asarray([3.0, 3.0, 0.6]),
        "model": make_robot(),
        "dynamics_factor": gp.VehicleDynamicsFactorVector,
        "movement_factor": gp.GaussianProcessPriorLinear,
        "obstacle_factor": gp.ObstaclePlanarSDFFactorPointRobot,
        "avg_vel": 0.1
    }

    configs = {
        "sdf_side": 8.,
        "sdf_step": 8. / 160.,
        "time_step": 0.1,
        "step_multiplier": 0.2,

        "epsilon_dist": 0.1,
        "sigma_goal": 2.,
        "sigma_goal_costco": 4.
    }

    planner = PlottingPlanner(robot, **configs)
    start = np.asarray([0.9, 0.9, 0.0])
    target = np.asarray([2.4, 2.5, -3 * np.pi / 4])
    
    dataset = Dynamic2Ddataset(160, 160, cell_size=configs["sdf_step"])
    dataset.init_obstacles(878923, 100)
    map = dataset.get_dataset(start, [configs["sdf_side"], configs["sdf_side"]]).map

    for i in range(map.shape[1]):
        map[0, i] = 1
        map[i, 0] = 1
        map[-1, i] = 1
        map[i, -1] = 1

    path, result = planner.plan(
        start,
        target,
        np.zeros(NUM_DOF),
        dataset.get_dataset(start, [configs["sdf_side"], configs["sdf_side"]]).map,
        1,
        32,
    )
    while (np.linalg.norm(path[-1] - target) > 0.25):
        path, result = planner.plan(
        path[-1],
        target,
        np.zeros(NUM_DOF),
        dataset.get_dataset(start, [configs["sdf_side"], configs["sdf_side"]]).map,
        1,
        32,
    )

    print("Controls:", result)


if __name__ == "__main__":
    main()
