from typing import Optional

import numpy as np
import gpmp2 as gp
import gtsam as gs

from jist.debug import PlottingPlanner

# from modules.planner import JISTPlanner
from jist.utils.robot import make_mobile_base, mobile_base_geometry, Sphere
from jist.datasets.dataset2D import make_random

NUM_DOF = 3

def print_graph(nodes):
    print("Graph:")
    for key in nodes:
        print(key, nodes[key].pose, nodes[key].vels, nodes[key].neighbours.keys())


def make_obstacle_map(
    rows: int,
    cols: int,
    side: float,
    obst_size: float,
    obst_num: int = 5,
    use_border: bool = False,
    seed: Optional[int] = None,
):
    map = np.zeros((rows, cols))
    obst_cols = int(np.ceil(obst_size / side * cols))

    if use_border:
        map[0, :] = 1
        map[:, 0] = 1
        map[-1, :] = 1
        map[:, -1] = 1

    rng = np.random.RandomState(seed)

    for _ in range(obst_num):
        origin_row = int(rng.uniform(0, rows))
        origin_col = int(rng.uniform(0, cols))
        other_row = int(min(rows, origin_row + obst_cols))
        other_col = int(min(cols, origin_col + obst_cols))

        map[origin_row:other_row, origin_col:other_col] = 1

    return map


def main():
    spheres = [
        Sphere(np.array([0.3, 0]), 0.6),
        Sphere(np.array([-0.3, 0]), 0.6),
    ]
    geometry = mobile_base_geometry(spheres)
    robot = make_mobile_base(geometry, 1.5, 0.6, 1)

    configs = {
        "sdf_side": 8.0,
        "sdf_step": 8.0 / 160.0,
        "node_budget": 160,
        "time_step": 0.05,
        "step_multiplier": 0.1,
        "epsilon_dist": 0.5,
        "cost_sigma": 0.1,
        "sigma_diff_control": 1e-3,
        "sigma_vel_limit": 1e-3,
        "sigma_goal_costco": 2,
        "num_path_interpolations": 3,
        "target_region_radius": 0.3,
        "avg_vel": 0.001,
    }

    planner = PlottingPlanner(robot, **configs)
    start = np.asarray([0.75, 0.75, -3 * np.pi / 4])
    target = np.asarray([7.1, 7.1, 0])

    # map = make_obstacle_map(
    #     160, 160, configs["sdf_side"], 0.4, obst_num=9, use_border=False, seed=150
    # )
    dataset = make_random(12, 1., (0., 10.))

    path, result = planner.plan(
        start,
        target,
        np.zeros(NUM_DOF),
        dataset.map(8., 8., 0.05, start),
        32,
    )
    print("Controls:", result)

    while np.linalg.norm(path[-1] - target) > configs["target_region_radius"]:
        path, result = planner.plan(
            path[-1],
            target,
            np.zeros(NUM_DOF),
            dataset.map(8., 8., 0.05, path[-1]),
            32,
        )
        print("Controls:", result)


if __name__ == "__main__":
    main()
