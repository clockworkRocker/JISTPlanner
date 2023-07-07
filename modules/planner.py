import numpy as np
import gtsam as gs
import gpmp2 as gp
from random import seed, Random
from copy import deepcopy
from utils.signedDistanceField2D import signedDistanceField2D
from node import Node


class JISTPlanner(object):
    """
    The class that describes the planner
    """

    class Defaults:
        """
        Default JIST Planner parameters
        """

        # Distance field parameters
        sdf_side = 20.0
        sdf_step = 0.001

        # Params for factors
        cost_sigma = 0.2
        epsilon_dist = 4.0
        sigma_goal = 2.0
        sigma_goal_costco = 4.0
        sigma_start = 0.0001
        sigma_vel_limit = 0.001
        use_trustregion_opt = False

        # RRT params
        node_budget = 64
        step_multiplier = 0.4

        # Target criteria
        target_region_radius = 0.5

        # General
        time_step = 0.01
        reuse_previous_graph = True

    def __init__(self, robot_model, **kwargs):
        self.robot = robot_model
        self.pose_dim = robot_model["num_dof"]
        self.control_dim = robot_model["num_controls"]

        # Distance field parameters
        self.sdf_side = (
            kwargs["sdf_side"]
            if "sdf_side" in kwargs
            else JISTPlanner.Defaults.sdf_side
        )
        self.sdf_step = (
            kwargs["sdf_step"]
            if "sdf_step" in kwargs
            else JISTPlanner.Defaults.sdf_step
        )

        # Params for factors
        self.cost_sigma = (
            kwargs["cost_sigma"]
            if "cost_sigma" in kwargs
            else JISTPlanner.Defaults.cost_sigma
        )
        self.epsilon_dist = (
            kwargs["epsilon_dist"]
            if "epsilon_dist" in kwargs
            else JISTPlanner.Defaults.epsilon_dist
        )
        self.sigma_goal = (
            kwargs["sigma_goal"]
            if "sigma_goal" in kwargs
            else JISTPlanner.Defaults.sigma_goal
        )
        self.sigma_start = (
            kwargs["sigma_start"]
            if "sigma_start" in kwargs
            else JISTPlanner.Defaults.sigma_start
        )
        self.sigma_vel_limit = (
            kwargs["sigma_vel_limit"]
            if "sigma_vel_limit" in kwargs
            else JISTPlanner.Defaults.sigma_vel_limit
        )

        # RRT params
        self.node_budget = (
            kwargs["node_budget"]
            if "node_budget" in kwargs
            else JISTPlanner.Defaults.node_budget
        )
        self.step_multiplier = (
            kwargs["step_multiplier"]
            if "step_multiplier" in kwargs
            else JISTPlanner.Defaults.step_multiplier
        )

        # Target criteria
        self.target_region_radius = (
            kwargs["target_region_radius"]
            if "target_region_radius" in kwargs
            else JISTPlanner.Defaults.target_region_radius
        )

        # General parameters
        self.time_step = (
            kwargs["time_step"]
            if "time_step" in kwargs
            else JISTPlanner.Defaults.time_step
        )
        self.reuse_previous_graph = (
            kwargs["reuse_previous_graph"]
            if "reuse_previous_graph" in kwargs
            else JISTPlanner.Defaults.reuse_previous_graph
        )

        self.random_seed = kwargs["seed"] if "seed" in kwargs else None
        self.random = Random(self.random_seed)

        self.factors = None
        self.nodes = {}
        self.values = None
        self.sdf = None

        self.new_node_id = 0
        self.current_node_id = 0

    # ---------------- Utility functions ----------------
    def __get_new_id(self):
        self.new_node_id += 1
        return self.new_node_id - 1

    def __get_edge_cost(self, id1, id2):
        cost = 0

        # Distance costs
        for factor_id in self.nodes[id1].neighbours[id2]:
            cost += self.factors.at(factor_id).error(self.values)

        # Obstacle cost
        cost += self.factors.at(self.nodes[id2].obstacle_factor_id).error(self.values)

        # Closeness to target cost
        cost += self.factors.at(self.nodes[id2].target_factor_id).error(self.values)

        return cost

    # ---------------- Initialization functions ----------------
    def __make_sdf(self, map):
        pass

    # ---------------- Algorithm steps ----------------
    def __grow_graph(self):
        """Use RRT to sample new nodes"""

        closest_dist = np.inf
        closest_id = None

        point = np.zeros(self.robot["num_dof"])
        for _ in range(len(self.nodes), self.node_budget):
            for i, limit in enumerate(self.robot["dof_limits"]):
                point[i] = np.random.uniform(limit[0], limit[1])

            for id in self.nodes:
                dist = np.linalg.norm(point - self.nodes[id].pose)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = id

            direction = point - self.nodes[closest_id].pose
            direction /= np.linalg.norm(direction)

            new_id = self.__get_new_id()
            self.nodes[new_id] = Node(
                new_id,
                self.nodes[closest_id].pose + direction * self.step_multiplier,
                np.ones(self.robot["num_controls"]) * self.robot["avg_vel"],
            )
            self.nodes[closest_id].add_neighbour(new_id)

    def __build_factors(self, start, target):
        self.factors = gs.NonlinearFactorGraph()

    def __optimize_graph(self):
        pass

    def __prune_graph(self):
        pass

    def __next_best_node(self):
        return -1

    # ---------------- Interface ----------------
    def plan(self, start, target, map, num_steps):
        step = 0
        controls = []
        """List of control velocities """

        while step < num_steps:
            self.__build_factors(start, target)
            self.__optimize_graph()
            next_node = self.__next_best_node()
            controls.append(self.nodes[next_node].vels)

            self.current_node_id = next_node
            self.__prune_graph()
            self.__grow_graph()

            step += 1

        return controls
