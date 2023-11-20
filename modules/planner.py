from queue import Queue
from random import Random
from copy import deepcopy
import numpy as np

import gtsam as gs
import gpmp2 as gp

from JISTPlanner.modules.utils.signedDistanceField2D import signedDistanceField2D
from JISTPlanner.modules.node import Node


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
        sigma_diff_control = 1e-4
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
        """
        Parameters
        ----------
        robot_model: A dictionary that describes the robot. The following keys must be present:
            * num_dof - Number of degrees of freedom
            * num_controls - Usually you want this equal to num_dof
            * control_limits - A NumPy ndarray with dimension num_dof
            * model - Object of class derivative from GPMP2::ForwardKinematics
            * dynamics_factor - A GPMP2 factor that includes movement constraints of the robot (for example, differential steering)
            * obstacle_factor - A GPMP2 factor that uses the obstacle data
            * movement_factor - A GPMP2 factor that estimates the cost of moving between two given states
        """
        self.robot = robot_model
        self.pose_dim = robot_model["num_dof"]
        self.control_dim = robot_model["num_controls"]

        # Distance field parameters
        self.sdf_side = kwargs.pop("sdf_side", JISTPlanner.Defaults.sdf_side)
        self.sdf_step = kwargs.pop("sdf_step", JISTPlanner.Defaults.sdf_step)

        # Params for factors
        self.cost_sigma = kwargs.pop("cost_sigma", JISTPlanner.Defaults.cost_sigma)
        self.epsilon_dist = kwargs.pop(
            "epsilon_dist", JISTPlanner.Defaults.epsilon_dist
        )
        self.sigma_goal = kwargs.pop("sigma_goal", JISTPlanner.Defaults.sigma_goal)

        self.sigma_goal_costco = kwargs.pop(
            "sigma_goal_costco", JISTPlanner.Defaults.sigma_goal_costco
        )
        self.sigma_start = kwargs.pop("sigma_start", JISTPlanner.Defaults.sigma_start)
        self.sigma_vel_limit = kwargs.pop(
            "sigma_vel_limit", JISTPlanner.Defaults.sigma_vel_limit
        )
        self.sigma_diff_control = kwargs.pop(
            "sigma_diff_control", JISTPlanner.Defaults.sigma_diff_control
        )
        self.use_trustregion_opt = kwargs.pop(
            "use_trustregion_opt", JISTPlanner.Defaults.use_trustregion_opt
        )

        # RRT params
        self.node_budget = kwargs.pop("node_budget", JISTPlanner.Defaults.node_budget)
        self.step_multiplier = kwargs.pop(
            "step_multiplier", JISTPlanner.Defaults.step_multiplier
        )

        # Target criteria
        self.target_region_radius = kwargs.pop(
            "target_region_radius", JISTPlanner.Defaults.target_region_radius
        )

        # General parameters
        self.time_step = kwargs.pop("time_step", JISTPlanner.Defaults.time_step)
        self.reuse_previous_graph = kwargs.pop(
            "reuse_previous_graph", JISTPlanner.Defaults.reuse_previous_graph
        )

        self.random_seed = kwargs.pop("seed", None)
        self.random = Random(self.random_seed)

        # Graphs
        self.factors = None
        self.nodes = {}
        self.values = None
        self.sdf = None

        # IDs
        self.new_node_id = 0
        self.current_node_id = 0

        # Models for factors
        self.start_pose_cost_model = gs.noiseModel.Isotropic.Sigma(
            self.robot["num_dof"], self.sigma_start
        )
        self.pose_cost_model = gs.noiseModel.Isotropic.Sigma(
            self.robot["num_dof"], self.sigma_goal_costco
        )
        self.start_vels_cost_model = gs.noiseModel.Isotropic.Sigma(
            self.robot["num_controls"], self.sigma_start
        )
        self.vels_cost_model = gs.noiseModel.Isotropic.Sigma(
            self.robot["num_controls"], self.sigma_goal_costco
        )
        self.vels_limit_cost_model = gs.noiseModel.Isotropic.Sigma(
            self.robot["num_controls"], self.sigma_vel_limit
        )
        self.qc_model = gs.noiseModel.Gaussian.Covariance(
            np.identity(self.robot["num_dof"])
        )

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
    def _make_sdf(self, grid, cell_size, origin):
        field = signedDistanceField2D(grid, cell_size)
        origin_point = gs.Point2(
            origin[0] - self.sdf_side / 2, origin[1] - self.sdf_side / 2
        )

        self.sdf = gp.PlanarSDF(origin_point, cell_size, field)

    def _make_graph(self, start):
        if len(self.nodes) == 0:
            n_id = self.__get_new_id()
            new_node = Node(
                n_id, start, np.ones(self.robot["num_controls"]) * self.robot["avg_vel"]
            )
            self.nodes = {n_id: new_node}
            self.current_node_id = n_id
        self._grow_graph()

    def _make_values(self):
        self.values = gs.Values()
        for id in self.nodes:
            pose_key = gs.symbol("x", id)
            vels_key = gs.symbol("v", id)
            self.values.insert(pose_key, self.nodes[id].pose)
            self.values.insert(vels_key, self.nodes[id].vels)

    # ---------------- Algorithm steps ----------------
    def _grow_graph(self):
        """Use RRT to sample new nodes"""

        point = np.zeros(self.robot["num_dof"])
        pose = self.nodes[self.current_node_id].pose
        for _ in range(len(self.nodes), self.node_budget):
            closest_dist = np.inf
            closest_id = -1
            for i, limit in enumerate(self.robot["dof_limits"]):
                point[i] = pose[i] + np.random.uniform(limit[0], limit[1])

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

    def _build_factors(self, start, target, target_vels):
        self.factors = gs.NonlinearFactorGraph()

        for id in self.nodes:
            pose_key = gs.symbol("x", id)
            vels_key = gs.symbol("v", id)

            # Start state factors
            if id == self.current_node_id:
                self.factors.push_back(
                    gs.PriorFactorVector(
                        pose_key, self.nodes[id].pose, self.start_pose_cost_model
                    )
                )
                self.factors.push_back(
                    gs.PriorFactorVector(
                        vels_key, self.nodes[id].vels, self.start_vels_cost_model
                    )
                )
            # Differential control constraint
            self.factors.push_back(
                self.robot["dynamics_factor"](
                    pose_key, vels_key, self.sigma_diff_control
                )
            )
            # Velocity limits
            self.factors.push_back(
                gp.VelocityLimitFactorVector(
                    vels_key,
                    self.vels_limit_cost_model,
                    self.robot["control_limits"],
                    np.ones(self.robot["num_controls"]),
                )
            )
            if id != self.current_node_id:
                # Distance to target
                self.factors.push_back(
                    gs.PriorFactorVector(pose_key, target, self.pose_cost_model)
                )
                self.nodes[id].target_factor_id = self.factors.size() - 1

                # Difference from target velocity
                self.factors.push_back(
                    gs.PriorFactorVector(vels_key, target_vels, self.vels_cost_model)
                )

                # Distance to obstacles
                self.factors.push_back(
                    self.robot["obstacle_factor"](
                        pose_key,
                        self.robot["model"],
                        self.sdf,
                        self.cost_sigma,
                        self.epsilon_dist,
                    )
                )
                self.nodes[id].obstacle_factor_id = self.factors.size() - 1

            # Cost to neighbours
            for n_id in self.nodes[id].neighbours:
                n_pose_key = gs.symbol("x", n_id)
                n_vels_key = gs.symbol("v", n_id)
                self.factors.push_back(
                    self.robot["movement_factor"](
                        pose_key,
                        vels_key,
                        n_pose_key,
                        n_vels_key,
                        self.time_step,
                        self.qc_model,
                    )
                )
                self.nodes[id].neighbours[n_id].append(self.factors.size() - 1)

    def _optimize_graph(self):
        # Prepare initial values
        self.values = gs.Values()
        for id in self.nodes:
            pose_key = gs.symbol("x", id)
            vels_key = gs.symbol("v", id)
            self.values.insert(pose_key, self.nodes[id].pose)
            self.values.insert(vels_key, self.nodes[id].vels)

        # Prepare optimizer
        if self.use_trustregion_opt:
            params = gs.DoglegParams()
            optimizer = gs.DoglegOptimizer(self.factors, self.values, params)
        else:
            params = gs.GaussNewtonParams()
            optimizer = gs.GaussNewtonOptimizer(self.factors, self.values, params)

        self.values = optimizer.optimize()

        # Update the nodes
        for id in self.nodes:
            self.nodes[id].pose = self.values.atVector(gs.symbol("x", id))
            self.nodes[id].vels = self.values.atVector(gs.symbol("v", id))

    def _prune_graph(self):
        old_nodes = deepcopy(self.nodes)
        self.nodes.clear()

        stack = [self.current_node_id]
        visits = []
        while len(stack) > 0:
            id = stack.pop()
            self.nodes[id] = deepcopy(old_nodes[id])
            for n_id in self.nodes[id].neighbours:
                if n_id not in visits:
                    stack.append(n_id)

            visits.append(id)

    def _next_best_node(self, target):
        parent_ids = {}
        leafs = {}
        costs = {}
        depths = {}
        visited = {}

        nodes = Queue()
        nodes.put(self.current_node_id)
        costs[self.current_node_id] = 0
        depths[self.current_node_id] = 1
        visited[self.current_node_id] = True

        min_cost = np.inf
        best_leaf = -1

        # Find costs of each leaf
        while not nodes.empty():
            id = nodes.get()
            if len(self.nodes[id].neighbours) == 0:
                costs[id] /= depths[id] ** 2
                costs[id] += (np.linalg.norm(self.nodes[id].pose - target)) ** 2 / 2
                leafs[id] = costs[id]

            for n_id in self.nodes[id].neighbours:
                if n_id in visited:
                    continue
                costs[n_id] = costs[id] + self.__get_edge_cost(id, n_id)
                nodes.put(n_id)
                visited[n_id] = True
                parent_ids[n_id] = id
                depths[n_id] = depths[id] + 1

        # Find best leaf
        for id in leafs:
            if leafs[id] < min_cost:
                min_cost = leafs[id]
                best_leaf = id

        # Retrace the path to the current node
        cur_id = best_leaf
        while parent_ids[cur_id] != self.current_node_id:
            cur_id = parent_ids[cur_id]

        return cur_id

    def _update_goal_models(self, start, target):
        # Why the #&*! does it get formatted like that???
        tempsigma = self.sigma_goal_costco * (
            1
            - (
                1
                / (
                    0.2 * np.linalg.norm(target - self.nodes[self.current_node_id].pose)
                    + 1
                )
            )
        )
        self.pose_cost_model = gs.noiseModel.Isotropic.Sigma(
            self.robot["num_dof"], tempsigma
        )
        self.vels_cost_model = gs.noiseModel.Isotropic.Sigma(
            self.robot["num_controls"], tempsigma
        )

    # ---------------- Interface ----------------
    def plan(
        self,
        start: np.ndarray,
        target: np.ndarray,
        target_vels: np.ndarray,
        grid: np.ndarray,
        num_steps: int,
    ):
        """
        Parameters
        ----------
            start: Starting position (dimension should be equal to the number of DOF)
            target: Target position
            target_vels: The desired velocities in the target position
            grid: The occupancy grid (a.k.a. A matrix that represents the obstacle data)
            num_steps: The number of steps to plan for
        """
        step = 0
        controls = []
        """List of control velocities """

        self._make_sdf(grid, self.sdf_step, start)
        self._make_graph(start)

        while step < num_steps:
            self._build_factors(start, target, target_vels)
            self._optimize_graph()
            next_node = self._next_best_node(target)
            controls.append(self.nodes[next_node].vels)

            self.current_node_id = next_node
            self._prune_graph()
            self._grow_graph()
            self._update_goal_models(start, target)

            step += 1

        return controls

    def plan_with_path(self, start, target, target_vels, grid, grid_grain, num_steps):
        step = 0
        controls = []
        path = []
        """List of control velocities """

        self._make_sdf(grid, grid_grain, start)
        self._make_graph(start)

        while step < num_steps:
            self._build_factors(start, target, target_vels)
            self._optimize_graph()
            next_node = self._next_best_node(target)
            controls.append(self.nodes[next_node].vels)
            path.append(self.nodes[self.current_node_id].pose)

            self.current_node_id = next_node
            self._prune_graph()
            self._grow_graph()
            self._update_goal_models(start, target)

            step += 1

        return path, controls
