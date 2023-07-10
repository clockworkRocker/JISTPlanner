import numpy as np
import gtsam as gs
import gpmp2 as gp
from random import Random
from copy import deepcopy
from utils.signedDistanceField2D import signedDistanceField2D
from node import Node
from Queue import Queue


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
        self.sigma_goal_costco = (
            kwargs["sigma_goal_costco"]
            if "sigma_goal_costco" in kwargs
            else JISTPlanner.Defaults.sigma_goal_costco
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
        self.use_trustregion_opt = (
            kwargs["use_trustregion_opt"]
            if "use_trustregion_opt" in kwargs
            else JISTPlanner.Defaults.use_trustregion_opt
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

        # Graphs
        self.factors = None
        self.nodes = {}
        self.values = None
        self.sdf = None

        # IDs
        self.new_node_id = 0
        self.current_node_id = 0

        # Models for factors
        self.start_pose_cost_model = gs.noiseModel_Isotropic.Sigma(
            self.robot["num_dof"], self.sigma_start
        )
        self.pose_cost_model = gs.noiseModel_Isotropic.Sigma(
            self.robot["num_dof"], self.sigma_goal_costco
        )
        self.start_vels_cost_model = gs.noiseModel_Isotropic.Sigma(
            self.robot["num_controls"], self.sigma_start
        )
        self.vels_cost_model = gs.noiseModel_Isotropic.Sigma(
            self.robot["num_controls"], self.sigma_goal_costco
        )
        self.vels_limit_cost_model = gs.noiseModel_Isotropic.Sigma(
            self.robot["num_controls"], self.sigma_vel_limit
        )
        self.qc_model = gs.noiseModel_Gaussian.Covariance(
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
        n_id = self.__get_new_id()
        new_node = Node(
            n_id, start, np.ones(self.robot["num_controls"]) * self.robot["avg_vel"]
        )
        self.nodes = {n_id: new_node}
        self._grow_graph()

    # ---------------- Algorithm steps ----------------
    def _grow_graph(self):
        """Use RRT to sample new nodes"""

        closest_dist = np.inf
        closest_id = -1

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

    def _build_factors(self, start, target, target_vels):
        self.factors = gs.NonlinearFactorGraph()

        for id in self.nodes:
            pose_key = gs.symbol(ord("x"), id)
            vels_key = gs.symbol(ord("v"), id)

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
            # Robot dynamics
            self.factors.push_back(
                self.robot["dynamics_factor"](pose_key, vels_key, self.cost_sigma)
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
                n_pose_key = gs.symbol(ord("x"), n_id)
                n_vels_key = gs.symbol(ord("v"), n_id)
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
            pose_key = gs.symbol(ord("x"), id)
            vels_key = gs.symbol(ord("v"), id)
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
            self.nodes[id].pose = self.values.atVector(gs.symbol(ord("x"), id))
            self.nodes[id].vels = self.values.atVector(gs.symbol(ord("v"), id))

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
        self.sigma_goal_costco *= np.linalg.norm(
            target - self.nodes[self.current_node_id].pose
        ) / np.linalg.norm(target - start)
        self.pose_cost_model = gs.noiseModel_Isotropic.Sigma(
            self.robot["num_dof"], self.sigma_goal_costco
        )
        self.vels_cost_model = gs.noiseModel_Isotropic.Sigma(
            self.robot["num_controls"], self.sigma_goal_costco
        )

    # ---------------- Interface ----------------
    def plan(self, start, target, target_vels, grid, grid_grain, num_steps):
        step = 0
        controls = []
        """List of control velocities """

        self._make_sdf(grid, grid_grain, start)
        self._make_graph(start)
        self.current_node_id = 0

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
