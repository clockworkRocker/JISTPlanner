import numpy as np
import gtsam as gs
import gpmp2 as gp
from cv2 import flip
import matplotlib.pyplot as plt
import matplotlib.collections as mc


from .planner import JISTPlanner
from .utils import plot_utils as pu
from .utils.signedDistanceField2D import signedDistanceField2D


class PlottingPlanner(JISTPlanner):
    def __init__(self, robot_model, **kwargs):
        super(PlottingPlanner, self).__init__(robot_model, **kwargs)
        self.pretty_field = None

    def _make_sdf(self, grid, cell_size, center):
        self.pretty_field = signedDistanceField2D(grid, cell_size)
        self.sdf_origin = gs.Point2(
            center[0] - self.sdf_side / 2, center[1] - self.sdf_side / 2
        )

        self.sdf = gp.PlanarSDF(self.sdf_origin, cell_size, self.pretty_field)

    def __plot_graph(self, axis):
        lines = []
        for id in self.nodes:
            start = self.nodes[id].pose[:2]
            for n_id in self.nodes[id].neighbours:
                end = self.nodes[n_id].pose[:2]
                lines.append([start, end])

        collection = mc.LineCollection(lines, colors=(0, 0, 0, 1.0), linewidth=0.5)
        axis.add_collection(collection)

        return collection

    def __plot_path(self, path, axis):
        xdata = []
        ydata = []
        for point in path:
            xdata.append(point[0])
            ydata.append(point[1])
        (whatever,) = axis.plot(xdata, ydata, "-g", linewidth=2)

        return whatever

    def __plot_map(self, axis):
        y, x = np.where(self.map == 1)
        x = self.sdf_origin[0] + x.astype(float) * self.sdf_step
        y = self.sdf_origin[1] + y.astype(float) * self.sdf_step
        axis.scatter(x, y, marker="s")

    def __plot_sdf(self, axis, start):
        axis.imshow(
            flip(self.pretty_field, flipCode=0),
            extent=[
                self.sdf_origin[0],
                self.sdf_origin[0] + self.sdf_side,
                self.sdf_origin[1],
                self.sdf_origin[1] + self.sdf_side,
            ],
            alpha=0.5,
        )

    def __plot(self, figure, axis, start, target, path, step):
        origin = path[-1] - self.sdf_side / 2

        axis.cla()
        axis.set_xlim([origin[0], origin[0] + self.sdf_side])
        axis.set_ylim([origin[1], origin[1] + self.sdf_side])

        # ---------------- Plottting params ----------------
        axis.set_title("JIST: {:5.2f} sec".format(self.time_step * step))
        axis.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
        )

        axis.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelbottom=False,
        )
        # Turn off tick labels
        axis.set_yticklabels([])
        axis.set_xticklabels([])
        # --------------------------------------------------

        self.__plot_graph(axis)
        # pu.plotPointRobot2D_theta(
        #     figure, axis, self.robot["model"], self.nodes[self.current_node_id].pose
        # )
        pu.plotRobotModel2D(
            figure, axis, self.robot["model"], self.nodes[self.current_node_id].pose
        )
        self.__plot_sdf(axis, start)
        self.__plot_map(axis)
        self.__plot_path(path, axis)
        axis.plot(start[0], start[1], "gx", markersize=10)
        axis.plot(target[0], target[1], "rx", markersize=10)
        figure.show()
        plt.pause(self.time_step)

    def plan(self, start, target, target_vels, grid, num_steps):
        step = 0
        controls = []
        path = [start]
        """List of control velocities """

        self.map = grid
        self._make_sdf(grid, self.sdf_step, start)

        if len(self.nodes) == 0 or not self.reuse_previous_graph:
            self._make_graph(start)

        figure = plt.figure(0, dpi=300)
        axis = figure.gca()
        self.__plot(figure, axis, start, target, path, step)

        while (
            np.linalg.norm(target - self.nodes[self.current_node_id].pose)
            > self.target_region_radius
            and step < num_steps
        ):
            x, y, w = self.nodes[self.current_node_id].pose
            vx, vy, vw = self.nodes[self.current_node_id].vels

            print(
                "Step:",
                step,
                "Position:",
                self.nodes[self.current_node_id].pose,
                "Velocity:",
                self.nodes[self.current_node_id].vels,
            )
            self._build_factors(start, target, target_vels)
            self._make_values()
            self._optimize_graph()

            next_node = self._next_best_node(target)

            controls.append(self.nodes[next_node].vels)
            path.append(self.nodes[next_node].pose)

            self.__plot(figure, axis, start, target, path, step)

            self.current_node_id = next_node
            self._prune_graph()
            self._grow_graph()
            self._update_goal_models(start, target)

            step += 1

        return path, controls
