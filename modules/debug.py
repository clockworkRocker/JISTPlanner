from planner import JISTPlanner
import gtsam as gs
import gpmp2 as gp
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import utils.plot_utils as pu
from utils.signedDistanceField2D import signedDistanceField2D


class PlottingPlanner(JISTPlanner):
    def __init__(self, robot_model, **kwargs):
        super(PlottingPlanner, self).__init__(robot_model, **kwargs)
        self.pretty_field = None

    def _make_sdf(self, grid, cell_size, origin):
        self.pretty_field = signedDistanceField2D(grid, cell_size)
        origin_point = gs.Point2(
            origin[0] - self.sdf_side / 2, origin[1] - self.sdf_side / 2
        )

        self.sdf = gp.PlanarSDF(origin_point, cell_size, self.pretty_field)

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

    def __plot_occupancy(self, axis, start):
        pose = start
        axis.imshow(
            self.pretty_field,
            extent=[
                pose[0] - self.sdf_side / 2,
                pose[0] + self.sdf_side / 2,
                pose[1] - self.sdf_side / 2,
                pose[1] + self.sdf_side / 2,
            ],
            alpha=0.3,
        )

    def __plot(self, figure, axis, start, target, path, step):
        axis.cla()

        # ---------------- Plottting params ----------------
        axis.set_title("JIST: {:5.2f} sec".format(self.time_step * step))
        axis.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        axis.tick_params(
            axis="y",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        # Turn off tick labels
        axis.set_yticklabels([])
        axis.set_xticklabels([])
        # --------------------------------------------------

        self.__plot_graph(axis)
        pu.plotPointRobot2D_theta(
            figure, axis, self.robot["model"], self.nodes[self.current_node_id].pose
        )
        self.__plot_occupancy(axis, start)
        self.__plot_path(path, axis)
        axis.plot(start[0], start[1], "gx", markersize=10)
        axis.plot(target[0], target[1], "rx", markersize=10)
        figure.show()
        plt.pause(self.time_step)

    def plan(self, start, target, target_vels, grid, grid_grain, num_steps):
        step = 0
        controls = []
        path = [start]
        """List of control velocities """

        self._make_sdf(grid, grid_grain, start)
        self._make_graph(start)
        self.current_node_id = 0

        figure = plt.figure(0, dpi=300)
        axis = figure.gca()
        self.__plot(figure, axis, start, target, path, step)

        while step < num_steps:
            print(
                "Step:", step, 
                "Position:", self.nodes[self.current_node_id].pose, 
                "Velocity: ", self.nodes[self.current_node_id].vels)
            self._build_factors(start, target, target_vels)
            self._make_values()
            self._optimize_graph()
            self.__plot(figure, axis, start, target, path, step)

            next_node = self._next_best_node(target)
            
            controls.append(self.nodes[next_node].vels)
            path.append(self.nodes[next_node].pose)

            self.current_node_id = next_node
            self._prune_graph()
            self._grow_graph()
            self._update_goal_models(start, target)

            self.__plot(figure, axis, start, target, path, step)

            step += 1

        return path, controls
