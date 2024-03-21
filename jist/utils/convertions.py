import gtsam as gs
import numpy as np

def to_numpy(p: gs.Pose2):
    return np.array([p.x(), p.y(), p.theta()])

def to_pose2(p: np.ndarray):
    return gs.Pose2(p[0], p[1], p[2])