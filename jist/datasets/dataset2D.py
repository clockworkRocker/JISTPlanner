from typing import Optional
from dataclasses import dataclass

import numpy as np


class Dataset2D:
    __slots__ = ["obstacles"]

    # ======================================================================== #
    @dataclass
    class Rectangle:
        xbl: float
        ybl: float
        xtr: float
        ytr: float

        def width(self):
            return self.xtr - self.xbl

        def height(self):
            return self.ytr - self.ybl

    # ======================================================================== #

    def __init__(self) -> None:
        self.obstacles = []

    # ------------------------------------------------------------------------ #

    def add_rectangle(self, x: float, y: float, width: float, height: float):
        self.obstacles.append(self.Rectangle(x, y, x + width, y + height))

    # ------------------------------------------------------------------------ #

    def map(self, width: float, height: float, resolution: float, pose: np.ndarray):
        rows = int(height / resolution)
        cols = int(width / resolution)
        map = np.zeros((rows, cols))

        origin = pose[0:2] - np.array([width / 2, height / 2])
        top_right = pose[0:2] + np.array([width / 2, height / 2])

        for rect in self.obstacles:
            if (
                rect.xbl > top_right[0]
                or rect.ybl > top_right[1]
                or rect.xtr < origin[0]
                or rect.ytr < origin[1]
            ):
                continue

            row_bl = max(0, int((rect.ybl - origin[1]) / resolution))
            row_tr = min(rows, int((rect.ytr - origin[1]) / resolution))
            col_bl = max(0, int((rect.xbl - origin[0]) / resolution))
            col_tr = min(cols, int((rect.xtr - origin[0]) // resolution))

            rect_rows = row_tr - row_bl
            rect_cols = col_tr - col_bl

            map[row_bl:row_tr, col_bl:col_tr] = np.ones((rect_rows, rect_cols))

        return map

# ============================================================================ #


def make_random(obst_num: int, obst_size: float, bounds: tuple, seed: Optional[int] = None):
    dataset = Dataset2D()
    rng = np.random.RandomState(seed)

    for _ in range(obst_num):
        point = rng.uniform(bounds[0], bounds[1], (2, ))
        sizes = rng.uniform(0, obst_size, (2, ))
        dataset.add_rectangle(point[0], point[1], sizes[0], sizes[1])
    
    return dataset
