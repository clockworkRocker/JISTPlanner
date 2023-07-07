import numpy as np


class Node(object):
    """
    Node of a exploring tree in the configuration space
    """

    def __init__(self, id, pose, vels=None):
        self.id = id
        self.pose = pose
        """Phase variables in given state"""
        self.vels = vels
        """Velocity controls in given state"""
        self.obstacle_factor_id = None
        self.target_factor_id = None

        self.neighbours = {}
        """
        Dictionary of lists where for each neighbour a list of corresponding connecting factor ID is stored
        """

    def add_neighbour(self, id):
        if id not in self.neighbours:
            self.neighbours[id] = []

    def remove_neighbour(self, id):
        if id in self.neighbours:
            del self.neighbours[id]

    def __str__(self):
        return str(self.id) + ": " + str(self.neighbours.keys())
