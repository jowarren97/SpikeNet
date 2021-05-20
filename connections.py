import numpy as np
from learning_rules import *

class Connection:
    def __init__(self, pars, source_node, target_node, weights, delay = 0):
        self.source = source_node
        self.target = target_node
        self.weights = weights
        self.delay = delay

    def __add__(self, other):
        if isinstance(self, Connection) != isinstance(other, Connection):
            raise TypeError("Cannot add an object of type", type(other).__name__, "to an object of type", type(self).__name__)
        elif type(self) != type(other):
            print("Warning: adding a connection of type", type(other).__name__, "to a connection of type", type(self).__name__, ". Type", type(self).__name__, "was assumed.")

        if (self.source, self.target, self.delay) != (other.source, other.target, other.delay):
            raise ValueError("Parameters of connections do not allow addition.")

        new_weights = self.weights + other.weights
        new_connection = self
        new_connection.weights = new_weights
        return new_connection

class PlasticConnection(Connection):
    def __init__(self, pars, source_node, target_node, weights, delay = 0, learning_rule = None):
        super().__init__(pars, source_node, target_node, weights, delay)
        self.pars = pars
        self.learningRule = learning_rule

        self.eligibility_trace = np.zeros(weights.shape)

    # def recordWeights(self):
    #     self.weightHistory[0] += [self.target.step]
    #     self.weightHistory[1] += [self.weights]

    def update(self):
        self.learningRule(self, self.source, self.target)  