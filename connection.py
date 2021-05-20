import numpy as np
from learning_rules import *

class Connection:
    def __init__(self, pars, source_node, target_node, weights, delay = 0, learning_rule = None):
        self.source = source_node
        self.target = target_node
        self.weights = weights
        self.delay = delay
        self.pars = pars
        self.learningRule = learning_rule
        self.eligibility_trace = np.zeros(weights.shape)

    def __add__(self, other):
        if isinstance(self, Connection) != isinstance(other, Connection):
            raise TypeError("Cannot add an object of type", type(other).__name__, "to an object of type", type(self).__name__)

        if (self.source, self.target, self.delay, self.learningRule) != (other.source, other.target, other.delay, other.learningRule):
            raise ValueError("Parameters of connections do not allow addition.")

        new_weights = self.weights + other.weights
        new_connection = self
        new_connection.weights = new_weights
        return new_connection

    def update(self):
        if self.learningRule is not None:
            self.learningRule(self, self.source, self.target)  