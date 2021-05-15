import numpy as np
from learning_rules import *

class Connection:
    def __init__(self, pars, sourceNode, targetNode, weights, delay = 0):
        self.source = sourceNode
        self.target = targetNode
        self.weights = weights
        self.delay = delay

    def __add__(self, other):
        if isinstance(self, Connection) != isinstance(other, Connection):
            raise TypeError("Cannot add an object of type", type(other).__name__, "to an object of type", type(self).__name__)
        elif type(self) != type(other):
            print("Warning: adding a connection of type", type(other).__name__, "to a connection of type", type(self).__name__, ". Type", type(self).__name__, "was assumed.")

        if (self.source, self.target, self.delay) != (other.source, other.target, other.delay):
            raise ValueError("Parameters of connections do not allow addition.")

        newWeights = self.weights + other.weights
        newConnection = self
        newConnection.weights = newWeights
        return newConnection

class PlasticConnection(Connection):
    def __init__(self, pars, sourceNode, targetNode, weights, delay = 0):
        super().__init__(pars, sourceNode, targetNode, weights, delay)
        self.pars = pars
        if pars.learning_rule == 'brendel2020':
            self.learningRule = brendel2020rec
        elif pars.learning_rule == 'eligibility_trace':
            self.learningRule = eligibilityTrace
        else:
            raise ValueError('Incorrect learning rule parameter')

        self.eligibilityTrace = np.zeros(weights.shape)

    # def recordWeights(self):
    #     self.weightHistory[0] += [self.target.step]
    #     self.weightHistory[1] += [self.weights]

    def update(self):
        self.learningRule(self, self.source, self.target)  