class Connection:
    def __init__(self, sourceNode, targetNode, weights, delay = 0):
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
    def __init__(self, sourceNode, targetNode, weights, delay = 0, lr = 0.01, beta = 2, logWeights = False):
        super().__init__(sourceNode, targetNode, weights, delay)
        self.logWeights = logWeights
        self.lr = lr
        self.beta = beta
        self.weightHistory = [[],[]]

    def recordWeights(self):
        self.weightHistory[0] += [self.target.step]
        self.weightHistory[1] += [self.weights]

    def update(self):
        step = self.target.step
        Vm = self.target.Vm[:,[step]] #post-synaptic population
        spiketrains = self.source.spiketrains[:,[step]] #pre-synaptic population
        rate = self.target.rate[:,[step-1]] #post-synaptic rate
        regL2 = self.target.regL2

        idxSpiked = np.where(spiketrains==1)

        for idx in idxSpiked:
            dw = - 2*(Vm + regL2*rate) - self.weights[:,idx]
            dw[idx] -= regL2
            self.weights[:,idx] += self.lr*dw     