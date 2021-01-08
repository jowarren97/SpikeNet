import numpy as np
import matplotlib.pyplot as plt
import graphing
#from pyNN.utility.plotting import plot_spiketrains

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

    def recordWeights(self, t):
        self.weightHistory[0] += [t]
        self.weightHistory[1] += [self.weights]

    def update(self, step):
        Vm = self.target.Vm[:,[step]] #post-synaptic population
        spiketrains = self.source.spiketrains[:,[step]] #pre-synaptic population
        rate = self.target.rate[:,[step-1]] #post-synaptic rate
        regL2 = self.target.regL2

        idxSpiked = np.where(spiketrains==1)

        for idx in idxSpiked:
            dw = - 2*(Vm + regL2*rate) - self.weights[:,idx]
            dw[idx] -= regL2
            self.weights[:,idx] += self.lr*dw
# class SlowConnection(Connection):
#     def __init__(self, sourceNode, targetNode, weights, decay, precision = 0.0001, delay = 0):
#         super().__init__(sourceNode, targetNode, weights, delay)
#         self.decay = decay
#         self.precision = precision
#         self.psp = np.empty([])        

class Node():
    def __init__(self, name, n_neurons):
        self.n_neurons = n_neurons
        self.name = name

    #@abstractmethod
    def initialise(self, steps):
        pass

   # @abstractmethod
    def propagate(self, steps):
        pass

# class Input(Node):
#     def __init__(self, n_neurons):
#         super().__init__('input', n_neurons)

class SpikingInput(Node):
    def __init__(self, name, n_neurons):
        super().__init__('name', n_neurons)

class CurrentInput(Node):
    def __init__(self, name, n_neurons):
        super().__init__(name, n_neurons)
        self.I = np.array([])

    def initialise(self, steps, timestep):
        pass

    def parseAmplitudes(self, amplitudes):
        if amplitudes == None:
            amplitudes = np.array(self.n_neurons*[[1]])
        elif type(amplitudes) == list:
            amplitudes = np.asarray([amplitudes]).T
        elif type(amplitudes) != np.ndarray:
            raise TypeError("Amplitude argument is neither a list nor nd array")
        
        if amplitudes.shape != (self.n_neurons, 1):
            raise ValueError("Amplitude argument is not of the right shape")
        return amplitudes

class ConstantCurrentInput(CurrentInput):
    def __init__(self, n_neurons, amplitudes = None):
        super().__init__('input', n_neurons)
        self.I = self.parseAmplitudes(amplitudes)

    def initialise(self, steps, timestep):
        self.I = np.repeat(self.I, steps, 1)

class SinusoidalCurrentInput(CurrentInput):
    def __init__(self, n_neurons, amplitudes = None, angularVelocity = 1/20):
        super().__init__('input', n_neurons)   
        self.amplitudes = self.parseAmplitudes(amplitudes)
        self.omega = angularVelocity
        self.x = np.array([])
        self.xdot = np.array([])

    def initialise(self, steps, timestep):
        t = np.arange(0, steps) * timestep
        self.x = self.amplitudes * np.sin(2*np.pi * self.omega * t)
        self.xdot = self.amplitudes * 2*np.pi * self.omega * np.cos(2*np.pi * self.omega * t)
        self.I = self.x + 10*self.xdot #!!! 10 is 1/leak !!! IMPLEMENT BETTER

class GaussianCurrentInput(CurrentInput):
    def __init__(self, n_neurons, mean, covariance):
        super().__init__('input', n_neurons)   
        self.name = 'input'
        self.n_neurons = n_neurons
        self.mean = mean
        self.covariance = covariance

    def initialise(self):
        if self.covariance.shape != (self.n_neurons, self.n_neurons):
            raise ValueError("Covariance array is not of the right shape")
        if self.mean.shape != (self.n_neurons, 1):
            raise ValueError("Mean array is not of the right shape")

        #self.I = #WRITE CODE

class Population(Node):
    def __init__(self, name = 'pop', n_neurons = 1, leak = 0.1, noise = 0.1):
        super().__init__(name, n_neurons)
        self.leak = leak
        self.noise = noise
        self.regL1 = 0.0
        self.regL2 = 0.01
        self.adaptiveThreshold = False
        self.Vt = np.zeros((self.n_neurons, 1))
        self.Vm = np.zeros((self.n_neurons, 1))
        self.rate = np.zeros((self.n_neurons, 1))
        self.output = np.array([])
        self.fastConnections = dict()
        self.slowConnections = dict()
        self.outputConnections = dict()

    def initialise(self, steps, timestep = None):
        #Initialise timeseries arrays of data (membrane V, spiketrains, threshold voltage, rates, output)
        self.Vm = np.zeros((self.n_neurons, steps))
        self.spiketrains = np.zeros((self.n_neurons, steps))
        self.Vt = np.zeros((self.n_neurons, steps))
        self.rate = np.zeros((self.n_neurons, steps))
        if 'output' in self.fastConnections:
            outputdim = self.output.shape[0]
            self.output = np.zeros((outputdim, steps))
        else:
            print("Warning: no output has been added to the population")

        #Initialise initial value for threshold voltage
        r = self.fastConnections['input'].weights
        for i in range(0, self.n_neurons):
            self.Vt[i,0] = 0.5 * (np.dot(r[:,i], r[:,i].T) + self.regL1*self.leak + self.regL2*self.leak**2)

        #Add recurrent connections that implement L1 & L2 regularisation on firing rates (Boerlin 2013)
        if not self.adaptiveThreshold:
            self.addReccurence(weights = - self.regL2 * self.leak**2 * np.eye(self.n_neurons), connType='fast')
            #print((self.slowConnections['pop']).weights)
            #self.addReccurence(weights = self.leak * np.eye(self.n_neurons) @ (self.fastConnections['pop']).weights, connType='slow')
            #print((self.slowConnections['pop']).weights)

    def addConnection(self, node, weights, connType = 'fast', delay = 0, plastic = False, logWeights = False):
        if weights.shape != (node.n_neurons, self.n_neurons):
            raise ValueError("Passed array is not of the right shape")

        if plastic:
            print("added plastic conn")
            proj = PlasticConnection(node, self, weights, delay) #NEED TO PROVIDE INPUT FOR LR AND BETA
        else:
            proj = Connection(node, self, weights, delay)

        if connType == 'fast':
            if node.name in self.fastConnections: #check if already existing connection
                self.fastConnections[node.name] += proj
                print("Warning: adding provided weight matrix to existing connection.")
            else:
                self.fastConnections[node.name] = proj
        elif connType == 'slow':
            if node.name in self.slowConnections: #check if already existing connection
                self.slowConnections[node.name] += proj
                print("Warning: adding provided weight matrix to existing connection.")
            else:
                self.slowConnections[node.name] = proj
        else:
            print("Unsuccessfully addition of connection from", node.name, "node to", self.name, "node. Please specify connection type either 'fast' or 'slow'.")
            return
        
        print("Successfully added connection from", node.name, "node to", self.name, "node.")

    def addReccurence(self, weights, connType = 'fast', delay = 0, plastic = False, logWeights = False):
        self.addConnection(self, weights, connType, delay, plastic, logWeights)

    def addOutput(self, weights, outputdim = 1):
        if weights.shape != (outputdim, self.n_neurons):
            raise ValueError("Passed array is not of the right shape")

        proj = Connection(None, None, weights)
        self.fastConnections['output'] = proj
        self.output = np.zeros((outputdim, 1))
        print("Successfully added output from", self.name)

    def updateWeights(self, step):
        for _, proj in self.fastConnections.items():
            if isinstance(proj, PlasticConnection):
                proj.update(step)
        for _, proj in self.slowConnections.items():
            if isinstance(proj, PlasticConnection):
                proj.update(step)

    def logWeights(self, step):
        for _, proj in self.fastConnections.items():
            if isinstance(proj, PlasticConnection) and proj.logWeights:
                proj.recordWeights(step)
        for _, proj in self.slowConnections.items():
            if isinstance(proj, PlasticConnection) and proj.logWeights:
                proj.recordWeights(step)

    def propagate(self, step, timestep, oneSpikePerStep):
        #LEAK MEMBRANE VOLTAGE
        self.Vm[:,[step]] = self.Vm[:,[step-1]] - timestep * self.leak * (self.Vm[:,[step-1]] + np.random.normal(0, self.noise * self.Vt[:,[0]], (self.n_neurons,1)))

        #PROCESS FAST CURRENTS
        for _, proj in self.fastConnections.items():
            node = proj.source
            weight = proj.weights
            delay = proj.delay
            if isinstance(node, CurrentInput):
                self.Vm[:,[step]] += timestep * self.leak * weight.T @ node.I[:,[step]]
            elif type(node) == Population:
                self.Vm[:,[step]] += weight @ node.spiketrains[:,[step-1]]

        #PROCESS SLOW CURRENTS
        for _, proj in self.slowConnections.items():
            node = proj.source
            weight = proj.weights
            delay = proj.delay
            self.Vm[:,[step]] += (1/self.leak) * weight @ node.rate[:,[step-1]]        

        #ADAPTIVE THRESHOLD
        if self.adaptiveThreshold:
            self.Vt[:,[step]] = self.Vt[:,[step-1]] - timestep * self.leak * (self.Vt[:,[step-1]] - self.Vt[:,[0]]) + self.regL2 * self.leak**2 * self.spiketrains[:,[step-1]]
        else:
            self.Vt[:,[step]] = self.Vt[:,[step-1]]

        if oneSpikePerStep:
            VaboveThresh = self.Vm[:,[step]] - self.Vt[:,[step]]
            if np.amax(VaboveThresh) < 0:
                pass
            else: #neuron spiked
                idx = np.argmax(VaboveThresh)
                self.spiketrains[idx,[step]] = 1

        #GETS STUCK IN INFINITE OSCILLATORY LOOP
        # if oneSpikePerStep:
        #     VaboveThresh = self.Vm[:,[step]] - self.Vt[:,[step]]

        #     while np.amax(VaboveThresh) > 0:
        #         idx = np.argmax(VaboveThresh)
        #         print("idx: ", idx, "\tVaboveThresh[idx]: ", VaboveThresh[idx])
        #         self.spiketrains[idx,[step]] = 1

        #         for _, proj in self.fastConnections.items():
        #             if proj.source == self:
        #                 print(VaboveThresh)
        #                 VaboveThresh += proj.weights[:,[idx]]
        #                 print(VaboveThresh)

        else:
            self.spiketrains[:,[step]] = np.greater(self.Vm[:,[step]], self.Vt)

        #UPDATE RATES
        self.rate[:,[step]] = (1 - self.leak * timestep) * self.rate[:,[step-1]] + self.spiketrains[:,[step]]

        #UPDATE OUTPUTS
        if 'output' in self.fastConnections:
            self.output[:,[step]] = self.output[:,[step-1]] + self.fastConnections['output'].weights @ self.spiketrains[:,[step]] + timestep * (-self.leak * self.output[:,[step-1]])

        #UPDATE WEIGHTS
        self.updateWeights(step)       

        #LOG WEIGHTS
        if step % 100:
            self.logWeights(step)



class Simulation:
    def __init__(self):
        self.populations = []
        self.inputs = []
        self.step = 1
        self.timestep = 0.1
        self.duration = 10
        self.steps = int(self.duration/self.timestep)
        self.oneSpikePerStep = True

    def addPopulations(self, populations):
        self.populations += populations

    def addInputs(self, inputs):
        self.inputs += inputs

    def setDuration(self, duration):
        self.duration = duration
        self.steps = int(duration/self.timestep)

    def setTimestep(self, timestep):
        self.timestep = timestep
        self.steps = int(self.duration/timestep)

    def initialise(self):
        for inp in self.inputs:
            inp.initialise(self.steps, self.timestep)

        for pop in self.populations:
            pop.initialise(self.steps)

    def propagate(self):
        for pop, inp in zip(self.populations, self.inputs):
            pop.propagate(self.step, self.timestep, self.oneSpikePerStep)
            #inp.propogate(self.step, self.timestep)
        self.step += 1

    def run(self, duration):
        self.setDuration(duration)
        self.initialise()

        while self.step < self.steps:
            self.propagate()

T = 4000
N = 100
pop = Population(name = 'pop', n_neurons = N)
inp = SinusoidalCurrentInput(n_neurons = 1, amplitudes = [2.0], angularVelocity=1/50)

r = np.array(int(N/2)*[[0.1]]+int(N/2)*[[-0.1]]).T
pop.addConnection(node = inp, weights = r)

w = - r.T @ r
#pop.addReccurence(weights = w)
w_rand = - 0.001*np.random.rand(N,N) - 0.005*np.eye(N,N)
pop.addReccurence(weights = w_rand, plastic=True)
pop.addOutput(r, 1)

sim = Simulation()
sim.addPopulations([pop])
sim.addInputs([inp])

sim.run(duration = T)

print("done simulation")

t = np.arange(0, T, 0.1)
#plt.figure()
#plt.plot(pop.fastConnections[pop.name].weightHistory[0,:], pop.fastConnections[pop.name].weightHistory[1])
fig = plt.figure()

ax = fig.add_subplot(321)
graphing.plotOutputInput(inp.x, pop.output, t, ax)

ax2 = fig.add_subplot(323)
for v in pop.Vm[:1]:
    ax2.set_xlim(0, t[-1])
    ax2.plot(t, v)
    ax2.set_xlabel("time /ms")
    ax2.set_ylabel("Vm")

for T in pop.Vt:
    ax2.plot(t, T, '--')

ax3 = fig.add_subplot(325)
graphing.plotSpiketrains(pop, ax3, t)

ax4 = fig.add_subplot(122)
graphing.plotISI(pop, ax=ax4)

plt.tight_layout()
plt.show()
