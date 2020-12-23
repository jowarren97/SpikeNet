import numpy as np
import matplotlib.pyplot as plt
from graphing import plotISI
#from pyNN.utility.plotting import plot_spiketrains

def handle_options(ax, options):
    if "xticks" not in options or options.pop("xticks") is False:
        plt.setp(ax.get_xticklabels(), visible=False)
    if "xlabel" in options:
        ax.set_xlabel(options.pop("xlabel"))
    if "yticks" not in options or options.pop("yticks") is False:
        plt.setp(ax.get_yticklabels(), visible=False)
    if "ylabel" in options:
        ax.set_ylabel(options.pop("ylabel"))
    if "ylim" in options:
        ax.set_ylim(options.pop("ylim"))
    if "xlim" in options:
        ax.set_xlim(options.pop("xlim"))


    # if label:
    #     plt.text(0.95, 0.95, label,
    #              transform=ax.transAxes, ha='right', va='top',
    #              bbox=dict(facecolor='white', alpha=1.0))

class Connection:
    def __init__(self, sourceNode, targetNode, weights, delay = 0):
        self.source = sourceNode
        self.target = targetNode
        self.weights = weights
        self.delay = delay

class SlowConnection(Connection):
    def __init__(self, sourceNode, targetNode, weights, decay, delay = 0):
        super().__init__(sourceNode, targetNode, weights, delay)
        self.decay = decay

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
        print(max(self.x))
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
        self.regL1 = 0.1
        self.regL2 = 0.1
        self.plasticThreshold = True
        self.Vt = np.zeros((self.n_neurons, 1))
        self.Vm = np.zeros((self.n_neurons, 1))
        self.output = np.array([])
        self.connections = dict()
        self.outputConnections = dict()

    def initialise(self, steps, timestep = None):
        self.Vm = np.zeros((self.n_neurons, steps))
        self.spiketrains = np.zeros((self.n_neurons, steps))
        self.Vt = np.zeros((self.n_neurons, steps))

        if 'output' in self.connections:
            outputdim = self.output.shape[0]
            self.output = np.zeros((outputdim, steps))
        else:
            print("Warning: no output has been added to the population")

        r = self.connections['input'].weights
        for i in range(0, self.n_neurons):
            self.Vt[i,0] = 0.5 * (np.dot(r[:,i], r[:,i].T) + self.regL1*self.leak + self.regL2*self.leak**2)
        
        print(self.Vt[:,[0]])

    def addConnection(self, node, weights, delay = 0):
        if weights.shape != (node.n_neurons, self.n_neurons):
            raise ValueError("Passed array is not of the right shape")

        proj = Connection(node, self, weights, delay)
        self.connections[node.name] = proj
        print("Successfully added connection from", node.name, "node to", self.name, "node.")

    def addReccurence(self, weights, delay = 0):
        if weights.shape != (self.n_neurons, self.n_neurons):
            raise ValueError("Passed array is not of the right shape")

        proj = Connection(self, self, weights, delay)
        self.connections[self.name] = proj 
        print("Successfully added connection from", self.name, "node to", self.name, "node.")

    def addOutput(self, weights, outputdim = 1):
        if weights.shape != (outputdim, self.n_neurons):
            raise ValueError("Passed array is not of the right shape")

        proj = Connection(None, None, weights)
        self.connections['output'] = proj
        self.output = np.zeros((outputdim, 1))
        print("Successfully added output from", self.name)


    def propagate(self, step, timestep, oneSpikePerStep):
        #leak
        self.Vm[:,[step]] = self.Vm[:,[step-1]] - timestep * self.leak * (self.Vm[:,[step-1]] + np.random.normal(0, self.noise * self.Vt[:,[0]], (self.n_neurons,1)))

        for _, proj in self.connections.items():
            node = proj.source
            weight = proj.weights
            delay = proj.delay
            if isinstance(node, CurrentInput):
                self.Vm[:,[step]] += timestep * self.leak * weight.T @ node.I[:,[step]]
            elif type(node) == Population:
                self.Vm[:,[step]] += weight @ node.spiketrains[:,[step-1]]

        if self.plasticThreshold:
            self.Vt[:,[step]] = self.Vt[:,[step-1]] + timestep * self.leak * ( - (self.Vt[:,[step-1]] - self.Vt[:,[0]]) + self.regL2 * self.leak * self.spiketrains[:,[step-1]])
        else:
            self.Vt[:,[step]] = self.Vt[:,[step-1]]

        if oneSpikePerStep:
            VaboveThresh = self.Vm[:,[step]] - self.Vt[:,[step]]
            if np.amax(VaboveThresh) < 0:
                pass
            else:
                idx = np.argmax(VaboveThresh)
                self.spiketrains[idx,[step]] = 1

        #GETS STUCK IN INFINITE OSCILLATORY LOOP
        # if oneSpikePerStep:
        #     VaboveThresh = self.Vm[:,[step]] - self.Vt[:,[step]]

        #     while np.amax(VaboveThresh) > 0:
        #         idx = np.argmax(VaboveThresh)
        #         print("idx: ", idx, "\tVaboveThresh[idx]: ", VaboveThresh[idx])
        #         self.spiketrains[idx,[step]] = 1

        #         for _, proj in self.connections.items():
        #             if proj.source == self:
        #                 print(VaboveThresh)
        #                 VaboveThresh += proj.weights[:,[idx]]
        #                 print(VaboveThresh)

        else:
            self.spiketrains[:,[step]] = np.greater(self.Vm[:,[step]], self.Vt)

        if 'output' in self.connections:
            self.output[:,[step]] = self.output[:,[step-1]] + self.connections['output'].weights @ self.spiketrains[:,[step]] + timestep * (-self.leak * self.output[:,[step-1]])


class DataProcessor:
    def __init__(self):
        self.populations = []

    

class Simulation:
    def __init__(self):
        self.populations = []
        self.inputs = []
        self.step = 1
        self.timestep = 0.01
        self.duration = 10
        self.steps = int(self.duration/self.timestep)
        #self.data = DataProcessor()
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

T = 100

pop = Population(name = 'pop', n_neurons = 100)
inp = SinusoidalCurrentInput(n_neurons = 1, amplitudes = [10.0], angularVelocity=1/50)

r = np.array(50*[[0.1]]+50*[[-0.1]]).T
pop.addConnection(node = inp, weights = r)

w = r.T @ r
pop.addReccurence(weights = -w)
pop.addOutput(r, 1)

sim = Simulation()
sim.addPopulations([pop])
sim.addInputs([inp])

sim.run(duration = T)

print("done simulation")

# n_neurons = 10
# timestep = 0.1 #ms
# duration = 50
t = np.arange(0, T, 0.01)
# leak = 0.1
# input = np.array([[1.0]])

# steps = int(duration/timestep)

# Vm = np.zeros((n_neurons, steps))
# spike_trains = np.zeros((n_neurons, steps))
# V_thresh = np.zeros((n_neurons, 1))
# output = np.zeros((input.shape[0], steps))

# r = np.array(n_neurons*[[0.1]]).T
# w = np.matmul(r.T, r)
# print("decoding weights r = ", r, "\n")
# print("lateral weights w = ", w, "\n")

# for i in range(0, n_neurons):
#     V_thresh[i] = 0.5*np.dot(r[:,i], r[:,i].T)
# print("threshold T = ", V_thresh, "\n")


# step = 1
# while step < steps:

#     Vm[:,[step]] = Vm[:,[step-1]] - w @ spike_trains[:,[step-1]] + timestep * leak * (r.T @ input - Vm[:,[step-1]] + np.random.normal(0, 10*V_thresh, (n_neurons,1)))
    
#     spike_trains[:,[step]] = np.greater(Vm[:,[step]], V_thresh)

#     output[:,[step]] = output[:,[step-1]] + r @ spike_trains[:,[step]] + timestep * (-leak * output[:,[step-1]])

#     step += 1
#     print(step)


fig = plt.figure()

for signal in pop.output:  
    ax = fig.add_subplot(321)
    ax.set_xlim(0, t[-1])
    ax.plot(t, signal)
    ax.set_ylabel("value")

for v in pop.Vm[:1]:
    ax2 = fig.add_subplot(323)
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

plt.show()
