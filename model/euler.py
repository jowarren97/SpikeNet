import numpy as np
import matplotlib.pyplot as plt
import graphing

class Node():
    def __init__(self, name, n_neurons):
        self.n_neurons = n_neurons
        self.name = name

    def initialise(self, steps):
        pass

    def propagate(self, steps):
        pass

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
        self.step = 1 #simulation counter

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

    def updateWeights(self):
        for _, proj in self.fastConnections.items():
            if isinstance(proj, PlasticConnection):
                proj.update()
        for _, proj in self.slowConnections.items():
            if isinstance(proj, PlasticConnection):
                proj.update()

    def logWeights(self):
        for _, proj in self.fastConnections.items():
            if isinstance(proj, PlasticConnection) and proj.logWeights:
                proj.recordWeights()
        for _, proj in self.slowConnections.items():
            if isinstance(proj, PlasticConnection) and proj.logWeights:
                proj.recordWeights()

    def propagate(self, step, timestep, oneSpikePerStep):
        self.step = step
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

        else: #BREAKS THE SIMULATION
            self.spiketrains[:,[step]] = np.greater(self.Vm[:,[step]], self.Vt[:,[step]])

        #UPDATE RATES
        self.rate[:,[step]] = (1 - self.leak * timestep) * self.rate[:,[step-1]] + self.spiketrains[:,[step]]

        #UPDATE OUTPUTS
        if 'output' in self.fastConnections:
            self.output[:,[step]] = self.output[:,[step-1]] + self.fastConnections['output'].weights @ self.spiketrains[:,[step]] + timestep * (-self.leak * self.output[:,[step-1]])

        #UPDATE WEIGHTS
        self.updateWeights()       

        #LOG WEIGHTS
        if step % 100:
            self.logWeights()


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

