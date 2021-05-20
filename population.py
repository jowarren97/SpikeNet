import numpy as np
from parameters import *
from connections import * 
from inputs import *  
from node import Node

class Population(Node):
    def __init__(self, pars, name = 'pop'):
        super().__init__(name, pars.n_neurons)
        self.pars = pars
        #pop parameters
        self.leak = pars.leak
        self.noise = pars.noise
        self.regL1 = pars.regL1
        self.regL2 = pars.regL2
        #timeseries
        self.Vt = np.zeros((self.n_neurons, 1))
        self.Vm = np.zeros((self.n_neurons, 1))
        self.rate = np.zeros((self.n_neurons, 1))
        self.output = np.array([])
        self.refrac_counter = np.zeros((self.n_neurons, 1))
        self.refrac_mask = np.ones((self.n_neurons, 1))
        #catalogue of connections
        self.fastConnections = dict()
        self.slowConnections = dict()
        self.outputConnections = dict()
        #book keeping
        self.step = 0

    def initialise(self, steps, timestep = None):
        #Initialise timeseries arrays of data (membrane V, spiketrains, threshold voltage, rates, output)
        self.Vm = np.zeros((self.n_neurons, steps))
        self.spiketrains = np.zeros((self.n_neurons, steps))
        # self.Vt = np.zeros((self.n_neurons, steps))
        self.rate = np.zeros((self.n_neurons, steps))
        if 'output' in self.fastConnections:
            outputdim = self.output.shape[0]
            self.output = np.zeros((outputdim, steps))
        else:
            print("Warning: no output has been added to the population")

        #Initialise initial value for threshold voltage
        r = self.fastConnections['input'].weights
        for i in range(0, self.n_neurons):
            # self.Vt[i,0] = 0.5 * (np.dot(r[:,i], r[:,i].T) + self.regL1*self.leak + self.regL2*self.leak**2)
            self.Vt[i] = 0.5 * (np.dot(r[:,i], r[:,i].T) + self.regL1*self.leak + self.regL2*self.leak**2)
        #Add recurrent connections that implement L1 & L2 regularisation on firing rates (Boerlin 2013)
        if not self.pars.adaptiveThreshold and not self.pars.learning:
            self.addConnection(self.pars, node=self, weights= - self.regL2 * self.leak**2 * np.eye(self.n_neurons), connType='fast')

    def addConnection(self, pars, node, weights, connType = 'fast', delay = 0, learning_rule = None):
        if weights.shape != (node.n_neurons, self.n_neurons):
            raise ValueError("Passed array is not of the right shape")

        if learning_rule is not None:
            print("added plastic conn")
            proj = PlasticConnection(pars, node, self, weights, delay, learning_rule=learning_rule)
        else:
            proj = Connection(pars, node, self, weights, delay)

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

    def addOutput(self, weights, outputdim = 1):
        if weights.shape != (outputdim, self.n_neurons):
            raise ValueError("Passed array is not of the right shape")

        proj = Connection(None, None, None, weights)
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

    def propagate(self, step, timestep):
        #book keeping
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
        # if self.pars.adaptiveThreshold:
        #     self.Vt[:,[step]] = self.Vt[:,[step-1]] - timestep * self.leak * (self.Vt[:,[step-1]] - self.Vt[:,[0]]) + self.regL2 * self.leak**2 * self.spiketrains[:,[step-1]]
        # else:
        #     self.Vt[:,[step]] = self.Vt[:,[step-1]]

        self.refrac_counter = np.maximum(0, self.refrac_counter - timestep)

        if self.pars.oneSpikePerStep:
            # VaboveThresh = self.Vm[:,[step]] - self.Vt[:,[step]]
            VaboveThresh = self.Vm[:,[step]] - self.Vt
            # VaboveThresh = VaboveThresh * self.refrac_mask

            if np.amax(VaboveThresh) < 0:
                pass
            else: #neuron spiked
                refrac_mask = (self.refrac_counter <= 0).flatten() #neurons not currently in refractory period
                if refrac_mask.any() == True:
                    subset_idx = np.argmax(VaboveThresh[refrac_mask])
                    parent_idx = np.arange(VaboveThresh.shape[0])[refrac_mask][subset_idx]

                    self.spiketrains[parent_idx,[step]] = 1

        else: #BREAKS THE SIMULATION
            self.spiketrains[:,[step]] = np.greater(self.Vm[:,[step]], self.Vt[:,[step]])

        self.refrac_counter += self.spiketrains[:,[step]] * self.pars.refractory_period

        #UPDATE RATES
        self.rate[:,[step]] = (1 - self.leak * timestep) * self.rate[:,[step-1]] + self.spiketrains[:,[step]]

        #UPDATE OUTPUTS
        if 'output' in self.fastConnections:
            self.output[:,[step]] = (1 - self.leak * timestep) * self.output[:,[step-1]] + self.fastConnections['output'].weights @ self.spiketrains[:,[step]]

        #UPDATE WEIGHTS
        if self.pars.learning:
            self.updateWeights()       