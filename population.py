import numpy as np
from parameters import *
from connection import Connection 
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
        #book keeping
        self.iter = 0

    def initialise(self, steps, timestep = None):
        #Initialise timeseries arrays of data (membrane V, spiketrains, threshold voltage, rates, output)
        self.Vm = np.zeros((self.n_neurons, steps))
        self.spiketrains = np.zeros((self.n_neurons, steps))
        # self.Vt = np.zeros((self.n_neurons, steps))
        self.rate = np.zeros((self.n_neurons, steps))
        self.output = np.zeros((2, steps)) #CHANGE 1

        #Initialise initial value for threshold voltage
        F = self.fastConnections['input'].weights
        for i in range(0, self.n_neurons):
            self.Vt[i] = 0.5 * (np.dot(F[:,i], F[:,i].T) + self.regL1*self.leak + self.regL2*self.leak**2)
        #Add recurrent connections that implement L1 & L2 regularisation on firing rates (Boerlin 2013)
        if not self.pars.adaptiveThreshold and not self.pars.learning:
            self.addConnection(self.pars, node=self, weights= - self.regL2 * self.leak**2 * np.eye(self.n_neurons), connType='fast')


    def addConnection(self, pars, node, weights, conn_type = 'fast', delay = 0, learning_rule = None):
        
        if weights.shape != (node.n_neurons, self.n_neurons):
            raise ValueError("Passed weight array is not of the right shape")

        new_proj = Connection(pars, node, self, weights, delay, learning_rule)

        if conn_type == 'fast':
            if node.name in self.fastConnections: #check if already existing connection
                self.fastConnections[node.name] += new_proj
                print("W: adding provided weight matrix to existing connection.")
            else:
                self.fastConnections[node.name] = new_proj
        elif conn_type == 'slow':
            if node.name in self.slowConnections: #check if already existing connection
                self.slowConnections[node.name] += new_proj
                print("W: adding provided weight matrix to existing connection.")
            else:
                self.slowConnections[node.name] = new_proj
        else:
            print("Unsuccessful addition of connection from", node.name, "node to", self.name, "node. Please specify connection type either 'fast' or 'slow'.")
            return
        
        print("Successfully added connection of type", conn_type, "from", node.name, "node to", self.name, "node.")


    def updateWeights(self):
        for _, proj in self.fastConnections.items():
            proj.update()
        for _, proj in self.slowConnections.items():
            proj.update()


    def step(self, iter, timestep):
        #book keeping
        self.iter = iter
        #LEAK MEMBRANE VOLTAGE
        self.Vm[:,[iter]] = self.Vm[:,[iter-1]] - timestep * self.leak * (self.Vm[:,[iter-1]] + np.random.normal(0, self.noise * self.Vt[:,[0]], (self.n_neurons,1)))

        #PROCESS FAST CURRENTS
        for _, proj in self.fastConnections.items():
            node = proj.source
            weight = proj.weights
            delay = proj.delay
            if isinstance(node, CurrentInput):
                self.Vm[:,[iter]] += timestep * self.leak * weight.T @ node.I[:,[iter]]
            elif type(node) == Population:
                self.Vm[:,[iter]] += weight @ node.spiketrains[:,[iter-1]]

        #PROCESS SLOW CURRENTS
        for _, proj in self.slowConnections.items():
            node = proj.source
            weight = proj.weights
            delay = proj.delay
            self.Vm[:,[iter]] += (1/self.leak) * weight @ node.rate[:,[iter-1]]        

        #ADAPTIVE THRESHOLD
        # if self.pars.adaptiveThreshold:
        #     self.Vt[:,[iter]] = self.Vt[:,[iter-1]] - timestep * self.leak * (self.Vt[:,[iter-1]] - self.Vt[:,[0]]) + self.regL2 * self.leak**2 * self.spiketrains[:,[iter-1]]
        # else:
        #     self.Vt[:,[iter]] = self.Vt[:,[iter-1]]

        self.refrac_counter = np.maximum(0, self.refrac_counter - timestep)

        if self.pars.oneSpikePerStep:
            # VaboveThresh = self.Vm[:,[iter]] - self.Vt[:,[iter]]
            VaboveThresh = self.Vm[:,[iter]] - self.Vt
            # VaboveThresh = VaboveThresh * self.refrac_mask

            if np.amax(VaboveThresh) < 0:
                pass
            else: #neuron spiked
                refrac_mask = (self.refrac_counter <= 0).flatten() #neurons not currently in refractory period
                if refrac_mask.any() == True:
                    subset_idx = np.argmax(VaboveThresh[refrac_mask])
                    parent_idx = np.arange(VaboveThresh.shape[0])[refrac_mask][subset_idx]

                    self.spiketrains[parent_idx,[iter]] = 1

        else: #BREAKS THE SIMULATION
            self.spiketrains[:,[iter]] = np.greater(self.Vm[:,[iter]], self.Vt[:,[iter]])

        self.refrac_counter += self.spiketrains[:,[iter]] * self.pars.refractory_period

        #UPDATE RATES
        self.rate[:,[iter]] = (1 - self.leak * timestep) * self.rate[:,[iter-1]] + self.spiketrains[:,[iter]]

        #UPDATE OUTPUTS
        if 'input' in self.fastConnections:
            self.output[:,[iter]] = (1 - self.leak * timestep) * self.output[:,[iter-1]] + self.fastConnections['input'].weights @ self.spiketrains[:,[iter]]

        #UPDATE WEIGHTS
        if self.pars.learning:
            self.updateWeights()   


    def reset(self, steps):
        self.Vm = np.zeros((self.n_neurons, steps))
        self.spiketrains = np.zeros((self.n_neurons, steps))
        # self.Vt = np.zeros((self.n_neurons, steps))
        self.rate = np.zeros((self.n_neurons, steps))
        self.output = np.zeros((3, steps)) #CHANGE 1


    def get_data(self):
        return