from population import Population
import numpy as np

class SCN:
    def __init__(self, pars, input):
        self.pars = pars
        self.populations = []
        self.input = input
        self.step = 1
        self.timestep = pars.timestep
        self.duration = pars.duration
        self.steps = pars.steps
        self.oneSpikePerStep = pars.oneSpikePerStep
        self.buildNet()
        self.initialise()

    def buildNet(self):
        self.addPopulation(Population(self.pars))

        for p in self.populations:
            #add feedforward connections
            r = np.array(int(p.n_neurons/2)*[[0.1]]+int(p.n_neurons/2)*[[-0.1]]).T
            p.addConnection(self.pars, node=self.input, weights=r, learning_rule=self.pars.learning_rule_rec)

            #add recurrent connections
            if not self.pars.learning:
                w_init = - r.T @ r
            else:
                # w_init = - 0.001*np.random.rand(N,N) - 0.005*np.eye(N,N)
                w_init = np.zeros((p.n_neurons,p.n_neurons))
            p.addConnection(self.pars, node=p, weights=w_init, learning_rule=self.pars.learning_rule_fwd)

            #add output connections
            p.addOutput(r, 1)

        return

    def addPopulation(self, population):
        self.populations += [population]

    def addInput(self, input):
        self.input = input

    def initialise(self):
        self.input.initialise(self.steps, self.timestep)

        for pop in self.populations:
            pop.initialise(self.steps)

    def propagate(self, step):
        self.input.propagate(step, self.timestep)

        for pop in self.populations:
            pop.propagate(step, self.timestep)
            #inp.propogate(self.step, self.timestep)

    def __call__(self):
        self.initialise()

        for i in range(self.steps):
            self.propagate(i)

