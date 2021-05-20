from population import Population
import numpy as np

class SCN:
    def __init__(self, pars, input):
        self.pars = pars
        self.populations = []
        self.input = input
        self.timestep = pars.timestep
        self.duration = pars.duration
        self.steps = pars.steps
        self.oneSpikePerStep = pars.oneSpikePerStep
        self.buildNet()
        self.initialise()

    def buildNet(self):
        self.populations += [Population(self.pars)]

        for p in self.populations:
            #add feedforward connections
            if self.pars.learning_rule_fwd is None:
                F = np.array(int(p.n_neurons/2)*[[0.1]]+int(p.n_neurons/2)*[[-0.1]]).T
            else:
                F = np.array(int(p.n_neurons/2)*[[0.1]]+int(p.n_neurons/2)*[[-0.1]]).T
            p.addConnection(self.pars, node=self.input, weights=F, learning_rule=self.pars.learning_rule_fwd)

            #add recurrent connections
            if self.pars.learning_rule_rec is None:
                w_init = - F.T @ F
            else:
                # w_init = - 0.001*np.random.rand(N,N) - 0.005*np.eye(N,N)
                w_init = np.zeros((p.n_neurons,p.n_neurons))
            p.addConnection(self.pars, node=p, weights=w_init, learning_rule=self.pars.learning_rule_rec)

        return

    def initialise(self):
        self.input.initialise(self.steps, self.timestep)

        for pop in self.populations:
            pop.initialise(self.steps)

    def step(self, iter, record_data):
        self.input.step(iter, self.timestep)

        for pop in self.populations:
            pop.step(iter, self.timestep)
            #inp.propogate(self.iter, self.timestep)

            if record_data:
                data = pop.get_data()

    def __call__(self, record_data=False):
        self.initialise()

        for i in range(self.steps):
            self.step(i, record_data)

