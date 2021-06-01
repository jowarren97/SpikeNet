from arb_functions import *
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
            if self.pars.weight_init == 'equidist':
                F = equidist_weights(p.n_neurons, self.input.n_neurons)
            elif self.pars.weight_init == 'random':
                F = np.random.normal(loc=0.0, scale=1.0, size=(self.input.n_neurons, p.n_neurons))
            elif self.pars.weight_init == 'amputated':
                F = equidist_weights(p.n_neurons, self.input.n_neurons, amputation_frac=0.5)
            elif self.pars.weight_init == 'bipolar':
                F = np.array(int(p.n_neurons/2)*[self.input.n_neurons*[1.0]]+int(p.n_neurons/2)*[self.input.n_neurons*[1.0]]).T
            else:
                raise ValueError('Incorrect weight initalisation option specified.')
            
            F = normalize(F, self.pars.weight_scale)
            p.inp_dim = self.input.n_neurons
            p.addConnection(self.pars, node=self.input, weights=F, learning_rule=self.pars.learning_rule_fwd)

            #add recurrent connections
            if self.pars.learning_rule_rec is None:
                w_init = - F.T @ F
            else:
                w_init = - 0.001*np.random.rand(p.n_neurons) - 0.005*np.eye(p.n_neurons)
                # w_init = np.zeros((p.n_neurons,p.n_neurons))
            p.addConnection(self.pars, node=p, weights=w_init, learning_rule=self.pars.learning_rule_rec)

        return

    def initialise(self):
        self.input.initialise(self.steps, self.timestep)

        for pop in self.populations:
            pop.initialise(self.steps)

    def step(self, iter, learning):
        self.input.step(iter, self.timestep)

        for pop in self.populations:
            pop.step(iter, self.timestep, learning)

    
    def __call__(self, learning=True):
        self.initialise()

        for i in range(self.steps):
            self.step(i, learning)

        if len(self.populations) > 1:
            raise NotImplementedError()
        else:
            return np.mean(np.linalg.norm((self.populations[0].output[:,-100:] - self.input.I[:,-100:]), axis=0))


