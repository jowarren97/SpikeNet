from learning_rules import *

class default_params():
    def __init__(self):
        self.n_neurons = 20
        self.leak = 0.01
        self.noise = 0.1
        self.regL1 = 0.0
        self.regL2 = 0.0 #doesnt seem to work with learning...?
        self.adaptiveThreshold = False
        self.refractory_period = 0 #ms

        self.duration = 10000 #ms
        self.timestep = 1 #ms

        if self.leak*self.timestep >= 1:
            raise ValueError('Leak/timestep combination is greater than 1')

        self.steps = int(self.duration/self.timestep)
        self.oneSpikePerStep = True
        self.save_data = ['output', 'spiketrains']

        #CONNECTIONS
        self.learning = ['fwd', 'rec']
        self.learning_onset = 0
        self.learning_rule_rec = None #'enforced' 
        self.learning_rule_fwd = None
        self.use_pseudo = False
        self.lr = 0.0001 #0.01
        self.beta = 2
        self.alpha = 1

    