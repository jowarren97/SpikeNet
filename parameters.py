from arb_functions import DotDict
from learning_rules import *

def default_params():
    params = DotDict()
    params.n_neurons = 4
    params.leak = 0.01
    params.noise = 0.1
    params.regL1 = 0.0
    params.regL2 = 0.0 #doesnt seem to work with learning...?
    params.adaptiveThreshold = False
    params.refractory_period = 0 #ms

    params.duration = 100000 #ms
    params.timestep = 1 #ms

    if params.leak*params.timestep >= 1:
        raise ValueError('Leak/timestep combination is greater than 1')

    params.steps = int(params.duration/params.timestep)
    params.oneSpikePerStep = True
    params.save_data = ['output', 'spiketrains']

    #CONNECTIONS
    params.learning = ['fwd', 'rec']
    params.learning_onset = 0
    params.learning_rule_rec = brendel2020rec #'enforced' 
    params.learning_rule_fwd = None
    params.use_pseudo = False
    params.enforced_norm = True
    params.lr_rec = 0.01 #0.01
    params.lr_fwd = 0.0001
    params.beta = 2
    params.alpha = 1
    params.upper_V_cap = True #when learning recurrent weights, helps to cap voltages at Vthresh to prevent divergence
    params.lower_V_cap = False

    return params

    