import numpy as np
from inputs import *
from arb_functions import *

def brendel2020rec(conn, pre_pop, post_pop):        
    iter = post_pop.iter
    Vm = post_pop.Vm[:,[iter]].flatten() #post-synaptic population
    spiketrains = pre_pop.spiketrains[:,[iter]] #pre-synaptic population
    rate = post_pop.rate[:,[iter-1]].flatten() #post-synaptic rate
    regL2 = post_pop.regL2

    if conn.pars.upper_V_cap:
        Vm = np.minimum(Vm, post_pop.Vt.flatten())
    if conn.pars.lower_V_cap:
        Vm = np.maximum(Vm, -post_pop.Vt.flatten())

    idxSpiked = np.where(spiketrains==1)[0]

    if iter > conn.pars.learning_onset:
        for idx in idxSpiked:
            dw = - conn.pars.beta * (Vm + regL2*rate) - conn.weights[:,idx]
            dw[idx] -= regL2
            conn.weights[:,idx] += conn.pars.lr_rec*dw

    return

def brendel2020fwd(conn, pre_pop, post_pop):
    iter = post_pop.iter
    post_spikes = post_pop.spiketrains[:,[iter]]

    if isinstance(pre_pop, CurrentInput):
        inp = pre_pop.I[:,[iter]]
    else:
        raise NotImplementedError()

    post_spiked = np.where(post_spikes==1)[0]

    if iter > conn.pars.learning_onset:
        for idx in post_spiked:
            dw = conn.pars.alpha * inp.flatten() - conn.weights[:,idx]
            conn.weights[:,idx] += conn.pars.lr_fwd * dw
    return

def enforcedRec(conn, pre_pop, post_pop):
    r = post_pop.fastConnections['input'].weights
    conn.weights = - r.T @ r - conn.pars.regL2 * np.eye(pre_pop.n_neurons, post_pop.n_neurons) #need to change for w L2 reg   

def eligibilityTraceRec(conn, pre_pop, post_pop):
    iter = post_pop.iter
    post_spikes = post_pop.spiketrains[:,[iter]]
    pre_spikes = pre_pop.spiketrains[:,[iter]]
    alpha = (1 - post_pop.leak * conn.pars.timestep)
    V = post_pop.Vm[:,[iter]]
    Vt = post_pop.Vt
    #             #output
    # W = #input [       ]
    conn.eligibility_trace =  alpha * conn.eligibility_trace + np.repeat(pre_spikes, post_pop.n_neurons, axis=1)

    if iter > conn.pars.learning_onset:
        if conn.pars.use_pseudo:
            pseudo_deriv = 1/Vt * np.maximum(np.zeros_like(V), np.ones_like(V) - np.abs((V - Vt) / Vt))
            dw = - V.T * pseudo_deriv.T * conn.eligibility_trace
            conn.weights += conn.pars.lr*dw 
        else:
            idx_spiked = np.where(post_spikes==1)[0]
            for idx in idx_spiked:
                dw = - V[idx] * conn.eligibility_trace[:,idx]
                conn.weights[:,idx] += conn.pars.lr_rec*dw
    return

def eligibilityTraceFwd(conn, pre_pop, post_pop):
    iter = post_pop.iter

    if isinstance(pre_pop, CurrentInput):
        inp = pre_pop.I[:,[iter]]
    else:
        inp = pre_pop.spiketrains[:,[iter]]

    alpha = (1 - post_pop.leak * conn.pars.timestep)
    
    V = post_pop.Vm[:,[iter]]
    Vt = post_pop.Vt
    #             #output
    # W = #input [       ]
    conn.eligibility_trace =  (alpha * conn.eligibility_trace) + np.repeat(inp, post_pop.n_neurons, axis=1)

    if iter > conn.pars.learning_onset:
        if conn.pars.use_pseudo:
            pseudo_deriv = 1/Vt * np.maximum(np.zeros_like(V), np.ones_like(V) - np.abs((V - Vt) / Vt))

        else:
            pseudo_deriv = post_pop.spiketrains[:,[iter]] == 1

        dw = V.T * pseudo_deriv.flatten() * conn.eligibility_trace
        new_weights = conn.weights + conn.pars.lr_fwd*dw 

        if conn.pars.enforced_norm:
            conn.weights = normalize(new_weights, conn.pars.weight_scale)
        else:
            conn.weights = new_weights

    return
