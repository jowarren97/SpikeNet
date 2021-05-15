import numpy as np

def brendel2020rec(conn, pre_pop, post_pop):        
    step = post_pop.step
    Vm = post_pop.Vm[:,[step]].flatten() #post-synaptic population
    spiketrains = pre_pop.spiketrains[:,[step]] #pre-synaptic population
    rate = post_pop.rate[:,[step-1]].flatten() #post-synaptic rate
    regL2 = post_pop.regL2

    idxSpiked = np.where(spiketrains==1)[0]

    if step > conn.pars.learning_onset:
        for idx in idxSpiked:
            dw = - conn.pars.beta * (Vm + regL2*rate) - conn.weights[:,idx]
            dw[idx] -= regL2
            conn.weights[:,idx] += conn.pars.lr*dw
    return

def brendel2020fwd(conn, pre_pop, post_pop):   
    return     

def eligibilityTrace(conn, pre_pop, post_pop):
    step = post_pop.step
    post_spikes = post_pop.spiketrains[:,[step]]
    pre_spikes = pre_pop.spiketrains[:,[step]]
    alpha = (1 - post_pop.leak * conn.pars.timestep)
    V = post_pop.Vm[:,[step]]
    Vt = post_pop.Vt
    #             #output
    # W = #input [       ]
    conn.eligibilityTrace =  alpha * conn.eligibilityTrace + np.repeat(pre_spikes, post_pop.n_neurons, axis=1)

    if step > conn.pars.learning_onset:
        if conn.pars.use_pseudo:
            pseudo_deriv = 1/Vt * np.maximum(np.zeros_like(V), np.ones_like(V) - np.abs((V - Vt) / Vt))
            dw = - V.T * pseudo_deriv.T * conn.eligibilityTrace
            conn.weights += conn.pars.lr*dw 
        else:
            idx_spiked = np.where(post_spikes==1)[0]
            for idx in idx_spiked:
                dw = - V[idx] * conn.eligibilityTrace[:,idx]
                conn.weights[:,idx] += conn.pars.lr*dw

    return
