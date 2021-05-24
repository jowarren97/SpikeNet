import numpy as np

class DotDict(dict):
    # dot.notation access to dictionary attributes

    def __getattr__(*args):
        val = dict.__getitem__(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def normalize(a, axis=0, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return 0.1 * a / np.expand_dims(l2, axis)

def equidist_weights(n_neurons, n_inp):
    
    W = np.zeros((n_inp, n_neurons))

    if n_inp >= 3:
        raise NotImplementedError()
    
    d_theta = 2*np.pi / n_neurons

    for i in range(n_neurons):
        theta = i * d_theta
        W[:,i] = [np.sin(theta), np.cos(theta)]

    return W

    