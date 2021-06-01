import numpy as np

class DotDict(dict):
    # dot.notation access to dictionary attributes

    def __getattr__(*args):
        val = dict.__getitem__(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def normalize(a, scale, axis=0, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return scale * a / np.expand_dims(l2, axis)

def equidist_weights(n_neurons, n_inp, amputation_frac=0.0):
    
    W = np.zeros((n_inp, n_neurons))

    if n_inp >= 3:
        raise NotImplementedError()
    
    # d_theta = 2*np.pi / n_neurons
    d_theta = 2 * (1-amputation_frac) * np.pi / n_neurons

    for i in range(n_neurons):
        theta = i * d_theta
        W[:,i] = [np.sin(theta), np.cos(theta)]

    return W

def rgb2gray(frame):
        return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])

def whiten_and_filter(frame, r):

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        return rho, phi

    h, w = frame.shape
    imf = np.fft.fftshift(np.fft.fft2(frame))
    fx, fy = np.meshgrid(np.arange(-w / 2, w / 2), np.arange(-h / 2, h / 2))
    rho, theta = cart2pol(fx, fy)
    filtf = rho * np.exp(-0.5 * (rho / (0.7 * r / 2)) ** 2)
    imwf = filtf * imf
    imw = np.real(np.fft.ifft2(np.fft.fftshift(imwf)))

    return imw

    