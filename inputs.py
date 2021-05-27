from node import Node
import numpy as np

class SpikingInput(Node):
    def __init__(self, name, n_neurons):
        super().__init__('name', n_neurons)

class CurrentInput(Node):
    def __init__(self, name, n_neurons):
        super().__init__(name, n_neurons)
        self.I = np.array([])

    def initialise(self, steps, timestep):
        pass

    def parseAmplitudes(self, amplitudes):
        if amplitudes == None:
            amplitudes = np.array(self.n_neurons*[[1]])
        elif type(amplitudes) == list:
            amplitudes = np.asarray([amplitudes]).T
        elif type(amplitudes) != np.ndarray:
            raise TypeError("Amplitude argument is neither a list nor nd array")
        
        if amplitudes.shape != (self.n_neurons, 1):
            raise ValueError("Amplitude argument is not of the right shape")
        return amplitudes

class ConstantCurrentInput(CurrentInput):
    def __init__(self, n_neurons, amplitudes = None):
        super().__init__('input', n_neurons)
        self.I = self.parseAmplitudes(amplitudes)

    def initialise(self, steps, timestep):
        self.I = np.repeat(self.I, steps, 1)

class SinusoidalCurrentInput(CurrentInput):
    def __init__(self, n_neurons, phases = None, amplitudes = None, angularVelocity = 1/20, pars=None):
        super().__init__('input', n_neurons)  
        self.amplitudes = self.parseAmplitudes(amplitudes)
        self.phases = phases
        self.omega = angularVelocity
        self.pars = pars

    def initialise(self, steps, timestep):
        t = np.arange(0, steps) * timestep
        self.x = np.zeros((self.n_neurons, steps))
        self.xdot = np.zeros((self.n_neurons, steps))

        for i in range(self.n_neurons):
            self.x[i,:] = self.amplitudes[i] * np.sin(2*np.pi * self.omega * t + self.phases[i] * 2*np.pi/360)
            self.xdot[i,:] = self.amplitudes[i] * 2*np.pi * self.omega * np.cos(2*np.pi * self.omega * t + self.phases[i] * 2*np.pi/360)

        self.I = self.x + (1/self.pars.leak)*self.xdot #!!! 10 is 1/leak !!! IMPLEMENT BETTER

class GaussianCurrentInput(CurrentInput):
    def __init__(self, n_neurons, mean, covariance):
        super().__init__('input', n_neurons)   
        self.name = 'input'
        self.n_neurons = n_neurons
        self.mean = mean
        self.covariance = covariance

    def initialise(self):
        if self.covariance.shape != (self.n_neurons, self.n_neurons):
            raise ValueError("Covariance array is not of the right shape")
        if self.mean.shape != (self.n_neurons, 1):
            raise ValueError("Mean array is not of the right shape")

        #self.I = #WRITE CODE

# class SquareCurrentInput(CurrentInput):
#     def __init__(self, on_duration, off_duration, amplitude):
        