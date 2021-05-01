import numpy as np
from population import *

class Simulator:
    def __init__(self):
        self.populations = []
        self.inputs = []
        self.step = 1
        self.timestep = 0.1
        self.duration = 10
        self.steps = int(self.duration/self.timestep)
        self.oneSpikePerStep = True

    def addPopulations(self, populations):
        self.populations += populations

    def addInputs(self, inputs):
        self.inputs += inputs

    def setDuration(self, duration):
        self.duration = duration
        self.steps = int(duration/self.timestep)

    def setTimestep(self, timestep):
        self.timestep = timestep
        self.steps = int(self.duration/timestep)

    def initialise(self):
        for inp in self.inputs:
            inp.initialise(self.steps, self.timestep)

        for pop in self.populations:
            pop.initialise(self.steps)

    def propagate(self):
        for pop, inp in zip(self.populations, self.inputs):
            pop.propagate(self.step, self.timestep, self.oneSpikePerStep)
            #inp.propogate(self.step, self.timestep)
        self.step += 1

    def run(self, duration):
        self.setDuration(duration)
        self.initialise()

        while self.step < self.steps:
            self.propagate()