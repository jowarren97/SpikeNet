class SCN:
    def __init__(self, pars):
        self.pars = pars
        self.populations = []
        self.inputs = []
        self.step = 1
        self.timestep = pars.timestep
        self.duration = pars.duration
        self.steps = pars.steps
        self.oneSpikePerStep = pars.oneSpikePerStep

    def addPopulations(self, populations):
        self.populations += populations

    def addInputs(self, inputs):
        self.inputs += inputs

    def initialise(self):
        for inp in self.inputs:
            inp.initialise(self.steps, self.timestep)

        for pop in self.populations:
            pop.initialise(self.steps)

    def propagate(self, step):
        for pop, inp in zip(self.populations, self.inputs):
            pop.propagate(step, self.timestep)
            #inp.propogate(self.step, self.timestep)

    def __call__(self):
        self.initialise()

        for i in range(self.steps):
            self.propagate(i)

