from collections import deque
from math import exp

class event:
    def __init__(self, time, source, target, synapse):
        self.time = time
        self.target = target
        self.source = source
        self.synapse = synapse
    
class synapse():
    def __init__(self):
        self.weight = 0
        self.type = None

class neuron:
    def __init__(self):
        self.id = 0
        self.Vm = 0.0
        self.V_thresh = 0.0
        self.V_lambda = 0
        self.projections = []
        self.last_update = 0

    def send_spike(self):
        synapse = synapse() #!!!
        #CHANGE SYNAPSE
        spike = event(sim.time, self, None, None)
        sim.eventStack.append(spike)
        return

    def receive_spike(self, event):
        time = event.time
        time_since_update = time - self.last_update

        self.Vm = self.Vm * exp(-self.V_lambda * t_since_update)
        self.Vm += synapse.weight

        self.last_update = time
        return

class sim:
    def __init__(self):
        self.eventStack = deque()
        self.population = []
        self.time = 0

    def makePopulation(self, n_neurons):
        self.population = [neuron() for i in range(n_neurons)]

    def run(self, duration):
        while self.time < duration:
            event = self.eventStack.popleft()
            self.time = event.time

            proj = event.source.projections
            for neuron in proj:
                neuron.receive_spike(event)


sim = sim()
sim.makePopulation(10)
print(sim.population)

sim.run(10)

