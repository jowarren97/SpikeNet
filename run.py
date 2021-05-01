from euler import *
from inputs import *

T = 4000
N = 100
pop = Population(name = 'pop', n_neurons = N)
inp = SinusoidalCurrentInput(n_neurons = 1, amplitudes = [2.0], angularVelocity=1/50)

r = np.array(int(N/2)*[[0.1]]+int(N/2)*[[-0.1]]).T
pop.addConnection(node = inp, weights = r)

w = - r.T @ r
#pop.addReccurence(weights = w)
w_rand = - 0.001*np.random.rand(N,N) - 0.005*np.eye(N,N)
pop.addReccurence(weights = w_rand, plastic=True)
pop.addOutput(r, 1)

sim = Simulation()
sim.addPopulations([pop])
sim.addInputs([inp])

sim.run(duration = T)

print("done simulation")

t = np.arange(0, T, 0.1)
#plt.figure()
#plt.plot(pop.fastConnections[pop.name].weightHistory[0,:], pop.fastConnections[pop.name].weightHistory[1])
fig = plt.figure()

ax = fig.add_subplot(321)
graphing.plotOutputInput(inp.x, pop.output, t, ax)

ax2 = fig.add_subplot(323)
for v in pop.Vm[:1]:
    ax2.set_xlim(0, t[-1])
    ax2.plot(t, v)
    ax2.set_xlabel("time /ms")
    ax2.set_ylabel("Vm")

for T in pop.Vt:
    ax2.plot(t, T, '--')

ax3 = fig.add_subplot(325)
graphing.plotSpiketrains(pop, ax3, t)

ax4 = fig.add_subplot(122)
graphing.plotISI(pop, ax=ax4)

plt.tight_layout()
plt.show()
