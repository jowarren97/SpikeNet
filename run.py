from parameters import *
from SCN_model import SCN
from population import Population
from learning_rules import *
from inputs import SinusoidalCurrentInput
import graphing
import matplotlib.pyplot as plt
import numpy as np

params = default_params()
T = params.duration
N = params.n_neurons
# pop = Population(params)
inp = SinusoidalCurrentInput(n_neurons = 1, amplitudes = [1.0], angularVelocity=1/500, pars=params)

# r = np.array(int(N/2)*[[0.1]]+int(N/2)*[[-0.1]]).T
# # r = np.array([[0.1]])
# pop.addConnection(params, node=inp, weights=r, learning_rule=None)

# if not params.learning:
#     w_init = - r.T @ r
# #pop.addReccurence(weights = w)
# else:
#     # w_init = - 0.001*np.random.rand(N,N) - 0.005*np.eye(N,N)
#     w_init = np.zeros((N,N))
# pop.addConnection(params, node=pop, weights=w_init, learning_rule=enforcedRec)
# pop.addOutput(r, 1)

# net = SCN(params)
# net.addPopulations([pop])
# net.addInputs([inp])
# net.initialise()

net = SCN(params, inp)

net()

print("done simulation")
print(pop.Vt[0,0:2])
t = np.arange(0, params.duration, params.timestep)
# #plt.figure()
# #plt.plot(pop.fastConnections[pop.name].weightHistory[0,:], pop.fastConnections[pop.name].weightHistory[1])
# fig = plt.figure()

# ax = fig.add_subplot(321)
# graphing.plotOutputInput(inp.x, pop.output, t, ax)

# ax2 = fig.add_subplot(323)
# for v in pop.Vm[:1]:
#     ax2.set_xlim(0, t[-1])
#     ax2.plot(t, v)
#     ax2.set_xlabel("time /ms")
#     ax2.set_ylabel("Vm")

# for T in pop.Vt:
#     ax2.plot(t, T, '--')

# ax3 = fig.add_subplot(325)
# graphing.plotSpiketrains(pop, ax3, t)

# ax4 = fig.add_subplot(122)
# graphing.plotISI(pop, ax=ax4)

# plt.tight_layout()
# plt.show()

fig1 = plt.figure()
graphing.plotOutputInput(inp.x, pop.output, t, fig1.gca())

fig2, axes = plt.subplots(nrows=1, ncols=3)
im = axes[0].imshow(w_init, vmin=-0.01, vmax=0.01, cmap='hot')
im = axes[1].imshow(pop.fastConnections[pop.name].weights, vmin=-0.01, vmax=0.01, cmap='hot')
im = axes[2].imshow(- r.T @ r, vmin=-0.01, vmax=0.01, cmap='hot')
fig2.subplots_adjust(right=0.8)
cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
fig2.colorbar(im, cax=cbar_ax)

fig3 = plt.figure()
graphing.plotSpiketrains(pop, fig3.gca(), t)

loss = (pop.output - inp.x)**2
fig4 = plt.figure()
plt.plot(t, loss.flatten())

plt.show()
