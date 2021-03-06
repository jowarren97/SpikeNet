from parameters import *
from SCN_model import SCN
from learning_rules import *
from inputs import SinusoidalCurrentInput
import plotting_functions
import matplotlib.pyplot as plt
import numpy as np

def train(model):
    model()
    return

def test(model):
    model(record_data=True)
    return

params = default_params()
T = params.duration
N = params.n_neurons

inp = SinusoidalCurrentInput(n_neurons = 2, amplitudes=[1,1], phases=[0,90], angularVelocity=1/500, pars=params)
net = SCN(params, inp)

# net()
for i in range(params.train_its):
    print('iteration', i)
    train(net)

pop = net.populations[0]
print("done simulation")
print(pop.Vt[0,0:2])
t = np.arange(0, params.duration, params.timestep)

fig1 = plt.figure()
plotting_functions.plotOutputInput(inp.x, pop.output, t, fig1.gca())

fig2, axes = plt.subplots(nrows=1, ncols=3)
# im = axes[0].imshow(w_init, vmin=-0.01, vmax=0.01, cmap='hot')
w_learnt = pop.fastConnections[pop.name].weights
w_true = - pop.fastConnections['input'].weights.T @ pop.fastConnections['input'].weights
v_min = np.min((np.min(w_true), np.min(w_learnt)))
v_max = np.min((np.max(w_true), np.max(w_learnt)))
im = axes[1].imshow(w_learnt, vmin=v_min, vmax=v_max, cmap='hot')
im = axes[2].imshow(w_true, vmin=v_min, vmax=v_max, cmap='hot')
fig2.subplots_adjust(right=0.8)
cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
fig2.colorbar(im, cax=cbar_ax)

fig3 = plt.figure()
plotting_functions.plotSpiketrains(pop, fig3.gca(), t)

loss = np.linalg.norm((pop.output - inp.x), axis=0)
fig4 = plt.figure()
plt.plot(t, loss.flatten())

plt.show()
