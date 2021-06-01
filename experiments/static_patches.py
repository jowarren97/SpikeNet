import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from parameters import *
from SCN_model import SCN
from learning_rules import *
from inputs import SinusoidalCurrentInput
import plotting_functions
import matplotlib.pyplot as plt
import numpy as np

def train(model):
    loss = model()
    spike_count = np.count_nonzero(model.populations[0].spiketrains)
    return loss, spike_count

def test(model, params):
    loss = 0
    spike_count = 0
    for i in range(params.test_its):
        loss += model(learning=False) / params.test_its
        spike_count += np.count_nonzero(model.populations[0].spiketrains) / params.test_its
    return loss, spike_count

params = patches_params()
T = params.duration
N = params.n_neurons

print(params.weight_init)

inp = ImagePatchInput('input', path='../data/test_images/', patch_size=(8,8), r=200)
net = SCN(params, inp)
pop=net.populations[0]
w_init = pop.fastConnections['input'].weights

av_loss = []

for i in range(params.train_its):
    print('iteration', i)
    _, s = train(net)
    print('spike count:', s)

    if i % params.test_freq == 0:    
        print('testing model')
        av_loss.append(test(net, params))


plt.figure()
test_ticks = np.arange(0, params.train_its, params.test_freq)
plt.plot(test_ticks, np.array(av_loss))

w = pop.fastConnections['input'].weights
n = params.n_neurons
scale = params.weight_scale

fig, axes = plt.subplots(4,5)

for i, ax in enumerate(axes.flatten()):
    if i < n:
        ax.imshow(np.reshape(w_init[:,i], (8,8)), cmap='gray')

fig, axes = plt.subplots(4,5)

for i, ax in enumerate(axes.flatten()):
    if i < n:
        ax.imshow(np.reshape(w[:,i], (8,8)), cmap='gray')

plt.show()