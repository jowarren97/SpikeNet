import numpy as np
import matplotlib.pyplot as plt

def plotSignals(signals, ax, t)
    ax.set_xlim(0, t[-1])
    ax.set_ylabel("value")
    for signal in signals:  
        ax.plot(t, signal)

def plotISI(pop, timestep = 0.01, ax = None):
        ax = ax or plt.gca()

        intervals = []

        for spiketrain in pop.spiketrains:
            spiketimes = np.dot(np.where(spiketrain == 1), timestep)[0]

            prevSpiketime = spiketimes[0]
            for nextSpiketime in spiketimes[1:]:
                intervals += [nextSpiketime - prevSpiketime]
                prevSpiketime = nextSpiketime

        ax.hist(intervals, density=True, bins=100)
        ax.set_xlabel("ISI /ms")
        ax.set_ylabel("count")
        ax.set_xlim(0, max(intervals))
        ax.set_title('ISI plot for {0}'.format(pop.name))

def plotSpiketrains(pop, ax, t, **options):
    #handle_options(ax, options)
    max_index = pop.spiketrains.shape[0]
    min_index = 1
    for idx, spiketrain in enumerate(pop.spiketrains):
        indexed_train = [(idx+1) * spike for spike in spiketrain]
        ax.plot(t, indexed_train,
                #  np.ones_like(spiketrain) * i,
                'k.', markersize = 2)
                #'k.', **options)

    ax.set_ylabel("Neuron index")
    ax.set_ylim(-0.5 + min_index, max_index + 0.5)
    ax.set_xlabel("Time /ms")
    ax.set_xlim(0, t[-1])

