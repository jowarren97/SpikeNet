import numpy as np
import matplotlib.pyplot as plt

def plotSignals(signals, ax, t):
    ax.set_xlim(0, t[-1])
    ax.set_ylabel("value")
    for signal in signals:  
        ax.plot(t, signal)

def plotOutputInput(inputs, outputs, t, ax=None):
    ax.set_xlim(0, t[-1])
    ax.set_xlabel('Time /ms')
    ax.set_ylabel("value")

    for output in outputs: 
        ax.plot(t, output, 'b')
    
    for input in inputs:
        ax.plot(t, input, 'r')
 


def plotISI(pop, timestep = 0.01, ax = None):
        ax = ax or plt.gca()
        intervals = []

        for spiketrain in pop.spiketrains:
            spiketimes = np.dot(np.where(spiketrain == 1), timestep)[0]

            if len(spiketimes) > 0:
                prevSpiketime = spiketimes[0]
                for nextSpiketime in spiketimes[1:]:
                    intervals += [nextSpiketime - prevSpiketime]
                    prevSpiketime = nextSpiketime

        ax.hist(intervals, density=True, bins=100)
        ax.set_xlabel("ISI /ms")
        ax.set_ylabel("probability density")
        ax.set_xlim(0, max(intervals))
        ax.set_title('ISI plot for {0}'.format(pop.name))

def plotSpiketrains(pop, ax, t, **options):
    max_index = pop.spiketrains.shape[0]
    min_index = 1
    ax.set_ylabel("Neuron index")
    ax.set_ylim(-0.5 + min_index, max_index + 0.5)
    ax.set_xlabel("Time /ms")
    ax.set_xlim(0, t[-1])
    ax.set_title('Spike trains for {0}'.format(pop.name))
    
    #handle_options(ax, options)

    for idx, spiketrain in enumerate(pop.spiketrains):
        indexed_train = [(idx+1) * spike for spike in spiketrain]
        ax.plot(t, indexed_train,
                #  np.ones_like(spiketrain) * i,
                'k.', markersize = 2)
                #'k.', **options)

def handle_options(ax, options):
    #from PyNN.plotting
    if "xticks" not in options or options.pop("xticks") is False:
        plt.setp(ax.get_xticklabels(), visible=False)
    if "xlabel" in options:
        ax.set_xlabel(options.pop("xlabel"))
    if "yticks" not in options or options.pop("yticks") is False:
        plt.setp(ax.get_yticklabels(), visible=False)
    if "ylabel" in options:
        ax.set_ylabel(options.pop("ylabel"))
    if "ylim" in options:
        ax.set_ylim(options.pop("ylim"))
    if "xlim" in options:
        ax.set_xlim(options.pop("xlim"))


