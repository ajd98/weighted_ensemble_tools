#!/usr/bin/env python
import h5py
import numpy
import matplotlib
import matplotlib.pyplot as pyplot

def convert_to_step(dataset):
    return numpy.hstack((numpy.array(0.0),
                        numpy.repeat(dataset, 2),
                        numpy.array(0.0)
                        ))

class DurationHistogram(object):
    def __init__(self):
        pass

    def from_list(self, kinetics_path_list, lastiter=2000, **kwargs):
        weights = []
        durations = []
        for path in kinetics_path_list: 
            print('Loading {:s}'.format(path))
            kinetics_file = h5py.File(path, 'r')
            if lastiter is not None:
                where = numpy.where(kinetics_file['durations'][:lastiter]\
                                                 ['weight'] > 0)
                d = kinetics_file['durations'][:lastiter]['duration'] 
                w = kinetics_file['durations'][:lastiter]['weight']
            else:
                where = numpy.where(kinetics_file['durations']['weight'] > 0)
                d = kinetics_file['durations']['duration'] 
                w = kinetics_file['durations']['weight']
            for i in range(where[1].shape[0]):
                weight = w[where[0][i],where[1][i]]
                duration = d[where[0][i],where[1][i]]
                if duration > 0:
                    durations.append(duration)
                else:
                    durations.append(where[0][i])
                weights.append(weight)

        weights = numpy.array(weights)
        durations = numpy.array(durations)
        print(durations.min())

        self.histogram(durations, weights, lastiter=lastiter, **kwargs)
        return

    def integrate(self, hist, edges):
        deltas = numpy.array([edges[i+1] - edges[i] \
                              for i in range(len(edges)-1)])
        integral = (deltas*hist).sum()
        return integral

    def normalize_density(self):
        integral = self.integrate(self.hist, self.edges)
        self.hist /= integral
        return 
         
    def histogram(self, durations, weights, binwidth=1, lastiter=None,  
                  correction=True, **kwargs):
        lb = 0
        ub = numpy.ceil(durations.max()) 
        edges = numpy.arange(lb, ub+binwidth, binwidth)
        hist, _ = numpy.histogram(durations, weights=weights, bins=edges,
                                  density=True)

        print(correction)
        if correction:
            factors = 1/(lastiter - numpy.arange(0, lastiter, 
                                                 binwidth, dtype=float))
            hist = hist*factors[:hist.shape[0]] 
        print(kwargs)
        self.hist = hist
        self.edges = edges
        self.normalize_density()
         
        return

    def plot_hist(self, outpath='durations.pdf', color='black', 
                  log=False, ax=None):
        matplotlib.rcParams['font.size'] = 9
        linewidth=1.5
        
        if ax is None:
            fig, ax = pyplot.subplots() 
            fig.set_size_inches(2.42,2)
        else:
            fig = pyplot.gcf()

        if log:
            ax.plot(numpy.repeat(self.edges, 2), numpy.log(convert_to_step(self.hist)), 
                    color=color)
        else:
            ax.plot(numpy.repeat(self.edges, 2), convert_to_step(self.hist), color=color)
        for kw in ('top', 'right'):
            ax.spines[kw].set_visible(False)
        for kw in ('bottom', 'left'):
            ax.spines[kw].set_linewidth(linewidth)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='out', width=linewidth)
        ax.set_xlabel('duration time (WE iterations)')
        if log:
            ax.set_ylabel('log(probability density)')
        else:
            ax.set_ylabel('probability density')
        if not log:
            ax.set_ylim(0, self.hist.max()*1.2)

        fig.subplots_adjust(bottom=0.2, left=0.25)
        pyplot.savefig(outpath)


def main():

    pathlist = ['/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/'
                'kinetics_files/71_60_N2NP_{:d}_kinetics.h5'.format(i)\
                for i in range(1,11)]
    pathlist = ['/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_60_2/analysis/'
                'kinetics_files/60_60_NP2N_{:d}_kinetics.h5'.format(i)\
                for i in range(1,11)]
    pathlist = ['./new_analysis/kinetics_files/71_60_N2NP_{:d}_kinetics.h5'\
                .format(i) for i in range(1,11)]
    h = DurationHistogram()
    h.from_list(pathlist, correction=True, lastiter=2000, binwidth=10)
    h.plot_hist(color='blue')
    

if __name__ == "__main__":
    main()
