#!/usr/bin/env python
import h5py
import numpy
import matplotlib.pyplot as pyplot
import scipy.stats

class CorrelationAnalysis:
    def __init__(self, kineth5pathlist, tstate, fi=None, li=None):
        self.kineth5pathlist = kineth5pathlist
        self.tstate = tstate
        self.fi = fi
        self.li = li
        self.open_kineth5_files()
    
    def open_kineth5_files(self):
        self.kineth5list = []
        for kineth5path in self.kineth5pathlist:
            self.kineth5list.append(h5py.File(kineth5path,'r+'))
        return

    def calculate(self, blocksize=1):
        autocorrels = []
        for kineth5 in self.kineth5list:
            if self.fi is not None and self.li is not None:
                number_flux = kineth5['total_fluxes']\
                                     [self.fi-1:self.li, self.tstate]
            elif self.fi is not None:
                number_flux = kineth5['total_fluxes'][self.fi-1:, self.tstate]
            elif self.li is not None:
                number_flux = kineth5['total_fluxes'][:self.li, self.tstate]

            number_flux = number_flux.reshape(-1,blocksize).mean(axis=1)
            autocorrel = self.estimate_autocorrelation(number_flux) 
            autocorrels.append(autocorrel)
        autocorrel_arr = numpy.array(autocorrels)
        autocorrel_mean = autocorrel_arr.mean(axis=0)
        autocorrel_std = autocorrel_arr.std(axis=0)
        autocorrel_se  = autocorrel_std/numpy.sqrt(autocorrel_arr.shape[0])
        student_t = scipy.stats.t.interval(0.95, autocorrel_arr.shape[0]-1)
        ci = autocorrel_se*student_t[1]
        return autocorrel_mean, ci

    def estimate_autocorrelation(self, x):
        """
        http://stackoverflow.com/q/14297012/190597
        http://en.wikipedia.org/wiki/Autocorrelation#Estimation
        """
        n = len(x)
        variance = x.var()
        x = x-x.mean()
        r = numpy.correlate(x, x, mode = 'full')[-n:]
        assert numpy.allclose(r, numpy.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
        result = r/(variance*(numpy.arange(n, 0, -1)))
        return result
        
    #def estimate_autocorrelation(self, X):
    #    n = len(X)
    #    Xbar = X.mean()
    #    corr = numpy.correlate(X-Xbar, X-Xbar)
    #    divisors = numpy.arange(n, 0, -1)
    #    autocorrel = corr/(X.var()*divisors) 
    #    return autocorrel

    def plot(self, blocksize, xlims=None, figname='autocorreltest.pdf'):
        mean, ci = self.calculate(blocksize)
        xs = numpy.arange(0, mean.shape[0]*blocksize, blocksize)
        fig, ax = pyplot.subplots(figsize=(7.25,4))
        ax.plot(xs, mean, color='black')
        ax.fill_between(xs, mean-ci, mean+ci, color=(0,0,0,0.2), 
                        linewidth=0.0)
        ax.set_ylim(-1,1)
        ax.set_xlabel('Lag time')
        ax.set_ylabel('autocorrelation of flux into target state')
        ax.axhline(y=0, color='black')
        if xlims is not None:
            ax.set_xlim(xlims)
        correl_time = 0
        for xval in xrange(0,mean.shape[0]):
            if mean[xval]-ci[xval] <=0:
                correl_time = xval*blocksize
                break
        print("Correlation time is {:d} iterations".format(correl_time))
        ax.axvline(x=correl_time, color='gray', ls='--')
       
        pyplot.savefig(figname)
        return
        

def main():
    s7160N2NPpaths = ['/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_1_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_2_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_3_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_4_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_5_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_6_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_7_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_8_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_9_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_10_kinetics.h5']

    ca = CorrelationAnalysis(s7160N2NPpaths, 1, fi=1, li=2000)
    ca.plot(1, xlims=(0,200), figname='1.pdf')
    ca.plot(2, xlims=(0,200), figname='2.pdf')
    ca.plot(5, xlims=(0,200), figname='5.pdf')
    ca.plot(10, xlims=(0,200), figname='10.pdf')
    ca.plot(20, xlims=(0,200), figname='20.pdf')
    ca.plot(50, xlims=(0,200), figname='50.pdf')

if __name__ == '__main__':
    main()
