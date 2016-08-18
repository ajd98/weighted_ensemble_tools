#!/usr/bin/env python
from histogram_durations import DurationHistogram
import numpy
import h5py
import matplotlib.pyplot as pyplot

class RateAnalysis(object):
    def __init__(self):
        pass

    def cum_mean(self, data, axis=0):
        N_arr = numpy.arange(1, data.shape[0]+1)
        cumsum = numpy.cumsum(data, axis=axis)

        if axis is not 0:
            numpy.swapaxes(cumsum, 0, axis)
        # Iterate over values along the 0th axis, which is now the one we want 
        # to cumulatively average over.
        for i in xrange(cumsum.shape[0]):
            cumsum[i] /= N_arr[i]

        if axis is not 0:
            numpy.swapaxes(cumsum, 0, axis)
             
        return cumsum 

    def get_iter_idx_range(self, h5file, fi, li, path):
        iter_start = h5file.attrs['iter_start']
        iter_stop = h5file.attrs['iter_stop']
        fidx = fi-iter_start
        lidx = fidx + (li-fi)
        if fidx < 0:
            raise IndexError('Data from iteration {:d} was requested, but '
                             'data in file {:s} starts at iteration {:d}'\
                             .format(fi, path, iter_start)
                             )
        if lidx > h5file['conditional_fluxes'].shape[0]:
            raise IndexError('Data from iteration {:d} was requested, but '
                             'data in file {:s} ends at iteration {:d}'\
                             .format(li-1, path, iter_stop-1)
                             )
        return fidx, lidx


    def load_conditional_flux(self, kineticsH5paths, istate, fstate, fi, li):
        '''
        Load the conditional flux based on data between iterations fi and li 
        from the w_kinetics HDF5 files specified in the iterable 
        kineticsH5paths. 
        '''
        cflist = []
        for path in kinavgH5paths:
            h5file = h5py.File(path, 'r+')
            cf = h5file['conditional_fluxes']
            fidx, lidx = self.get_iter_idx_range(h5file, fi, li, path)
            cflist.append(cf[fidx:lidx,istate, fstate])
        return numpy.vstack(cflist)

    def load_total_flux(self, kinavgH5paths, fstate, fi, li):
        '''
        Load the total flux based on data between iterations fi and li from the 
        w_kinetics HDF5 files specified in the iterable kineticsH5paths. 
        '''
        fluxlist = []
        for path in kinavgH5paths:
            h5file = h5py.File(path, 'r+')
            flux = h5file['total_fluxes']
            fidx, lidx = self.get_iter_idx_range(h5file, fi, li, path)
            fluxlist.append(flux[fidx:lidx, fstate])
        return numpy.vstack(fluxlist)
 

    def load_pops(self, assignH5paths, istate, fi, li):
        '''
        Load the labeled populations based on data between iterations fi
        and li from the assignment HDF5 files specified in the iterable 
        assignH5paths 
        '''
        poplist = []
        for path in assignH5paths:
            h5file = h5py.File(path, 'r+')
            # labeled_populations is indexes as iteration, state, bin
            pops = h5file['labeled_populations'][fi-1:li-2, istate].sum(axis=1)
            poplist.append(pops)
        return numpy.vstack(poplist)
            
    #def calc_rate_from_conditional_flux(self, kineticsH5paths, assignH5paths, 
    #                                    istate, fstate, fi, li):
    #    '''
    #    Calculate the rate from state ``istate`` to state ``fstate`` based on data
    #    on the conditional flux from iterations ``fi`` to ``li`` (right 
    #    exclusive) in kinetics HDF5 files found in ``kineticsH5paths``, and 
    #    labeled state populations found in assignH5paths.  Return a 2-tuple of 
    #    arrays representing the mean and standard errors in the rate constant.
    #    '''
    #    flux_arr = self.load_conditional_flux(kineticsH5paths, istate, fstate, 
    #                                          fi, li)
    #    pop_arr  = self.load_pops(assignH5paths, istate, fi, li)
    #    pop_list = []
    #    for simpop in pop_arr:
    #        pop_list.append(self.cum_mean(simpop))
    #    pop_arr = numpy.array(pop_list)

    #    flux_list = [] 
    #    for simflux in flux_arr:
    #        flux_list.append(self.cum_mean(simflux))
    #    flux_arr = numpy.array(flux_list)
    #    rates = flux_arr/pop_arr 
    #
    #    rate_mean = rates.mean(axis=0)     
    #    rate_se = rates.std(axis=0, ddof=1)/numpy.sqrt(rates.shape[0])
    #
    #    return rate_mean, rate_se 

    def calc_rate_from_total_flux(self, kineticsH5paths, fstate, 
                                  li, ax=None, durationbinwidth=1):
        '''
        Calculate the rate into state ``fstate`` based on data on the total flux
        from iterations 1 to ``li`` (right exclusive) in kinavg HDF5 files found
        in ``kineticsH5paths``.  This method assumes that the initial state 
        population is one. (IMPORTANT!) Return a 2-tuple of arrays representing 
        the mean and standard errors in the rate constant.
 
        kineticsH5paths:
          (Iterable) When iterated through, should return paths to the WESTPA 
          w_kinetics output files which should be analyzed.

        fstate:
          (int) The index of the "final" state for the rate calculation. This 
          method calculates the total flux into this state, and assumes this 
          is a valid rate (in general, this is only valid for steady-state 
          simulations.

        li:
          (int) The index of the final iteration to include in the analysis.
          Based on one-indexed iterations.

        ax:
          (matplotlib Axes object) (optional) An axis on which to plot results 
          from cumulative averaging.

        durationbinwidth:
          (float or int) The width of bins to be used in generating the 
          histogram that estimates the event duration distribution
        '''
        self.durationhistogram = DurationHistogram() 
        self.durationhistogram.from_list(kineticsH5paths, lastiter=li, 
                                         correction=True, 
                                         binwidth=durationbinwidth)

        flux_arr = self.load_total_flux(kineticsH5paths, fstate, 1, li)
        flux_arr = self.cum_mean(flux_arr, axis=0)
        for i in xrange(flux_arr.shape[1]):
            correction_factor = self.durationhistogram.integrate(
                                    self.durationhistogram.hist,
                                    self.durationhistogram.edges,
                                    ub = i)
            if i % 100 == 0: print(correction_factor)
            flux_arr[:,i] /= correction_factor

        
        rates = flux_arr 
        rate_mean = rates.mean(axis=0)     
        rate_se = rates.std(axis=0, ddof=1)/numpy.sqrt(rates.shape[0])
        
        rate_mean *= (10**10)
        rate_se *= (10**10)
        loglb = numpy.log(rate_mean-rate_se)/numpy.log(10)
        logub = numpy.log(rate_mean+rate_se)/numpy.log(10)
        logmean = numpy.log(rate_mean)/numpy.log(10)
     
        xs = numpy.arange(1, li, 1)
        xs = xs*.05

        print("mean rate is {:e}".format(rate_mean[-1]))
        print("se_k is {:e}".format(rate_se[-1]))

        if ax is not None:
            print('plotting')
            ax.fill_between(xs, loglb, logub, facecolor=(0.8,0.8,0.8,1), linewidth=0)
            ax.plot(xs, logmean, color=(0,0,0,1))
    
        return rate_mean, rate_se 

def test():
    pathlist = ['./new_analysis/kinetics_files/71_60_N2NP_{:d}_kinetics.h5'\
                .format(i) for i in range(1,11)]
    ra = RateAnalysis()

    fig = pyplot.gcf() 
    fig.set_size_inches(4,4)
    ax = fig.add_subplot(1,1,1)

    ra.calc_rate_from_total_flux(pathlist, 1, 2000, ax=ax, durationbinwidth=1)
    pyplot.savefig('test.pdf')


if __name__ == "__main__":
    test()
