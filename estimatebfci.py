#!/usr/bin/env python
from __future__ import print_function
import numpy 
import matplotlib
import matplotlib.pyplot as pyplot

class BFCIEstimate(object):
    def __init__(self, k, t):
        '''
        Estimate the confidence interval for a rate constant estimate from 
        brute force simulations on a system with rate constant ``k``, simulated 
        for aggregate time ``t``.
        '''
        self.k = k 
        self.t = t

    def convert_to_step(self, dataset):
        return numpy.hstack((numpy.array(0.0),
                            numpy.repeat(dataset, 2),
                            numpy.array(0.0)
                            ))

    def estimate(self, nsamples=10000, plot=False):
        k = self.k
        t = self.t
        se_k_list = []
        duration_list = []
        for isample in xrange(nsamples):
            cumulative_time = 0
            fpt_list = []
            while True:
                rand = numpy.random.random() 
                # cumulative density function of exponential distribution is 
                # 1 - exp(-kx) 
                # rand = 1 - exp(-kx)
                # 1 - rand = exp(-kx)
                # ln(1-rand) = -kx 
                # -ln(1-rand)/k = x
                fpt = -numpy.log(1-rand)/k
                cumulative_time += fpt
                if cumulative_time > t:
                    break
                else:
                    fpt_list.append(fpt)
                    duration_list.append(fpt)
            fpt_array = numpy.array(fpt_list) 
            # Calculate mean and standard error
            mfpt = fpt_array.mean()
            se_mfpt = fpt_array.std(ddof=1)/numpy.sqrt(fpt_array.shape[0])
            # The relative error is the same when we take a reciprocal (first order error propagation)
            se_k = se_mfpt/(mfpt**2)
            #se_k = abs(-se_mfpt/(mfpt**2)+(se_mfpt)**2/(mfpt**3))
            se_k_list.append(se_k)
        se_k_array = numpy.array(se_k_list)
        pyplot.hist(duration_list, bins=100, normed=True)
        xs = numpy.linspace(0,.008,10000)
        pyplot.plot(xs, k*numpy.exp(-k*xs), color='black')
        pyplot.savefig('durations.pdf')
        pyplot.clf()
        print(se_k_array.mean())
        print(numpy.nanmean(se_k_array))
        print(se_k_array)
        if plot:
            self._plot(se_k_array)

    def estimate_2(self, nsims, simlength, nsamples=1000, bootstrap_samples=200, plot=False):
        k = self.k
        se_k_list = []
        k_list = []
        full_fpt_list = [] 
        for i in xrange(nsamples):
            #print("\r{:06d}".format(i), end='')
            event_array = numpy.zeros(nsims)
            fpt_list = []
            for sim in xrange(nsims):
                rand = numpy.random.random()
                fpt = -numpy.log(1-rand)/k
                if fpt < simlength:
                    full_fpt_list.append(fpt)
                    fpt_list.append(fpt)
                    event_array[sim] = 1

            fpt_list = numpy.array(fpt_list)
            k_array = numpy.zeros(bootstrap_samples)
            total_waiting_time = (nsims-fpt_list.shape[0])*simlength + fpt_list.sum()
            for j in xrange(bootstrap_samples):
                synthetic_event_array = numpy.random.choice(event_array, size=nsims)
                k_hat = synthetic_event_array.sum()/total_waiting_time
                k_array[j] = k_hat 
            mean_k = k_array.mean()
            k_list.append(mean_k)
            se = k_array.std(ddof=1)
            se_k_list.append(se)
            print("{:f}:  {:f}".format(mean_k, se))
        se_k_array = numpy.array(se_k_list)
        #densities, edges, _ = pyplot.hist(full_fpt_list, bins=100, normed=True)
        #print(edges)
        xs = numpy.linspace(0,.008,10000)
        pyplot.plot(xs, k*numpy.exp(-k*xs), color='black')
        pyplot.axvline(max(full_fpt_list), color='green') 
        pyplot.savefig('durations.pdf')
        pyplot.clf()
        print(numpy.array(k_list).mean())
        print(se_k_array.mean())
        print(numpy.nanmean(se_k_array))
        if plot:
            self._plot(se_k_array)

    def _plot(self, se_k_array):
        matplotlib.rcParams['font.size'] = 10
        hist, edges = numpy.histogram(se_k_array, density=True, bins=100) 
        ys = self.convert_to_step(hist) 
        fig, ax = pyplot.subplots(figsize=(3.42,2))
        fig.subplots_adjust(bottom=0.35, left=0.25)
        ax.plot(edges.repeat(2), ys, color='black', lw=1.5)
        for kw in ('top', 'right'):
            ax.spines[kw].set_visible(False)
        for kw in ('left', 'bottom'):
            ax.spines[kw].set_linewidth(1.5)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='out', width=1.5)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.set_xlabel('se$_{\hat k}$')
        ax.set_ylabel('probability')
        ax.set_xlim((0, edges[-1]))
        pyplot.savefig('err_dist.pdf')
        
def main():
    # 7160, N -> N'
    #e = BFCIEstimate(2.15*10**4, .003)
    #e.estimate(nsamples=100000, plot=True)
    #e.estimate_2(3000, .000001, nsamples=1000, plot=True)

    # 7160, N' -> N
    #e = BFCIEstimate(12.1, .003)
    #e.estimate(nsamples=100000, plot=True)

    # 60602, N' -> N
    #e = BFCIEstimate(12.1, .003)
    #e.estimate(nsamples=100000, plot=True)

    # 60602, N -> N'
    #e = BFCIEstimate(1.39*10**3, .003)
    #e.estimate_2(3000, .000001, nsamples=1000, plot=True)
    #e.estimate(nsamples=100000, plot=True)

    # 7160F50pF66m05
    e = BFCIEstimate(3.86*10**5, 3*1.5*200*.000001)
    #e.estimate_2(3*1.5*200, .000001, nsamples=1000, plot=True)
    e.estimate_2(900, .000001, nsamples=1000, plot=True)
    #e.estimate(nsamples=100000, plot=True)


if __name__ == '__main__':
    main()
           
        
       
