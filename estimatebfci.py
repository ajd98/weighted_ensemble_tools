#!/usr/bin/env python
from __future__ import print_function
import numpy 
import matplotlib
import matplotlib.pyplot as pyplot
import scipy.stats

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
        '''
        Estimate the size of confidence intervals from brute force simulations
        with the following setup:
      
        - Run a single simulation of length self.t, for a process with rate
          constant self.k.  Estimate the mean first passage time as the 
          arithmetic mean of the observed first passage times. Calculate the 
          standard error of the rate constant using propagation of error, 
          starting with the standard error of the mean first passage time. 

        - Repeat this process ``nsamples`` times and find the mean of the 
          observed standard erros.
        '''
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
        if plot:
            self._plot(se_k_array)

    def estimate_2(self, nsims, simlength, nsamples=1000, bootstrap_samples=200, plot=False):
        '''
        Estimate the size of confidence intervals from brute force simulations
        with the following setup:
      
        - Run ``nsims`` simulations each of length ``simlength`` (in units of 
          k^{-1}).

        - If we observe an event in a given simulation, we assume that the 
          simulation never returns to the initial state (e.g., because the
          reverse rate is slow compared to the timescale of each simulation).
          Thus, we observe a maximum of one event per simulation.

        - Calculate the aggregate waiting time as the sum of the times before
          the first event in each simulation.  If we do not observe an event in
          a given simulation, then that entire simulation is treated as waiting
          time.

        - Estimate the rate constant as:
                       (total number of events)/(total waiting time)
         
        - Resample with replacement from the set of tuples (EVENT, WAITING_TIME)
          where EVENT is a boolean indicating whether an event occured in a 
          given simulation, and WAITING_TIME is that simulation's contribution 
          to the total waiting time. Resampling generates one synthetic dataset
          of length ``nsims``.

        - Repeat the resampling procedure ``bootstrap_samples`` times, and 
          calculate the rate constant from each.

        - The standard deviation of the sampling distribution of rate constant
          estimates is the standard error of the mean.

        - Repeat this entire procedure ``nsamples`` times, and return the mean
          of the observed standard errors.
        '''
        k = self.k
        se_k_list = []
        k_list = []
        interval_width = scipy.stats.t.interval(0.95, nsims)[1]
        t_positive_array = numpy.zeros(nsamples)
        true_positive_array = numpy.zeros(nsamples)

        for i in xrange(nsamples):
            print("\r{:06d}".format(i), end='')
            event_array = numpy.zeros(nsims)
            rand = numpy.random.random(size=nsims)
            fpt = -numpy.log(1-rand)/k
            event_array = (fpt <= simlength)
            waiting_time_array = numpy.zeros(event_array.shape)
            waiting_time_array[:] = simlength 
            waiting_time_array[event_array] = fpt[event_array]

            k_array = numpy.zeros(bootstrap_samples)
            total_waiting_time = waiting_time_array.sum() 
            pairs = numpy.vstack(event_array, waiting_time_array).transpose()
            for j in xrange(bootstrap_samples):
                synthetic_pairs_array = numpy.random.choice(event_array, size=nsims)
                synthetic_pairs_array = \
                    pairs[numpy.random.randint(pairs.shape[0], size=nsims), :] 
                k_hat = synthetic_pairs_array[:,0].sum()/synthetic_pairs_array[:,1].sum()
                k_array[j] = k_hat 
            mean_k = k_array.mean()
            k_list.append(mean_k)
            se = k_array.std(ddof=1)
            se_k_list.append(se)
            if numpy.abs(mean_k - k) < se*interval_width:
                t_positive_array[i] = 1
            k_array.sort()
            if (k_array[lbi] <= k) and (k_array[ubi] >= k):
                true_positive_array[i] = 1 
        se_k_array = numpy.array(se_k_list)
        print("\nMean k from bootstrapping: {:e}"\
              .format(numpy.array(k_list).mean()))
        print('Mean standard error: {:e}'.format(se_k_array.mean()))
        print('Mean standard error, ignoring NaNs: {:e}'\
              .format(numpy.nanmean(se_k_array)))
        if plot:
            self._plot(se_k_array)
        return se_k_array.mean(), true_positive_array.mean(), t_positive_array.mean()


    def estimate_3(self, nsims, simlength, nsamples=1000, bootstrap_samples=200, plot=False):
        '''
        Estimate the size of confidence intervals from brute force simulations
        with the following setup:
      
        - Run ``nsims`` simulations each of length ``simlength`` (in units of 
          k^{-1}).

        - If we observe an event in a simulation, that simulation immediately 
          returns to the initial state.  Thus all simulation time is waiting 
          time. Moreover, we can observe multiple events per simulation, with
                      (number of events) ~ Poi(k*simlength)

        - Estimate the rate constant as:
                       (total number of events)/(total waiting time)
         
        - Resample with replacement from the set of values (NUMBER_OF_EVENTS)  
          where NUMBER_OF_EVENTS is the number of events observed in a given
          simulation. Resampling generates one synthetic dataset of length 
          ``nsims``.

        - Repeat the resampling procedure ``bootstrap_samples`` times, and 
          calculate the rate constant from each.

        - The standard deviation of the sampling distribution of rate constant
          estimates is the standard error of the mean.

        - Repeat this entire procedure ``nsamples`` times, and return the mean
          of the observed standard errors.
        '''
        k = self.k
        se_k_list = []
        k_list = []

        interval_width = scipy.stats.t.interval(0.95, nsims)[1]

        true_positive_array = numpy.zeros(nsamples)
        t_positive_array = numpy.zeros(nsamples)
        lbi = int(round(bootstrap_samples*0.025,0))
        ubi = int(round(bootstrap_samples*0.975,0))

        for i in xrange(nsamples):
            print("\r{:06d}".format(i), end='')
            event_array = scipy.stats.poisson.rvs(simlength*k, size=nsims) 

            k_array = numpy.zeros(bootstrap_samples)
            total_waiting_time = nsims*simlength 
            for j in xrange(bootstrap_samples):
                synthetic_event_array = numpy.random.choice(event_array, size=nsims)
                k_hat = synthetic_event_array.sum()/total_waiting_time
                k_array[j] = k_hat 
            mean_k = k_array.mean()
            k_list.append(mean_k)
            se = k_array.std(ddof=1)
            se_k_list.append(se)
            #print("{:f}:  {:f}".format(mean_k, se))
            if numpy.abs(mean_k - k) < se*interval_width:
                t_positive_array[i] = 1
            k_array.sort()
            if (k_array[lbi] <= k) and (k_array[ubi] >= k):
                true_positive_array[i] = 1 
        se_k_array = numpy.array(se_k_list)
        print("\nMean k from bootstrapping: {:e}"\
              .format(numpy.array(k_list).mean()))
        print('Mean standard error: {:e}'.format(se_k_array.mean()))
        print('Mean standard error, ignoring NaNs: {:e}'\
              .format(numpy.nanmean(se_k_array)))
        if plot:
            self._plot(se_k_array)
        return se_k_array.mean(), true_positive_array.mean(), t_positive_array.mean()

    def _plot(self, se_k_array):
        '''
        Plot and save the observed distribution of standard errors ``se_k_array``
        '''
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

    nsim_list = [10,50,100,300,500,1000,3000,5000,10000]
    # 7160F50pF66m05
    e = BFCIEstimate(3.86*10**5, 3*1.5*200*.000001)
    #e.estimate_2(3*1.5*200, .000001, nsamples=1000, plot=True)
    #e.estimate_2(900, .000001, nsamples=1000, plot=True)
    #e.estimate_2(300, .000003, nsamples=1000, plot=True)
    e.estimate(nsamples=100000, plot=True)
    ys = []
    actual_type1_rates = []
    for nsims in nsim_list:
        simlength = .0009/nsims
        se_k, type1rate = e.estimate_3(nsims, simlength, nsamples=1000)
        ys.append(se_k)
        actual_type1_rates.append(type1rate)
    fig, (ax1, ax2) = pyplot.subplots(2,1)
    ax1.plot(nsim_list, ys)
    ax2.plot(nsim_list, actual_type1_rates)
    ax2.set_ylim(0,1)
    ax1.set_ylim(0,25000)
    pyplot.savefig('test2.pdf')

if __name__ == '__main__':
    main()
