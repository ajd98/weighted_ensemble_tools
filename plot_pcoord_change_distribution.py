#!/usr/bin/env python
# By Alex DeGrave, July 15, 2015.
import argparse 
import h5py
import math
import matplotlib.pyplot as pyplot
import numpy

'''
Calculate and plot the distribution of changes in the progress coordinate
value for each segment of each iteration for a weighted ensemble simulation.
Plot probability as a function of change in the progress coordinate value and
the initial progress coordinate value.
'''

class PCoordAnalyzer:
    def __init__(self):
        self._parse_args()
        self.westh5 = h5py.File(self.args.westh5_path) 
        self._get_iter_range() 
        self._scan_iter_range()
        self.value_array = numpy.zeros((len(self.iter_range),
                                        self.MAX_SEG_COUNT,
                                        2))
        self.weight_array = numpy.zeros((len(self.iter_range),
                                         self.MAX_SEG_COUNT,
                                         1))                                
        self.pcoord_index = self.args.pcoord_index
    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-W', dest='westh5_path', default='west.h5',
                            help='The west.h5 file in which the simulation '
                                 'stored progress coordinates.')
        parser.add_argument('--first-iter', dest='first_iter', default=1,
                            help='Use data starting from the specified '
                                 'iteration.',
                            type=int)
        parser.add_argument('--last-iter', dest='last_iter', default=None,
                            help='Use data up to and including the specified '
                                 'iteration.',
                            type=int)
        parser.add_argument('--pcoord-index', dest='pcoord_index', default=0,
                            help='Plot data using the specified pcoord.',
                            type=int)
        parser.add_argument('--binexpr', dest='binexpr', default=None,
                            help='A Python expression for a list describing the'
                                 ' bin boundaries that the simulation used '
                                 'along the specified progress coordinate '
                                 ' dimension.',
                            type=str) 
        parser.add_argument('--ybins', dest='ybins', default=100,
                            help='The number of bins to use in the y direction '
                                 '(which describes the change in progress ' 
                                 'coordinate value for each iteration.',
                            type=int)
        parser.add_argument('--ybinexpr', dest='ybinexpr', default=None,
                            help='A Python expression for a list describing the'
                                 ' bins boundaries to use along the y axis, '
                                 'which describes the change in progress '
                                 'coordinate value for each iteration.',
                            type=str)
        parser.add_argument('--mode', dest='mode', default='plot_values',
                            help='MODE should either be ``plot_values`` or '
                                 '``plot_delta``.  The mode ``plot_values`` '
                                 'will plot a two-dimensional histogram with '
                                 'the x-axis as the starting progress ' 
                                 'coordinate value for an iteration, and the ' 
                                 'y-axis as the ending progress coordinate '
                                 'value for an iteration. The mode '
                                 '``plot_delta`` will plot a two-dimensional '
                                 'histogram with the x-axis as the starting '
                                 'progress coordinate value, and the y axis ' 
                                 'as the change in the progress coordinate '
                                 'value over the iteration.',
                            type=str)
        parser.add_argument('--counts', dest='counts', default=False,
                            help='Plot walker counts rather than probabilties.'
                                 ' Color values denote the average i -> j '
                                 'transition count per iteration.',
                            type=bool)
        self.args = parser.parse_args()
        if not (self.args.mode == 'plot_values' or
                self.args.mode == 'plot_delta'):
            if self.args.mode is None:
                print('Please specify a plotting mode using the '
                      '``--mode flag``. See ``--help`` for more information.')
            else:
                print('Please specify a valid plotting mode. You specified '
                      '``%s``.  See ``--help`` for valid options.'\
                      %self.args.mode)
            exit(1)
        if self.args.binexpr is None:
            print('Please specify a bin expression for the bins that the '
                  'simulation used for propagation along the specified '
                  'progress coordinate axis (the default progress coordinate '
                  'axis is ``0``).')
            exit(1)
    def _get_iter_range(self):
        first_iter = self.args.first_iter
        if self.args.last_iter is not None:
            last_iter = self.args.last_iter
        else:
            last_iter = self.westh5.attrs['west_current_iteration'] - 1 
        self.iter_range = range(first_iter,last_iter+1)
    def _format_iter_str(self,val):
        return "iter_%08d"%val
    def _scan_iter_range(self):
        '''
        Scans the specified range of iterations for the max number of segments.
        Sets ``self.MAX_SEG_COUNT`` to this value.
        '''
        m = 0
        for iiter in self.iter_range:
            iter_seg_count = self.westh5['iterations/' +\
                                         self._format_iter_str(iiter) +\
                                         '/seg_index'].shape[0] 
            if iter_seg_count > m:
                m = iter_seg_count 
        self.MAX_SEG_COUNT = m 
    def _get_bins_from_expr(self, binexpr):
        '''
        Converts ``binexpr`` from a string to a python iteratable.  This method
        makes the ``numpy`` and ``math`` libraries available for evaluation of
        the string.  Use ``inf`` to access ``float('inf')``. 
        '''
        self.binexpr_namespace = {'numpy': numpy,
                                  'math':math,
                                  'inf':float('inf')}
        bins_ = eval(binexpr,self.binexpr_namespace)
        return bins_
    def _calculate_delta_pcoord(self):
        '''
        Crawl over the specified iteration range and find the starting 
        progress coordinate values and change in progress coordinate values
        for each segment. 
        '''
        for index, iiter in enumerate(self.iter_range):
            iter_group = self.westh5['iterations/' +\
                                     self._format_iter_str(iiter)] 
            # Extract the segments' weights for the current iteration.
            weights = numpy.array(iter_group['seg_index']['weight']) 

            # Extract progress coordinate values for the current iteration. 
            pcoords = numpy.array(iter_group['pcoord'][:,:,self.pcoord_index]) 
            pcoord_starts  = pcoords[:,0]
            pcoord_ends    = pcoords[:,-1]
            pcoord_changes = pcoord_ends - pcoord_starts

            # Store the starting progress coordinate value,
            # change in the progress coordinate value over the iteration,
            # and weights of each segment in ``self.value_array`` and
            # ``self.weight_array``.
            self.value_array[index, :weights.shape[0],0] = pcoord_starts 
            self.value_array[index, :weights.shape[0],1] = pcoord_changes
            if self.args.counts is True:
                self.weight_array[index, :weights.shape[0], 0] = 1
            else:
                self.weight_array[index, :weights.shape[0],0] = weights 

    def _calculate_pcoord_vs_pcoord(self):
        '''
        Crawl over the specified iteration range and find the starting 
        progress coordinate values and ending progress coordinate values
        for each segment. 
        '''
        for index, iiter in enumerate(self.iter_range):
            iter_group = self.westh5['iterations/' +\
                                     self._format_iter_str(iiter)] 
            # Extract the segments' weights for the current iteration.
            weights = numpy.array(iter_group['seg_index']['weight']) 

            # Extract progress coordinate values for the current iteration. 
            pcoords = numpy.array(iter_group['pcoord'][:,:,self.pcoord_index]) 
            pcoord_starts  = pcoords[:,0]
            pcoord_ends    = pcoords[:,-1]

            # Store the starting progress coordinate value,
            # ending progress coordinate value, 
            # and weights of each segment in ``self.value_array`` and
            # ``self.weight_array``.
            self.value_array[index, :weights.shape[0], 0] = pcoord_starts 
            self.value_array[index, :weights.shape[0], 1] = pcoord_ends
            if self.args.counts is True:
                self.weight_array[index, :weights.shape[0], 0] = 1 
            else:
                self.weight_array[index, :weights.shape[0], 0] = weights 

    def _histogram(self):
        '''
        Histogram ``self.value_array`` using the weights from 
        ``self.weight_array``.  Store output in ``self.H`` (the histogram),
        ``self.xedges (bin boundaries along the x dimension), and 
        ``self.yedges`` (bin boundaries along the y dimension).
        Take the bin boundaries for the first dimension of the histogram from 
        the xbinexpr command line argument. Take the bin boundaries for the 
        second dimension of the histogram from the ybinexpr command line 
        argument (if the user specified it).  Otherwise, get the bin boundaries
        for the y dimension by dividing up the the range between the maximum 
        and minimum values observed in ``self.value_array[:,:,1]`` into a 
        number of bins. This function gets this number of bins from the command
        line argument ``--ybins``, if the user specified a number.  Otherwise,
        the number should default to 100.
        '''
        self.xbins = self._get_bins_from_expr(self.args.binexpr)
        if self.args.mode == 'plot_delta':
            if self.args.ybinexpr is not None:
                self.ybins = self._get_bins_from_expr(self.args.ybinexpr)
            else:
                y_min = self.value_array[:,1].min()
                y_max = self.value_array[:,1].max()
                y_range = y_max - y_min  
                y_bin_count = self.args.ybins 
                delta = float(y_range)/y_bin_count
                self.ybins = [y_min + delta*i for i in range(y_bin_count+1)] 
        if self.args.mode == 'plot_values':
            self.ybins = self.xbins
        bins_ = [numpy.array(self.xbins), numpy.array(self.ybins)] 
        xvals = self.value_array[:,:,0][...].ravel()
        yvals = self.value_array[:,:,1][...].ravel()
        weights_ = self.weight_array.ravel()
        H, xedges, yedges = numpy.histogram2d(xvals,
                                              yvals,
                                              bins=bins_, 
                                              weights=weights_)
        # Normalize along so each column along the x axis sums to 1.
        # In other words, the total probabilty per propagation bin is ``1``.
        column_sums = H.sum(axis=1)

        # If the user specified to use the ``counts`` mode, then normalize by 
        # dividing by the total number of iterations.  This gives the average 
        # transition count per iteration.
        if self.args.counts is True:
            column_sums[:] = len(self.iter_range) 
        normed_H = H / column_sums[:, numpy.newaxis]

        # Take care of ``nan`` and ``inf`` values that could arise if a column
        # sums to zero.
        normed_H[H==0] = 0

        # Save the histogram and its edges as an attribute.
        self.H = normed_H
        self.xedges = xedges
        self.yedges = yedges
    def _plot_histogram(self):
        '''
        Plot the histogram from ``self._histogram``, stored as ``self.H``,
        using values from ``self.xedges`` and ``self.yedges`` to specify the 
        extent of the histogram. 
        '''
        pyplot.pcolormesh(self.xedges, 
                          self.yedges, 
                          self.H.transpose(), 
                          cmap='hot', 
                          edgecolors='None', 
                          vmin=0, 
                          vmax=1,
                          rasterized=True)


    def _plot_bins_delta_pcoord(self):
        for binbound in self.xbins:
            pyplot.axvline(binbound,color='gray',linewidth=.5)
        num_bins = len(self.xbins)
        # Hack to add on a reasonable legend.
        # The first time these lines plot a line of a certain color, they 
        # add a label, so that information about this line's color will appear
        # in a legend.  Subsequent lines this function plots in the same color
        # do not recieve a label.  This prevents the same color from appearing
        # many times in the legend.

        # For the cyan lines
        first_one = True

        # For the violet lines
        first_two = True

        # Iterate over each of the bins.
        for i in range(num_bins - 1):
            if i < (num_bins - 2):
                xs = [self.xbins[i], self.xbins[i+1]]
                y = self.xbins[i+1] - self.xbins[i]  
                ys = [y,y] 
                if first_one:
                    pyplot.plot(xs, ys, color='cyan', linewidth=2, 
                                label='Boundary of current bin')
                    first_one = False
                else:
                    pyplot.plot(xs, ys, color='cyan', linewidth=2)
            if i < (num_bins - 3):
                xs = [self.xbins[i], self.xbins[i+1]]
                y = self.xbins[i+2] - self.xbins[i]  
                ys = [y,y] 
                if first_two:
                    pyplot.plot(xs, ys, color='violet', linewidth=2, 
                                label='Boundary of adjacent bin')
                    first_two = False
                else:
                    pyplot.plot(xs, ys, color='violet', linewidth=2)
            if i > 0:
                xs = [self.xbins[i], self.xbins[i+1]]
                y = self.xbins[i-1] - self.xbins[i]
                ys = [y,y] 
                pyplot.plot(xs, ys, color='cyan', linewidth=2)
            if i > 1:
                xs = [self.xbins[i], self.xbins[i+1]]
                y = self.xbins[i-2] - self.xbins[i]
                ys = [y,y] 
                pyplot.plot(xs, ys, color='violet', linewidth=2)

    def _plot_bins_pcoord_vs_pcoord(self):
        for ibin, binbound in enumerate(self.xbins[:-1]):
            lower = binbound
            upper = self.xbins[ibin+1]
            pyplot.vlines(x=lower,
                          ymin=lower,
                          ymax=upper,
                          color='cyan')
            print("plotting line at x=%f from ymin=%f to ymax=%f"%(lower,lower,upper))
            pyplot.vlines(x=upper,
                          ymin=lower,
                          ymax=upper,
                          color='cyan')
            pyplot.hlines(y=lower,
                          xmin=lower,
                          xmax=upper,
                          color='cyan')
            pyplot.hlines(y=upper,
                          xmin=lower,
                          xmax=upper,
                          color='cyan')

    def _set_plot_style_delta_pcoord(self):
        pyplot.xlim((self.xedges[0],self.xedges[-1]))
        pyplot.ylim((self.yedges[0],self.yedges[-1]))
        cbar = pyplot.colorbar()
        cbar.set_label("Probability, Normalized to 1 for Each \nStarting Progress Coordinate Bin")
        pyplot.xlabel('Progress Coordinate Value at Beginning of Iteration')
        pyplot.ylabel('Change in Progress Coordinate Value Over Iteration') 
        #ax = pyplot.subplot(111)
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0,
        #                 box.width, box.height * 0.9])
        #pyplot.legend(loc='upper center',bbox_to_anchor=(0.5,1.25),
        #        fancybox=True, ncol=2,prop={'size':6})

    def _set_plot_style_pcoord_vs_pcoord(self):
        pyplot.xlim((self.xedges[0],self.xedges[-1]))
        pyplot.ylim((self.yedges[0],self.yedges[-1]))
        cbar = pyplot.colorbar()
        cbar.set_label("Probability, Normalized to 1 for Each \nStarting Progress Coordinate Bin")
        pyplot.xlabel('Progress Coordinate Value at Beginning of Iteration')
        pyplot.ylabel('Progress Coordinate Value at End of Iteration') 

    def _do_delta_pcoord_plot(self):
        self._calculate_delta_pcoord()
        self._histogram()
        self._plot_histogram()
        self._plot_bins_delta_pcoord()
        self._set_plot_style_delta_pcoord()

    def _do_pcoord_vs_pcoord_plot(self):
        self._calculate_pcoord_vs_pcoord()
        self._histogram()
        self._plot_histogram()
        self._plot_bins_pcoord_vs_pcoord()
        self._set_plot_style_pcoord_vs_pcoord()

    def run(self):
        '''
        Main public method of the PCoordAnalyzer class. Use to analyze progress
        coordinate data from the specified HDF5 file, histogram, and plot. 
        '''
        if self.args.mode == 'plot_values':
            self._do_pcoord_vs_pcoord_plot()
        if self.args.mode == 'plot_delta':
            self._do_delta_pcoord_plot()
        pyplot.savefig('test.pdf')
        #self._calculate()
        #self._histogram()
        #self._plot_histogram()
        #self._plot_bins()
        #self._set_plot_style()
         
def main():
    pcoordanalyzer = PCoordAnalyzer()
    pcoordanalyzer.run()   

if __name__ == '__main__':
    main()
