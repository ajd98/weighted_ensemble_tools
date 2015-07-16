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
        parser.add_argument('--first-iter',dest='first_iter',default=1,
                            help='Use data starting from the specified '
                                 'iteration.',
                            type=int)
        parser.add_argument('--last-iter',dest='last_iter', default=None,
                            help='Use data up to and including the specified '
                                 'iteration.',
                            type=int)
        parser.add_argument('--pcoord-index',dest='pcoord_index', default=0,
                            help='Plot data using the specified pcoord.',
                            type=int)
        parser.add_argument('--binexpr',dest='binexpr',default=None,
                            help='A Python expression for a list describing the'
                                 ' bin boundaries that the simulation used '
                                 'along the specified progress coordinate '
                                 ' dimension.',
                            type=str) 
        parser.add_argument('--ybins',dest='ybins',default=100,
                            help='The number of bins to use in the y direction '
                                 '(which describes the change in progress ' 
                                 'coordinate value for each iteration.',
                            type=int)
        parser.add_argument('--ybinexpr',dest='ybinexpr',default=None,
                            help='A Python expression for a list describing the'
                                 ' bins boundaries to use along the y axis, '
                                 'which describes the change in progress '
                                 'coordinate value for each iteration.',
                            type=str)
        self.args = parser.parse_args()
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
    def _calculate(self):
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
            self.value_array[index][:weights.shape[0],0] = pcoord_starts 
            self.value_array[index][:weights.shape[0],1] = pcoord_changes
            self.weight_array[index][:weights.shape[0],0] = weights 
    def _histogram(self):
        '''
        Histogram ``self.value_array`` using the weights from 
        ``self.weight_array``.  Store output in ``self.H`` (the histogram),
        ``self.xedges (bin boundaries along the x dimension), and 
        ``self.yedges`` (bin boundaries along the y dimension).
        '''
        self.xbins = self._get_bins_from_expr(self.args.binexpr)
        if self.args.ybinexpr is not None:
            self.ybins = self._get_bins_from_expr(self.args.ybinexpr)
        else:
            y_min = self.value_array[:,1].min()
            y_max = self.value_array[:,1].max()
            y_range = y_max - y_min  
            y_bin_count = self.args.ybins 
            delta = float(y_range)/y_bin_count
            self.ybins = [y_min + delta*i for i in range(y_bin_count+1)] 
        bins_ = [numpy.array(self.xbins), numpy.array(self.ybins)] 
        print(bins_)
        xvals = self.value_array[:,:,0][...].ravel()
        yvals = self.value_array[:,:,1][...].ravel()
        weights_ = self.weight_array.ravel()
        H, xedges, yedges = numpy.histogram2d(xvals,
                                              yvals,
                                              bins=bins_, 
                                              weights=weights_)
        # Normalize along so each column along the x axis sums to 1.
        # In other words, the total probabilty per propagation bin is ``1``.
        column_sums = H.sum(axis=0)
        normed_H = H / column_sums[numpy.newaxis, :]
        #normed_H[:,column_sums == 0] = 0 
        normed_H[H==0] = 0
        self.H = normed_H
        self.xedges = xedges
        self.yedges = yedges
    def _plot_histogram(self):
        '''
        Plot the histogram from ``self._histogram``, stored as ``self.H``,
        using values from ``self.xedges`` and ``self.yedges`` to specify the 
        extent of the histogram. 
        '''
        #extent_ = [self.xbins[0], self.xbins[-1], self.ybins[0], self.ybins[-1]]
        #extent_ = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]
        #asp = float(extent_[1]-extent_[0])/(extent_[3]-extent_[2])
        pyplot.pcolormesh(self.xedges, 
                          self.yedges, 
                          self.H.transpose(), 
                          cmap='hot', 
                          edgecolors='None', 
                          vmin=0, 
                          vmax=1,
                          rasterized=True)


        #pyplot.imshow(self.H.transpose(), 
        #              extent=extent_,
        #              origin='lower', 
        #              aspect=asp, 
        #              cmap='hot',
        #              interpolation='None')
        #pyplot.savefig('pcoord_analysis.pdf')
    def _plot_bins(self):
        for binbound in self.xbins:
            pyplot.axvline(binbound,color='gray',linewidth=.5)
        num_bins = len(self.xbins)
        # Hack to add on a reasonable legend.
        first_one = True
        first_two = True
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
    def _set_plot_style(self):
        pyplot.xlim((self.xedges[0],self.xedges[-1]))
        pyplot.ylim((self.yedges[0],self.yedges[-1]))
        cbar = pyplot.colorbar()
        cbar.set_label("Probability")
        pyplot.xlabel('Progress Coordinate Value at Beginning of Iteration')
        pyplot.ylabel('Change in Progress Coordinate Value Over Iteration') 
        ax = pyplot.subplot(111)
        box = ax.get_position()
        #ax.set_position([box.x0, box.y0,
        #                 box.width, box.height * 0.9])
        pyplot.legend(loc='upper center',bbox_to_anchor=(0.5,1.25),
                fancybox=True, ncol=2,prop={'size':6})

    def run(self):
        '''
        Main public method of the PCoordAnalyzer class. Use to analyze progress
        coordinate data from the specified HDF5 file, histogram, and plot. 
        Calls the ``_calculate``, ``_histogram`` and ``_plot`` methods of the 
        ``self`` class.  
        '''
        self._calculate()
        self._histogram()
        self._plot_histogram()
        self._plot_bins()
        self._set_plot_style()
        pyplot.savefig('test.pdf')
         
def main():
    pcoordanalyzer = PCoordAnalyzer()
    pcoordanalyzer.run()   

if __name__ == '__main__':
    main()
