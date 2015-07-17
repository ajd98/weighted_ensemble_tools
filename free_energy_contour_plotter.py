#!/usr/bin/env python
# By Alex DeGrave, July 16 2015
import numpy
import argparse
import pylab

'''
Class to plot contour maps of free energy
'''

class ContourPlotter:
    def __init__(self):
        self._parse_args()

    def _parse_args(self):
        '''
        Parse command line arguments.
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-coords',
                            dest='coords',
                            help='The file containing input coordinates. '
                                 'It should be ``numpy.loadtxt``-able.',
                            type=str)
        parser.add_argument('--xbins', default=100,
                            dest='xbins',
                            help='The number of bins to use for histogramming '
                                 'along the x-axis. Divide the range between '
                                 'the minimum and maximum observed x-value '
                                 'into this many bins',
                            type=int)
        parser.add_argument('--xbinexpr', default=None,
                            dest='xbinexpr',
                            help='A Python expression for a list of bin edges '
                                 'to use along the x-axis.',
                            type=str)
        parser.add_argument('--ybins', default=100,
                            dest='ybins',
                            help='The number of bins to use for histogramming' 
                                 'along the y-axis.  Divide the range between '
                                 'the minimum and maxium observed y-value '
                                 ' into this many bins.',
                            type=int)
        parser.add_argument('--ybinexpr', default=None,
                            dest='ybinexpr',
                            help='A Python expression for a list of bin edges '
                                 'to use along the y-axis.',
                            type=str)
        parser.add_argument('--Q-x', default=None,
                            dest='xcontact_count',
                            help='By specifying an integer for XCONTACT_COUNT,'
                                 ' this setting will automatically generate '
                                 'bins along the x-axis.  This assumes that '
                                 'the x-axis contains fraction of native '
                                 'contacts data, and that a total of '
                                 'YCONTACT_COUNT native contacts are possible.'
                                 ' Uses one bin per native contact.',
                            type=int)
        parser.add_argument('--Q-y', default=None,
                            dest='ycontact_count',
                            help='By specifying an integer for YCONTACT_COUNT,'
                                 ' this setting will automatically generate '
                                 'bins along the y-axis.  This assumes that '
                                 'the y-axis contains fraction of native '
                                 'contacts data, and that a total of '
                                 'YCONTACT_COUNT native contacts are possible.'
                                 ' Uses one bin per native contact.',
                            type=int)
        parser.add_argument('--xlabel', default=None,
                            dest='xlabel',
                            help='A string to use as the label for the x-axis.'
                                 ' The x-axis with be the zero-th dimension of '
                                 'the dataset you supply.',
                            type=str)
        parser.add_argument('--ylabel', default=None,
                            dest='ylabel',
                            help='A string to use as the label for the y-axis.'
                                 ' The y-axis with be the first dimension of '
                                 'the dataset you supply.',
                            type=str)
        parser.add_argument('--output', default=None,
                            dest='output_path',
                            help='The filename to which the plot will be saved.'
                                 ' Various image formats are available.  You ' 
                                 'may choose one by specifying an extension',
                            type=str)
        parser.add_argument('--title', default=None,
                            dest='title',
                            help='Add ``TITLE`` as a title for the plot.',
                            type=str)
        parser.add_argument('--cmap', default='hot_r',
                            dest='cmap',
                            help='The colormap to use for the z-axis of the ' 
                                 'contour plot. Ex: ``hot``,``jet``,``RdBu``. '
                                 'See MatPlotLib documentation for more '
                                 'options.',
                            type=str)
        self.args = parser.parse_args()

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

    def _make_bin_bounds(self):
        '''
        Set the attributes ``xbins`` and ``ybins`` by parsing the command line
        arguments ``xbins``, ``ybins``, ``xbinexpr``, ``ybinexpr``, 
        ``xcontact_count``, and ``ycontact_count``. If the user specified 
        ``xbinexpr`` or ``ybinexpr``, these take precedence over ``xbins`` and 
        ``ybins``. Similarly, ``xcontact_count`` and ``ycontact_count`` take 
        precedence over ``xbinexpr`` and ``ybinexpr``.
        '''
        if self.args.xcontact_count is not None:
            delta = 1./self.args.xcontact_count
            self.xbins=numpy.arange(-delta/2,1+delta,delta)
        elif self.args.xbinexpr is not None:
            self.xbins = self._get_bins_from_expr(self.args.xbinexpr)
        else:
            minx = self.data[:,0].min()
            maxx = self.data[:,0].max()
            r = maxx - minx
            delta = r/self.args.xbins
            self.xbins = numpy.arange(minx,maxx+r,r)
        if self.args.xcontact_count is not None:
            delta = 1./self.args.ycontact_count
            self.ybins=numpy.arange(-delta/2,1+delta,delta)
        elif self.args.ybinexpr is not None:
            self.ybins = self._get_bins_from_expr(self.args.ybinexpr)
        else:
            miny = self.data[:,0].min()
            maxy = self.data[:,0].max()
            r = maxy - miny
            delta = r/self.args.ybins
            self.ybins = numpy.arange(miny,maxy+r,r)

    def _load_from_text_file(self):
        '''
        Load data from a text file.  This is just a wrapper for the 
        ``loadtxt`` method of the main numpy module.
        '''
        self.data = numpy.loadtxt(self.args.coords)

    def _histogram(self):
        '''
        Call ``_make_bin_bounds`` to generate bin bounds, and then use these
        bins to histogram the data in ``self.data``.  Normalize the data so 
        that the histogram is an estimate of an underlying probability density
        function (the histogram should integrate to 1). Save the resulting data
        in the attributes ``self.H`` (a histogram), ``self.xedges``, and 
        ``self.yedges`` (bin boundaries for the x and y axes).
        '''
        self._make_bin_bounds()
        H, xedges, yedges = numpy.histogram2d(self.data[:,0], self.data[:,1], 
                                              bins=[self.xbins,self.ybins],
                                              normed=True)
        self.H = H
        self.xedges = xedges
        self.yedges = yedges

    def _edges_to_midpoints(self):
        '''
        Get the midpoints of bins stored in ``self.xedges`` and ``self.yedges``.
        Store midpoints in ``self.x_mids`` and ``self.y_mids``.
        '''
        self.x_mids = [(self.xedges[i]+self.xedges[i+1])/2\
                       for i in range(self.xedges.shape[0] -1)]
        self.y_mids = [(self.yedges[i]+self.yedges[i+1])/2\
                       for i in range(self.yedges.shape[0] -1)]

    def _make_energy_contour_plot(self):
        '''
        Make a contour plot of free energy using the probabilities stored in
        ``self.H``.
        '''
        X = self.x_mids
        Y = self.y_mids

        # Convert to free energy
        Z = -1*numpy.log(self.H)

        # Make everything relative to minimum, at zero.
        Z -= Z.min()

        # Take care of 'inf' values.
        Z[self.H==0] = float('nan') 

        # Contour levels to plot.  
        levels = numpy.arange(0,5.5,.5) 

        # Specify a color map.
        cmap = pylab.cm.cmap_d[self.args.cmap]

        # Plot contours.
        p = pylab.contourf(X, Y, Z.transpose(), levels, 
                           cmap=pylab.cm.get_cmap(cmap, len(levels)-1)
                           )
        # Add a colorbar
        cbar = pylab.colorbar()

        # Add a label to the colorbar.
        cbar.set_label('$\Delta G/k_BT$')

    def _add_plot_labels(self):
        if self.args.xlabel is not None:
            pylab.xlabel(self.args.xlabel)
        if self.args.ylabel is not None:
            pylab.ylabel(self.args.ylabel)

    def run(self):
        self._load_from_text_file()
        self._histogram()
        self._edges_to_midpoints()
        self._make_energy_contour_plot()
        self._add_plot_labels()
        pylab.savefig(self.args.output_path)

if __name__ == "__main__":
    contourplotter = ContourPlotter()
    contourplotter.run()
