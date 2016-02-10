#!/usr/bin/env python
from __future__ import print_function, division; __metaclass__ = type
import h5py
import logging
import numpy as np
import westpa
from westtools import (WESTTool, WESTDataReader, ProgressIndicatorComponent)

log = logging.getLogger('westtools.w_postanalysis_push')

class ConsistencyError(Exception):
    '''Exception raised for inconsistent input files.'''
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ArgumentError(Exception):
    '''Exception raised for conflicting or improperly used command line 
    arguments.'''
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class WPostanalysisPush(WESTTool):
    prog ='w_postanalysis_push'
    description = '''\
Apply weights calculated using the postanalysis reweighting scheme (see 
"w_postanalysis_reweight --help" or "w_multi_reweight --help") to a WESTPA data
file (usually "west.h5") by scaling the probability in each bin.  Bin
assignments (usually "assign.h5") and a corresponding postanalysis reweighting
output file (usually "kinrw.h5" or "multi_rw.h5") must be supplied, in addition
to the iteration from which to pull weights.

WARNING: Output from this script may not be compatible with some tools. Due to
the nature of the postanalysis reweighting process, some walkers may be assigned
zero weight. Additionally, total weight may not sum to one for some iterations, 
as a bin predicted to have nonzero weight by the postanalysis reweighting scheme
may depopulate during the course a simulation. 

--------------------------------------------------------------------------------
Output format
--------------------------------------------------------------------------------

The output file (-o/--output, usually "west_rw.h5") is of the same format as
the original WESTPA data file. New weights are found in place of the original 
weights, located in:

/iterations/iter_{N_ITER:08d}/seg_index/ 

--------------------------------------------------------------------------------
Command-line options
--------------------------------------------------------------------------------
'''

    def __init__(self):
        super(WPostanalysisPush, self).__init__()
        
        self.data_reader = WESTDataReader()
        self.progress = ProgressIndicatorComponent()
        
        self.output_filename = None
        self.rw_filename = None
        self.assignment_filename = None
        
        self.output_file = None
        self.rw_file = None
        self.assignments_file = None

        self.weights_attributes_initialized = False
        self.weights_already_calculated = False
        self.time_average_scaling_vector_calculated = False

        
    def add_args(self, parser):
        self.progress.add_args(parser)
        self.data_reader.add_args(parser)

        iogroup = parser.add_argument_group('input/output options')
        iogroup.add_argument('-W', '--west', dest='westH5_path', 
                             default='west.h5', metavar='WESTH5',
                             help='''Apply weights to data from WESTH5, creating
                             a new file that either links to or duplicates data
                             in WESTH5 (WESTH5 will not be altered).''')

        iogroup.add_argument('-rw', dest='rw_H5_path', metavar='RW_FILE',
                             default='kinrw.h5',
                             help='''Pull weights from RW_FILE. This should be
                             the output file from either w_postanalysis_reweight 
                             or w_multi_reweight.''')

        iogroup.add_argument('-a', '--assignments', dest='assignH5_path',
                             default='assign.h5', metavar='ASSIGNH5',
                             help='''Rescale weights based on bin assignments
                             in ASSIGNH5. This file should be consistent with 
                             RW_FILE and WESTH5''')

        iogroup.add_argument('-o', '--output', dest='output', 
                             default='west_rw.h5',
                             help='''Store results in OUTPUT 
                             (default: %(default)s).''')

        iogroup.add_argument('-c', '--copy', dest='copy', action='store_true', 
                             help='''If specified, copy all data from WESTH5
                             to OUTPUT.  Otherwise (default), link to data in
                             WESTH5.''')

        cogroup = parser.add_argument_group('calculation options')
        cogroup.add_argument('-n', '--n-iter', dest='n_iter', default=None,
                             type=int,
                             help='''Pull weights from N_ITER and push to data
                             from all iterations in WESTH5.  By default, use
                             the final iteration available in RW_FILE.
                             Alternatively, the weights from each iteration may
                             be push to the data from the corresponding
                             iteration in WESTH5 (see "-e/--evolution-mode"). 
                             ''')

        cogroup.add_argument('-e', '--evolution-mode', action='store_true', 
                             help='''If specified, push weights from each
                             iteration available in RW_FILE to the data from the
                             corresponding iteration in WESTH5. For iterations
                             not available in RW_FILE, copy weights directly 
                             from WESTH5 without rescaling.  By default, pull 
                             weights from a single iteration, specified using 
                             "-n/--n-iter".''')
        
        cogroup.add_argument('-nc', '--no-color', action='store_true', 
                             dest='no_color',
                             help='''If specified, do not use colored bins for
                             rescaling weights. By default, use colored bins.
                             ''')

        cogroup.add_argument('--time-average', action='store_true',
                             dest='time_average',
                             help='''If specifed, scale weights of walkers the
                             total weight of all walkers in all iterations in a
                             given bin sum to the weight given by the 
                             postanalysis reweighting output.  Weights in a
                             given iteration will likely no longer sum to one. 
                             This options is not compatible with evolution mode
                             (see -e/--evolution-mode).''')

        cogroup.add_argument('--iter-range', dest='iter_range', type=str,
                             default=None,
                             help='''This option may be used only if 
                             --time-average is specified.  ITER_RANGE should be
                             a tuple of the form (first_iter, last_iter).  If
                             this option is specified, use only information on 
                             walkers between first_iter and last_iter 
                             (inclusive) when calculating scaling coefficients.
                             However, all iterations in the WESTPA data file
                             will be rescaled according to these weights.''')
 
                              
                                                         
    def process_args(self, args):
        '''Process the arguments defined in ``add_arguments``, making them 
        available as attributes of the main tool class.'''
        self.progress.process_args(args)
        self.data_reader.process_args(args)
       
        # I/O arguments
        self.westH5_path = args.westH5_path
        self.rwH5_path = args.rw_H5_path
        self.assignH5_path = args.assignH5_path
        self.output_path = args.output

        # Calculation arguments
        self.copy = args.copy
        self.n_iter = args.n_iter
        self.evolution_mode = args.evolution_mode

        if args.no_color:
            self.i_use_color = False
        else:
            self.i_use_color = True

        if args.time_average:
            self.i_time_average = True
        else:
            self.i_time_average = False

        if self.i_time_average and self.evolution_mode:
            raise ArgumentError("Error. Time averaging and evolution modes are "
                                "not compatible! See options "
                                "-e/--evolution-mode and --time-average for "
                                "more information.")  
        self.first_iter = None
        self.last_iter = None
        if args.iter_range is not None:
            if not self.i_time_average:
                raise ArgumentError("Error. Specifying an iteration range is "
                                    "only compatible with time-averaging! See "
                                    "--time-average for more information.")
            iter_tuple = eval(args.iter_range)
            try:
                self.first_iter = int(iter_tuple[0]) 
                self.last_iter = int(iter_tuple[1]) 
            except (ValueError, TypeError):
                raise ArgumentError("An error occurred while parsing the "
                                    "supplied iteration range. The iteration "
                                    "range should evaluate to a Python tuple "
                                    "of integers.  Input: ({:s}). Please see "
                                    "the documentation for the --iter-range "
                                    "option for more information."
                                    .format(args.iter_range)                  )
                                    
            print("Using data between iterations {:d} and {:d} for calculation "
                  "of rescaling vector."
                  .format(self.first_iter, self.last_iter)
                  )


    def open_files(self):
        '''Open the WESTPA data file, the reweighting output file, the
        assignments file, and the output file.''' 
        self.westH5 = h5py.File(self.westH5_path, 'r') 
        self.rwH5 = h5py.File(self.rwH5_path, 'r')
        self.assignments = h5py.File(self.assignH5_path, 'r')
        self.output = h5py.File(self.output_path, 'w')
 

    def check_consistency_of_input_files(self):
        '''Check that the assignment file and west.h5 file have the same number
        of walkers for each iteration, and check that the reweighting output 
        file and the assignment file use the same number of bins.'''
        ## First check that the assignment and west.h5 file have the same ##
        ## number of walkers.                                             ##
        # Assume that the assignments file includes data for ALL iterations.
        # At the time of this code's writing, it must, but this may change
        # later.
        last_iter = self.assignments['assignments'].shape[0] # Zero-indexed 
        # Get the total number of bins.  Indices corresponding to walkers not 
        # present in a given iteration will be assigned this integer.
        n_bins = self.assignments['bin_labels'].shape[0] 
        self.pi.new_operation('Checking input files for consistency',
                              last_iter)

        for iiter in xrange(1, last_iter+1): #iiter is one-indexed
            try:
                iter_group = self.westH5['iterations/iter_{:08d}'.format(iiter)]
            except KeyError:
                raise ConsistencyError("Iteration {:d} exists in {:s} but not "
                                       " in {:s}!".format(iiter,
                                                          self.assignH5_path,
                                                          self.westH5_path)    ) 
            # Number of walkers in this iteration, for the westh5 file.
            westh5_n_walkers = iter_group['seg_index'].shape[0]
            # Number of walkers in this iteration, for the assignh5 file. The
            # size of the assignments array is the same for all iterations, and
            # equals the maximum observed number of walkers.  For iterations
            # with fewer than the maximum number of walkers, the rest of the
            # indices are filled with the integer ``n_bins``. 
            # Switch to zero-indexing, and only look at the first time point.
            assign_n_walkers = np.count_nonzero(
              np.array(self.assignments['assignments'][iiter-1][:,0]) != n_bins 
                                                )
            if not westh5_n_walkers == assign_n_walkers:
                raise ConsistencyError("The number of walkers in the WESTPA "
                                       "data file ({:s}, {:d}) and the number "
                                       "of walkers in the assignments file "
                                       "({:s}, {:d}) for iteration {:d} do not "
                                       "match!".format(self.westH5_path,
                                                       westh5_n_walkers,
                                                       self.assignH5_path,
                                                       assign_n_walkers,
                                                       iiter)                  ) 
            self.pi.progress += 1
        ## Now check that the reweighting output file and the assignment file ##
        ## use the same number of bins.                                       ##
        rw_n_bins = self.rwH5['bin_prob_evolution'].shape[1]
        # rw_n_bins is colored, but n_bins (from the assignments file) is not.
        assign_nstates = self.assignments['state_labels'].shape[0]
        rw_nstates = self.rwH5['state_labels'].shape[0]
        if not assign_nstates == rw_nstates:
            raise ConsistencyError("The number of states used in the "
                                   "assignments file ({:d}) does not match the "
                                   "number of states used in the reweighting "
                                   "file ({:d})!".format(assign_nstates, 
                                                         rw_nstates)           )
        if not assign_nstates*n_bins == rw_n_bins:
            raise ConsistencyError("The number of bins used in the assignments "
                                   "file ({:d}) does not match the number of "
                                   "bins used in the reweighting file ({:d})!."
                                   .format(nbins, rw_n_bins/rw_nstates)        )   
        self.pi.clear()

    def initialize_output(self):
        '''Copy or link datasets besides the seg_index datasets from the input 
        WESTPA data file to the output (reweighted) data file. '''
        self.pi.new_operation('Initializing output file',
                              len(self.westH5['iterations'].keys()))
        for key in self.westH5.keys():
             if key != 'iterations':
                 if self.copy:
                     self.westH5.copy(key, self.output)
                 else:
                     self.output[key] = h5py.ExternalLink(self.westH5_path,
                                                          key)
        for name, val in self.westH5.attrs.items():
            self.output.attrs.create(name, val)
         
        self.output.create_group('iterations')
        for key1 in self.westH5['iterations']:
            self.output.create_group('iterations/{:s}'.format(key1))
            for key2 in self.westH5['iterations/{:s}'.format(key1)]:
                if key2 != 'seg_index':
                    key = 'iterations/'+key1+'/'+key2 
                    if self.copy:
                        self.westH5.copy(key, self.output['iterations/'+key1])  
                    else:
                        self.output[key] = h5py.ExternalLink(self.westH5_path,
                                                             key)
            for name, val in self.westH5['iterations/{:s}'.format(key1)].attrs.items():
                self.output['iterations/{:s}'.format(key1)].attrs.create(name, val) 
            self.pi.progress += 1
        self.pi.clear()


    def get_new_weights(self, n_iter):
        '''Generate and return a length-nbins numpy array representing a vector
        of weights, where weights[i] represents the total weight that should be
        in bin i, based on results from the postanalysis reweighting scheme.'''
        # Build map between indexing of 'conditional_flux_evolution' or
        # 'bin_prob_evolution' (the indexing is the same for both) and 
        # weighted ensemble iteration indices
        if not self.weights_attributes_initialized:
            cfe = self.rwH5['conditional_flux_evolution']
            self.idx_map = np.empty(cfe.shape[0], 
                                    dtype=self.assignments['assignments'].dtype)
            for i in xrange(cfe.shape[0]):
                # Axes are (timepoint index, beginning state, ending state)
                # and final index gets the "iter_stop" data
                # stop_iter is exclusive, so subtract one
                self.idx_map[i] = cfe[i,0,0][1]-1 
            self.weights_attributes_initialized = True

        if (not self.evolution_mode) and (not self.weights_already_calculated):
            idx = np.where(self.idx_map == self.n_iter)
            self.new_weights = np.array(self.rwH5['bin_prob_evolution'])[idx]\
                               .squeeze() 
            self.weights_already_calculated = True
            if self.i_use_color:
                return self.new_weights
            else:
                # Convert colored to non-colored vector. self.nstates should
                # have been set by self.go()
                colored_new_weights = np.copy(self.new_weights)
                self.new_weights = np.zeros(
                        int(self.new_weights.shape[0]/self.nstates)
                                            )
                for i in xrange(self.new_weights.shape[0]):
                    self.new_weights[i] = np.sum(
                        colored_new_weights[self.nstates*i:self.nstates*(i+1)]
                                                 )
                return self.new_weights

        elif (not self.evolution_mode) and self.weights_already_calculated:
            return self.new_weights

        else: # if self.evolution mode:
            idx = np.where(self.idx_map == n_iter) 
            new_weights = np.array(self.rwH5['bin_prob_evolution'][idx])\
                          .squeeze()
            if self.i_use_color:
                return new_weights
            else:
                # Convert colored to non-colored vector. self.nstates should
                # have been set by self.go()
                colored_new_weights = new_weights
                new_weights = np.zeros(new_weights.shape[0]/self.nstates)
                for i in xrange(new_weights.shape[0]/self.nstates):
                    new_weights[i] = np.sum(
                      colored_new_weights[self.nstates*i:self.nstates*(i+1)]
                                            )
                return new_weights

    def get_input_assignments(self, iiter):
        '''Get the bin assignments for the first timepoint of iteration
        ``iiter``, for all walkers in the input assignments file. Return a
        1-dimension numpy array ``assignments``, where ``assignments[i]`` gives
        the bin index assigned to walker i.  Bin indices take into account
        whether or not the colored scheme is used.''' 
        # Get the total weight in each bin for the input assignments file
        # Look only at the first time point! -----------------------------V
        assignments = np.array(self.assignments['assignments'][iiter-1][:,0])
        if self.i_use_color:
            traj_labels = np.array(self.assignments['trajlabels'][iiter-1][:,0])
            assignments = assignments*nstates+traj_labels
        return assignments
               

    def calculate_scaling_coefficients(self, iiter):
        '''Calculate and return a vector of scaling coefficients for 
        the weighted ensemble iteration ``iiter``.  scaling_coefficients[i] 
        gives the value by which to scale (multiply) the weight of any walker 
        in bin i.  

        This method first calls self.get_new_weights(iiter) to find what the
        weights output by the postanalysis reweighting scheme are.  Next, it 
        considers where time averaging is enabled, and whether or not the 
        colored scheme is used.  Finally, it calculates the input assignments
        if necessary.'''
        if not self.i_time_average:
            # Get length-n_bins vector, where new_weights[i] is the total
            # weight in bin i according to the postanalysis reweighting 
            # scheme.
            new_weights = self.get_new_weights(iiter)
            input_assignments = self.get_input_assignments(iiter)
            input_iter_group = self.westH5['iterations/iter_{:08d}'
                                           .format(iiter)]
            seg_weights = np.array(input_iter_group['seg_index']['weight'])
            
            # Calculate the weight in each bin.  the assignments should already
            # have similar information in 'labeled_populations', but these weights
            # invlude ALL time points in each iteration.  In this tool, we must
            # scale the weight of each segment uniformly across any given weighted
            # ensemble iteration, as the weight is only specifed once per iteration.
            # Somewhat arbitrarily, we scale the weights only according to the
            # first timepoint only (we could also choose the another timepoint, or
            # average them perhaps).  For this reason, we need to re-calculate the
            # labelled populations, only looking at the first time point.
            # New weights will already be the correct length (ie, adjusted 
            # for color/no color).
            input_weights = np.zeros(new_weights.shape)
            # Calculate the weight in each bin, in the input WESTPA data
            # file.  This is necessary because w_assign does not calculate
            # populations with color labels.
            for i in xrange(input_weights.shape[0]):
                input_weights[i] = np.sum(
                        seg_weights[np.where(input_assignments == i)] 
                                          )
            # Suppress division errors
            with np.errstate(all='ignore'):
                scaling_coefficients = new_weights/input_weights
                # Set nonsensical value to zero; is this really necessary?
                #scaling_coefficients[~np.isfinite(scaling_coefficients)] = 0
        else: # if self.i_time_average
            if self.time_average_scaling_vector_calculated:
                return self.time_average_scaling_vector
            else: # if not self.time_average_scaling_vector_calculated:
                # Calculate the scaling vector  
                # old_weights will contain the total weight observed in each 
                # bin during the FIRST timepoint of all iterations; this must
                # be calculated here rather than using labeled_populations from
                # the assignments file, as labeled_populations includes all
                # timepoints
                new_weights = self.get_new_weights(iiter)
                old_weights = np.zeros(new_weights.shape)
                if self.first_iter is None:
                    iter_strs = self.westH5['iterations/'].keys().sort()
                    self.first_iter = int(iter_strs[0][5:])
                    self.last_iter_iter = int(iter_strs[-1][5:])
                # Iterate over all WE iterations and sum up the weight in each
                # bin
                for iiter in xrange(self.first_iter, self.last_iter+1):
                    weights = np.array(
                            self.westH5['iterations/iter_{:08d}/seg_index'
                                        .format(iiter)]['weight']
                                       )
                    assignments = self.get_input_assignments(iiter) 
                    for bin_idx in xrange(old_weights.shape[0]):
                        where = np.where(assignments == bin_idx)
                        old_weights[bin_idx] += np.sum(weights[where])
                with np.errstate(all='ignore'):
                    self.time_average_scaling_vector = new_weights/old_weights
                    self.time_average_scaling_vector[
                            ~np.isfinite(self.time_average_scaling_vector)
                                                     ] = 0.0
                self.time_average_scaling_vector_calculated = True
                return self.time_average_scaling_vector
                    

    def go(self):
        '''
        Main function. Calls:
          - self.open_files()
          - self.check_consistency_of_input_files()
          - self.initialize_output()
        and then iterates through all weighted ensemble iterations, rescaling 
        weights of segments and saving a new ``seg_index`` dataset in the output
        file using the rescaled weights.
        '''
        pi = self.progress.indicator
        with pi as self.pi:
            
            # Open files
            self.open_files()
            
            # Check files for consistency
            self.check_consistency_of_input_files() 

            # Initialize the output file.
            self.initialize_output()

            last_iter = self.assignments['assignments'].shape[0] 
            self.nstates = len(self.assignments['state_labels'])

            # If weights are to be pulled from a single iteration, get the weights 
            pi.new_operation('Creating new WESTPA data file with scaled '
                             'weights.', last_iter)
            for iiter in xrange(1, last_iter+1): # iiter is one-indexed 
                scaling_coefficients = self.calculate_scaling_coefficients(iiter)
                input_iter_group = self.westH5['iterations/iter_{:08d}'
                                               .format(iiter)]
                input_assignments = self.get_input_assignments(iiter) 
                # Get the HDF5 group for this iteration. It was already created
                # while initializing the output file.
                output_iter_group = self.output['iterations/iter_{:08d}'
                                                .format(iiter)]

                input_seg_index = np.array(input_iter_group['seg_index']) 
                seg_weights = input_seg_index['weight']
                # Build the new seg_index piece by piece.  Start with an empty
                # list and add data for one segment at a time. Then convert to 
                # a numpy array.
                output_seg_index = []
                for iseg in xrange(input_seg_index.shape[0]):
                    # Only look at the first time point for assignments!
                    bin_idx = input_assignments[iseg]
                    coeff = scaling_coefficients[bin_idx] 
                    output_seg_index.append((seg_weights[iseg]*coeff,)
                                             + tuple(input_seg_index[iseg])[1:])
                output_seg_index = np.array(output_seg_index, 
                                            dtype=input_seg_index.dtype)
                # Save the newly created seg_index (with new weights)
                output_iter_group.create_dataset('seg_index',
                                                 data=output_seg_index,
                                                 dtype=output_seg_index.dtype)
                self.pi.progress += 1
                         
            
                
if __name__ == '__main__':
    WPostanalysisPush().main()
