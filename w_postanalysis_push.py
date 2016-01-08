#!/usr/bin/env python
from __future__ import print_function, division; __metaclass__ = type
import logging

import numpy as np
import h5py

import westpa
from west.data_manager import weight_dtype, n_iter_dtype
from westtools import (WESTTool, WESTDataReader, IterRangeSelection,
                       ProgressIndicatorComponent)
from westpa import h5io
from westtools.dtypes import iter_block_ci_dtype as ci_dtype

log = logging.getLogger('westtools.w_multi_reweight')

class ConsistencyError(Exception):
    '''Exception raised for inconsistent input files.'''
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

-----------------------------------------------------------------------------
Output format
-----------------------------------------------------------------------------

The output file (-o/--output, usually "west_rw.h5") is of the same format as
the original WESTPA data file. 

-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
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
                self.idx_map[i] = cfe[i,0,0][1] # Last iteration included in this
                                           # averaging window
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
                for i in xrange(int(self.new_weights.shape[0]/self.nstates)):
                    self.new_weights[i] = np.sum(
                        colored_new_weights[self.nstates*i:self.nstates*(i+1)]
                                                 )
                return self.new_weights

        elif (not self.evolution_mode) and self.weights_already_calculated:
            return self.new_weights

        else: # if self.evolution mode:
            # If self.evolution mode is True, trap calls to this function
            # inside the "while True" statement.
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
                # Get length-n_bins vector, where new_weights[i] is the total
                # weight in bin i according to the postanalysis reweighting 
                # scheme.
                new_weights = self.get_new_weights(iiter)
                # Get the total weight in each bin for the input assignments file
                # Only look at the first time point!---------------------------|
                # First get non-colored assignments                            V
                input_assignments = self.assignments['assignments'][iiter-1][:,0]
                # Get colored assignments if necessary
                if self.i_use_color:
                    state_assignments = self.assignments['trajlabels'][iiter-1][:,0]
                    input_assignments = input_assignments*self.nstates+\
                                        state_assignments 
                input_iter_group = self.westH5['iterations/iter_{:08d}'
                                               .format(iiter)]
                seg_weights = np.array(input_iter_group['seg_index']['weight'])
                # New weights will already be the correct length (ie, adjusted 
                # for color/no color).
                input_weights = np.zeros(new_weights.shape)
                # Calculate the weight in each bin, in the input WESTPA data
                # file.  This is necessary because w_assign does not calculate
                # labeled populations.
                for i in xrange(input_weights.shape[0]):
                    input_weights[i] = np.sum(
                            seg_weights[np.where(input_assignments == i)] 
                                              )
                # Suppress division errors
                #with np.errstate(divide='ignore', invalid='ignore'):
                with np.errstate(all='ignore'):
                    scaling_coefficients = new_weights/input_weights
                    # Set nonsensical value to zero; is this really necessary?
                    #scaling_coefficients[~np.isfinite(scaling_coefficients)] = 0
                # Get the HDF5 group for this iteration. It was already created
                # while initializing the output file.
                output_iter_group = self.output['iterations/iter_{:08d}'
                                                .format(iiter)]

                input_seg_index = np.array(input_iter_group['seg_index']) 
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
