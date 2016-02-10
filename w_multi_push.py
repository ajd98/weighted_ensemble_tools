#!/usr/bin/env python
from __future__ import print_function, division; __metaclass__ = type
import h5py
import logging
import numpy as np
import westpa
import westtools
from westtools import (WESTMultiTool, WESTDataReader, ProgressIndicatorComponent)

log = logging.getLogger('westtools.w_multi_push')

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

class WPostanalysisPush(WESTMultiTool):
    prog ='w_multi_push'
    description = '''\
Apply weights calculated using the postanalysis reweighting scheme (see 
"w_postanalysis_reweight --help" or "w_multi_reweight --help") to a set of 
WESTPA data files (usually named similarly to "west.h5") by scaling the 
probability in each bin. Bin assignments (one file per WESTPA data file, 
usually named "assign.h5" or similar) and a corresponding postanalysis 
reweighting output file (usually "kinrw.h5" or "multi_rw.h5", only one file) 
must be supplied, in addition to the iteration from which to pull weights.

WARNING: Output from this script may not be compatible with some tools. Due to
the nature of the postanalysis reweighting process, some walkers may be assigned
zero weight. Additionally, total weight may not sum to one for some iterations, 
as a bin predicted to have nonzero weight by the postanalysis reweighting scheme
may depopulate during the course a simulation. 

--------------------------------------------------------------------------------
Output format
--------------------------------------------------------------------------------

The output files will be of the same format as
the original WESTPA data files. New weights are found in place of the original 
weights, located in:

/iterations/iter_{N_ITER:08d}/seg_index/ 

for each output file.

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
        self.rw_file = None

        self.weights_attributes_initialized = False
        self.weights_already_calculated = False
        self.time_average_scaling_vector_calculated = False

        
    def add_args(self, parser):
        self.progress.add_args(parser)
        self.data_reader.add_args(parser)

        iogroup = parser.add_argument_group('input/output options')

        iogroup.add_argument('-rw', dest='rw_H5_path', metavar='RW_FILE',
                             default=None,
                             help='''Pull weights from RW_FILE. This should be
                             the output file from either w_postanalysis_reweight 
                             or w_multi_reweight. Alternatively, this file may 
                             be specified in the YAML file''')

        iogroup.add_argument('-y', '--yaml', dest='yamlpath', 
                             metavar='YAMLFILE', 
                             default='multi_reweight.yaml',
                             required=True,
                             help='''Load options from YAMLFILE. For each 
                             simulation, specify an assignments file, input 
                             WESTPA data file (e.g., 'west.h5'), and output 
                             WESTPA data file.  Search for files in 
                             ['simulations'][SIMNAME]['assignments'], 
                             ['simulations'][SIMNAME]['input_west']
                             ['simulations'][SIMNAME]['output_west'].  
                             Additionally, a reweighting file may be specified
                             under the key ['reweighitng_data'] at the top 
                             level of the YAML file. 
                             ''')

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

        cogroup.add_argument('-nc', '--no-color', action='store_true', 
                             dest='no_color',
                             help='''If specified, do not use colored bins for
                             rescaling weights. By default, use colored bins.
                             ''')

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
        self.rwH5_path = args.rw_H5_path
        # Load the yaml input file; Make self.yamlargdict available
        self.yamlpath = args.yamlpath
        self.parse_from_yaml(args.yamlpath)

        # Calculation arguments
        self.copy = args.copy
        self.n_iter = args.n_iter

        if args.no_color:
            self.i_use_color = False
        else:
            self.i_use_color = True

        self.first_iter = None
        self.last_iter = None
        if args.iter_range is not None:
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

    def process_yaml(self):
        '''Process information specified in the yaml file.'''
        class YAMLArgumentError(ArgumentError):
            pass

        if not 'simulations' in self.yamlargdict.keys():
            raise YAMLArgumentError("Key `simulations` not found at uppermost "
                                    "level of yaml file {:s}.  See -h/--help "
                                    "for more information."
                                    .format(self.yamlpath)
                                    )
        self.simnames = []
        self.westH5_paths = []
        self.assignments_paths = []
        self.output_paths = []
        self.rwH5_path = None
        for simname in self.yamlargdict['simulations'].keys():
            self.simnames.append(simname)
            self.westH5_paths.append(self.yamlargdict['simulations']\
                                                     [simname]['input_west'])
            self.assignments_paths.append(
                    self.yamlargdict['simulations'][simname]['assignments']
                                          )
            self.output_paths.append(self.yamlargdict['simulations']\
                                                     [simname]['output_west'])
        if 'reweighting_data' in self.yamlargdict.keys(): 
            if self.rwH5_path is not None:
                raise ArgumentError("Reweighting data files were specified both"
                                    " via command line (see -rw, path specified"
                                    ": {:s}) and via YAML input (see -y/--yaml,"
                                    " path specified: {:s}). Please only "
                                    "specify using one option."
                                    )
            self.rwH5_path = self.yamlargdict['reweighting_data'] 


    def open_files(self):
        '''Open the WESTPA data files, the reweighting output file, the
        assignments files, and the output files.''' 
        
        self.westH5_files = []
        self.assignments_files = []
        self.output_files = []
        for isim in xrange(len(self.simnames)):
            self.westH5_files.append(h5py.File(self.westH5_paths[isim], 'r'))
            self.assignments_files.append(
                    h5py.File(self.assignments_paths[isim], 'r')
                                          )
            self.output_files.append(h5py.File(self.output_paths[isim], 'w'))
        self.rwH5 = h5py.File(self.rwH5_path, 'r')
 

    def check_consistency_of_input_files(self):
        '''Check that the assignment file and west.h5 file have the same number
        of walkers for each iteration, and check that the reweighting output 
        file and the assignment file use the same number of bins.'''
        self.iter_sum = 0
        last_iter_list = []
        # First iterate over the simulations and collect a few statistics to 
        # be used for the progress bar.
        for isim in xrange(len(self.simnames)):
            assignments = self.assignments_files[isim]
            westH5 = self.westH5_files[isim]
            # Assume that the assignments file includes data for ALL iterations.
            # At the time of this code's writing, it must, but this may change
            # later.
            last_iter = assignments['assignments'].shape[0] # Zero-indexed 
            last_iter_list.append(last_iter)
            self.iter_sum += last_iter 
        self.pi.new_operation('Checking input files for consistency',
                              self.iter_sum)
        # This will be used later to make sure the number of bins in each 
        # assignments file matches that in the reweighting output file
        rw_n_bins = self.rwH5['bin_prob_evolution'].shape[1]
        rw_nstates = self.rwH5['state_labels'].shape[0]
        for isim in xrange(len(self.simnames)): 
            assignments = self.assignments_files[isim]
            westH5 = self.westH5_files[isim]
            ## First check that the assignment and west.h5 file have the same ##
            ## number of walkers.                                             ##
            # Get the total number of bins.  Indices corresponding to walkers  
            # not present in a given iteration will be assigned this integer.
            n_bins = assignments['bin_labels'].shape[0] 

            for iiter in xrange(1, last_iter_list[isim]+1):#iiter is one-indexed
                try:
                    iter_group = westH5['iterations/iter_{:08d}'.format(iiter)]
                except KeyError:
                    raise ConsistencyError("Error. Iteration {:d} exists in the"
                                           " assignments file {:s} but not in "
                                           "the associated WESTPA data file "
                                           "{:s}!".format(iiter,
                                                    self.assignments_paths[isim], 
                                                    self.westH5_paths[isim])
                                           )
                # Number of walkers in this iteration, for the westh5 file.
                westh5_n_walkers = iter_group['seg_index'].shape[0]
                # Number of walkers in this iteration, for the assignh5 file. 
                # The size of the assignments array is the same for all 
                # iterations, and equals the maximum observed number of walkers.
                # For iterations with fewer than the maximum number of walkers, 
                # The rest of the indices are filled with the integer `n_bins`. 
                # Switch to zero-indexing, and only look at the first time point
                assign_n_walkers = np.count_nonzero(
                  np.array(assignments['assignments'][iiter-1][:,0]) != n_bins 
                                                    )
                if not westh5_n_walkers == assign_n_walkers:
                    raise ConsistencyError("The number of walkers in the WESTPA"
                                           " data file ({:s}, {:d}) and the " 
                                           "number of walkers in the associated"
                                           " assignments file ({:s}, {:d}) for " 
                                           "iteration {:d} do not match!"
                                           .format(self.westH5_paths[isim],
                                                   westh5_n_walkers,
                                                   self.assignments_paths[isim],
                                                   assign_n_walkers,
                                                   iiter)                   
                                           )
                self.pi.progress += 1
            ## Now check that the reweighting output file and the assignments ## 
            ## file use the same number of bins.                              ##
            # rw_n_bins is colored, but n_bins (from the assignments file) is not.
            assign_nstates = assignments['state_labels'].shape[0]
            if not assign_nstates == rw_nstates:
                raise ConsistencyError("The number of states used in the "
                                       "assignments file ({:s}, {:d} states) "
                                       "does not match the number of states "
                                       "used in the reweighting output file "
                                       "({:d})!".format(
                                                 self.assignments_paths[isim],
                                                 assign_nstates, 
                                                 rw_nstates)           )
            if not assign_nstates*n_bins == rw_n_bins:
                raise ConsistencyError("The number of bins used in the " 
                                       "assignments file ({:s}, {:d} bins) does" 
                                       " not match the number of bins used in " 
                                       "the reweighting output file ({:d}, "
                                       "converted to without color)."
                                       .format(nbins, rw_n_bins/rw_nstates)    )   
        self.pi.clear()

    def initialize_output(self):
        '''Copy or link datasets besides the seg_index datasets from the input 
        WESTPA data file to the output (reweighted) data file. '''
        self.pi.new_operation('Initializing output file',
                              self.iter_sum)
        for isim in xrange(len(self.simnames)):
            westH5 = self.westH5_files[isim]
            westH5_path = self.westH5_paths[isim]
            output = self.output_files[isim]
            for key in westH5.keys():
                 if key != 'iterations':
                     if self.copy:
                         westH5.copy(key, output)
                     else:
                         output[key] = h5py.ExternalLink(westH5_path,
                                                         key)
            for name, val in westH5.attrs.items():
                output.attrs.create(name, val)
             
            output.create_group('iterations')
            for key1 in westH5['iterations']:
                output.create_group('iterations/{:s}'.format(key1))
                for key2 in westH5['iterations/{:s}'.format(key1)]:
                    if key2 != 'seg_index':
                        key = 'iterations/'+key1+'/'+key2 
                        if self.copy:
                            westH5.copy(key, output['iterations/'+key1])  
                        else:
                            output[key] = h5py.ExternalLink(westH5_path,
                                                            key)
                for name, val in westH5['iterations/{:s}'.format(key1)].attrs.items():
                    output['iterations/{:s}'.format(key1)].attrs.create(name, val) 
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
                   dtype=self.assignments_files[0]['assignments'].dtype
                                    )
            for i in xrange(cfe.shape[0]):
                # Axes are (timepoint index, beginning state, ending state)
                # and final index gets the "iter_stop" data
                # stop_iter is exclusive, so subtract 1
                self.idx_map[i] = cfe[i,0,0][1] -1 # Last iteration included in this
                                                   # averaging window
            self.weights_attributes_initialized = True

        if not self.weights_already_calculated:
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

        else:
            return self.new_weights

    def get_input_assignments(self, isim, iiter):
        '''Get the bin assignments for the first timepoint of iteration
        ``iiter``, for all walkers in the input assignments file with index 
        `isim`. Return a 1-dimension numpy array ``assignments``, where 
        ``assignments[i]`` gives the bin index assigned to walker i.  Bin 
        indices take into account whether or not the colored scheme is used.''' 
        # Get the total weight in each bin for the input assignments file
        # Look only at the first time point! --------------------------V
        assignments = np.array(
                self.assignments_files[isim]['assignments'][iiter-1][:,0]
                               )
        if self.i_use_color:
            traj_labels = np.array(
                    self.assignments_files[isim]['trajlabels'][iiter-1][:,0]
                                   )
            assignments = assignments*nstates+traj_labels
        return assignments
               

    def calculate_scaling_coefficients(self):
        '''Calculate and return a vector of scaling coefficients, which should 
        be applied equally across all simulations and all iterations.  
        scaling_coefficients[i] gives the value by which to scale (multiply) the 
        weight of any walker in bin i.  

        This method first calls self.get_new_weights(self.n_iter) to find what 
        weights output by the postanalysis reweighting scheme are.  Next, it 
        considers whether or not the colored scheme is used.  Finally, it 
        calculates the input assignments if necessary.'''
        if self.time_average_scaling_vector_calculated:
            return self.time_average_scaling_vector
        else: # if not self.time_average_scaling_vector_calculated:
            self.pi.new_operation("Calculating scaling vector.",
                                  self.iter_sum)
            # Calculate the scaling vector  
            # old_weights will contain the total weight observed in each 
            # bin during the FIRST timepoint of all iterations; this must
            # be calculated here rather than using labeled_populations from
            # the assignments file, as labeled_populations includes all
            # timepoints
            new_weights = self.get_new_weights(self.n_iter)
            old_weights = np.zeros(new_weights.shape)
            # Add up weights in each bin from all simulations
            for isim in xrange(len(self.simnames)):
                # Calcuate the iteration range to use if not specified.
                if self.first_iter is None:
                    iter_strs = self.westH5_files[isim]['iterations/'].keys().sort()
                    first_iter = int(iter_strs[0][5:])
                    last_iter = int(iter_strs[-1][5:])
                else:
                    first_iter = self.first_iter
                    last_iter = self.last_iter
                # Iterate over all WE iterations and sum up the weight in each
                # bin
                for iiter in xrange(first_iter, last_iter+1):
                    weights = np.array(
                            self.westH5_files[isim]\
                                             ['iterations/iter_{:08d}/seg_index'
                                              .format(iiter)]['weight']
                                       )
                    assignments = self.get_input_assignments(isim, iiter) 
                    for bin_idx in xrange(old_weights.shape[0]):
                        where = np.where(assignments == bin_idx)
                        old_weights[bin_idx] += np.sum(weights[where])
                    self.pi.progress += 1
            with np.errstate(all='ignore'):
                self.time_average_scaling_vector = new_weights/old_weights
                self.time_average_scaling_vector[
                        ~np.isfinite(self.time_average_scaling_vector)
                                                 ] = 0.0
            self.time_average_scaling_vector_calculated = True
            self.pi.clear()
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
            
            self.process_yaml()
            self.open_files()
            self.check_consistency_of_input_files() 
            self.initialize_output()

            pi.new_operation('Creating new WESTPA data file with scaled '
                             'weights.', self.iter_sum)

            # This is guaranteed to be the same for all simulations, due to 
            # self.check_consistency_of_input_files()
            self.nstates = len(self.assignments_files[0]['state_labels'])

            for isim in xrange(len(self.simnames)):
                last_iter = self.assignments_files[isim]['assignments'].shape[0] 

                for iiter in xrange(1, last_iter+1): # iiter is one-indexed 
                    scaling_coefficients = self.calculate_scaling_coefficients()
                    input_iter_group = self.westH5_files[isim]\
                                                        ['iterations/iter_{:08d}'
                                                         .format(iiter)]
                    input_assignments = self.get_input_assignments(isim, iiter) 
                    # Get the HDF5 group for this iteration. It was already 
                    # created while initializing the output file.
                    output_iter_group = self.output_files[isim]\
                                                       ['iterations/iter_{:08d}'
                                                        .format(iiter)]

                    input_seg_index = np.array(input_iter_group['seg_index']) 
                    seg_weights = input_seg_index['weight']
                    # Build the new seg_index piece by piece.  Start with an 
                    # empty list and add data for one segment at a time. Then  
                    # convert to a numpy array.
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
