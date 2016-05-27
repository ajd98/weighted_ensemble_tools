#!/usr/bin/env python
from __future__ import print_function
import h5py
import numpy
import os
import sys
sys.path.append("/home/ajd98/development/weighted_ensemble_tools/")
import wegraph

class GetWeight:
    '''
    Class for getting weights from a west.h5 data file. Stores weight arrays in
    memory as they are accessed; this speeds up retrieval and prevents HDF5
    errors.
    '''
    def __init__(self, westh5):
        self.westh5 = westh5
        self.weightdict = {}
        return

    def add_weight_iteration(self, niter):
        '''Store weight array in memory.'''
        self.weightdict[niter] = numpy.array(
            self.westh5['iterations/iter_{:08d}/seg_index'.format(niter)]\
                  ['weight']                 )
        # Delete older entries to conserve memory
        try:
            del self.weightdict[niter+1]
        except KeyError:
            pass
        return

    def get_weight(self, niter, nseg):
        '''
        Retrieve the weight corresponding to the nseg segment of iteration 
        niter. 
        '''
        try:
            w = self.weightdict[niter][nseg]
        except KeyError:
            self.add_weight_iteration(niter)
            w = self.weightdict[niter][nseg]
        return {'weight': w}

class GetState:
    '''
    Class for getting state assignments from a w_assign output file. Stores 
    state arrays in memory as they are accessed; this speeds up retrieval and 
    prevents HDF5 errors.
    '''
    def __init__(self, assignh5, track_states=[]):
        '''
        track_states: a list, tuple, or array of state indices for the initial
          and final states in a reaction.  If a segment enters a bin
          corresponding to any state in track_states during an iteration, then
          it takes on that state id for the entire iteration. 
        '''
        self.assignh5 = assignh5
        self.state_map = numpy.array(assignh5['state_map'])
        self.nstates = assignh5.attrs['nstates']
        self.track_states = track_states
        self.statedict = {}
        return

    def add_state_iteration(self, niter):
        '''Store state array in memory'''
        assignments = numpy.array(self.assignh5['assignments'][niter-1]) 
        # Map bin assignments into state assignments
        states = self.state_map[assignments]  

        # By default, use the state assignment for the last timepoint
        condensed_states = states[:,-1] 

        # If the segment was in one of the track_states during the iteration,
        # set it's state id to that state. Note this operation can possibly 
        # depend on the order of track_states, but this should rarely matter.
        for stateid in self.track_states:
            w = numpy.any(states==stateid, axis=1)
            condensed_states[w] = stateid
        self.statedict[niter] = condensed_states 

        # Delete older entries to conserve memory
        try:
            del self.statedict[niter+1]
        except KeyError:
            pass

        return

    def get_state(self, niter, nseg):
        '''
        Retrieve the state corresponding to the nseg segment of iteration 
        niter. 
        '''
        try:
            i = self.statedict[niter][nseg]
        except KeyError:
            self.add_state_iteration(niter)
            i = self.statedict[niter][nseg]
        return {'state': i}

class GetStateWeight(GetWeight, GetState):
    def __init__(self, westh5, assignh5, succlist, track_states=[]):
        '''
        See GetWeight and GetState class for explanation of arguments.
        '''
        GetWeight.__init__(self, westh5)
        GetState.__init__(self, assignh5, track_states=track_states)
        #self.succlist = set(succlist)
        return

    def get(self, niter, nseg):
        print('\r  {:04d}'.format(niter), end='')
        d = self.get_state(niter, nseg)
        #if (niter, nseg) in self.succlist:
        d.update(self.get_weight(niter, nseg))
        return d 
        


class TPAnalysis(wegraph.WEGraph):
    def __init__(self, westh5, assignh5, succpath, last_iter=None, copy=False):
        '''
        westh5: a west.h5 data file (h5py file object)
        assignh5: output file from w_assign (h5py file object)
        succpath: path to text output from w_succ (string)
        '''
        super(TPAnalysis, self).__init__(westh5, last_iter) 
        self.assignh5 = assignh5
        self.succpath = succpath
        self.load_succ_list(self.succpath)
        self.copy = copy
        return

    def _init_output(self):
        if os.path.exists(self.outputpath):
            print("Specified output path {:s} already exists!"\
                  .format(self.outputpath))
            print("Exiting...")
            exit(0)
        self.outputwest = h5py.File(self.outputpath)
        return

    def _trace_and_add(self, child, istate):
        child_weight = self.graph.node[child]['weight']
        node = child
        check = True
        while check:
            self.graph.node[node]['new_weight'] += child_weight 
            predecessors = self.graph.predecessors(node) 
            if len(predecessors) > 1:
                raise ValueError("Segment {:d} of iteration {:d} has more than "
                                 "one parent! Exiting..."\
                                 .format(node[1], node[0]))
            if len(predecessors) == 0:
                break
            # predecessors should be a list with one element if it gets this 
            # far.  Get that element.
            node = predecessors[0]

            # Make sure the segment index is nonnegative.  Segment indices are 
            # negative for new trajectories. Also, make sure the segment is 
            # still in the transition path region.
            check = (node[1] > 0) and self.graph.node[node]['state'] != istate
        return

    def _make_new_seg_index(self, niter):
        original_seg_index = numpy.array(
                                 self.westh5['iterations/iter_{:08d}/seg_index'\
                                             .format(niter)] 
                                         )
        new_seg_index = [] 
        for segid in xrange(len(original_seg_index)):
            entry = (self.graph.node[(niter, segid)]['new_weight'],
                     original_seg_index[segid][1],
                     original_seg_index[segid][2],
                     original_seg_index[segid][3],
                     original_seg_index[segid][4],
                     original_seg_index[segid][5],
                     original_seg_index[segid][6],
                     original_seg_index[segid][7])
            new_seg_index.append(entry)
        new_seg_index = numpy.array(new_seg_index, 
                                    dtype=original_seg_index.dtype)
        return new_seg_index
            
            

    def _write_new_westh5(self):
        '''Copy or link datasets besides the seg_index datasets from the input 
        WESTPA data file to the output (reweighted) data file. '''
        inputwestpath = self.westh5.filename
        for key in self.westh5.keys():
             if key != 'iterations':
                 if self.copy:
                     self.westh5.copy(key, self.outputwest)
                 else:
                     self.outputwest[key] = h5py.ExternalLink(inputwestpath,
                                                              key)
        for name, val in self.westh5.attrs.items():
            self.outputwest.attrs.create(name, val)

        self.outputwest.create_group('iterations')
        for key1 in sorted(self.westh5['iterations'].keys()):
            if int(key1[5:]) > self.last_iter: 
                break
            self.outputwest.create_group('iterations/{:s}'.format(key1))
            for key2 in self.westh5['iterations/{:s}'.format(key1)]:
                key = 'iterations/'+key1+'/'+key2
                if key2 == 'seg_index':
                    niter = int(key1[5:])
                    new_seg_index = self._make_new_seg_index(niter)
                    self.outputwest.create_dataset(key, data=new_seg_index,
                                                   dtype=new_seg_index.dtype)
                #if key2 != 'seg_index':
                else:
                    if self.copy:
                        self.westh5.copy(key, 
                                         self.outputwest['iterations/'+key1])
                    else:
                        self.outputwest[key] = h5py.ExternalLink(inputwestpath,
                                                        key)
            for name, val in self.westh5['iterations/{:s}'.format(key1)]\
                                        .attrs.items():
                self.outputwest['iterations/{:s}'.format(key1)]\
                               .attrs.create(name, val)
        self.outputwest.close()
        return

    def run(self, outputpath, istate=0, fstate=1):
        '''
        Generate a new west.h5 file at outputpath, with weights of
        nonsuccessful pathways zero, and weights of successful pathways
        determined by the weights of their children.

        istate, fstate: indices of initial and final states.
        '''
        # Initialize output west.h5 file
        self.outputpath = outputpath 
        self._init_output()

        # Build the history graph, including weights as attributes
        getstateandweight = GetStateWeight(self.westh5, self.assignh5, 
                                           self.succ_list,
                                           track_states=(istate, fstate))
        print("Building graph...")
        self.build(get_props=getstateandweight.get)

        # Set new weights all to zero 
        print("Initializing new weights")
        for node in self.graph.nodes():
            self.graph.node[node]['new_weight'] = 0

        # Update new weights according to sum of weights of successful children
        print("Calculating new weights")
        for succ_child in self.succ_list: 
            if succ_child[0] <= self.last_iter: 
                self._trace_and_add(succ_child, istate)
 
        # Write the new west.h5 file with updated weights.
        print("Writing output file")
        self._write_new_westh5()
        return
