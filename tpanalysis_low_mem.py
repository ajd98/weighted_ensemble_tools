#!/usr/bin/env python
from __future__ import print_function
import h5py
import numpy
import os
import sys

#class PathManager(object):
#    def __init__(self):
#        self.activepaths = {}
#
#    def update(self, events, niter, weights):
#        '''
#        events: a list of tuples (miter, segid) corresponding to successful 
#          events.
#        niter: The iteration for which to add successful events
#        weights: a length ``nsegs`` array of weights for segments in iteration
#          niter. 
#        '''
#        self.activepaths  

class TPAnalysis(object):
    def __init__(self, westh5, assignh5, succpath, last_iter=None, copy=False):
        '''
        westh5: a west.h5 data file (h5py file object)
        assignh5: output file from w_assign (h5py file object)
        succpath: path to text output from w_succ (string)
        '''
        self.westh5 = westh5
        self.assignh5 = assignh5
        self.succpath = succpath
        self.load_succ_list(self.succpath)
        self.copy = copy

        if last_iter is not None:
            self.last_iter=last_iter
        else:
            self.last_iter=self._get_last_iter()

        self._load_state_map()

        return

    def _get_last_iter(self):
        return self.westh5.attrs['west_current_iteration'] - 1

    def _init_output(self):
        if os.path.exists(self.outputpath):
            print("Specified output path {:s} already exists!"\
                  .format(self.outputpath))
            print("Exiting...")
            exit(0)
        self.outputwest = h5py.File(self.outputpath)
        return

    def load_succ_list(self, wsucc_txt_output):
        # Formatted as [[iteration, seg_id]
        #               [iteration, seg_id]
        #               ...                ] 
        fpath = os.path.expanduser(wsucc_txt_output)
        pairs = numpy.loadtxt(fpath, usecols=(0,1), skiprows=1, 
                              dtype=int)
        self.succ_list = [] 
        for pair in pairs:
            self.succ_list.append(tuple(pair))
        self.succ_list.sort()
        return

    def _get_seg_index(self, niter):
        '''Return the seg_index (h5py dataset) for niter''' 
        segindex = self.westh5['iterations/iter_{:08d}/seg_index'.format(niter)]
        return segindex

    def _load_state_map(self):
        self.state_map = numpy.array(self.assignh5['state_map'])
        return

    def _check_active(self, niter, segid):
        '''This method is only currently designed to work for steady state 
        simulations!'''
        states = self.state_map[self.assignments[segid]] 
        if numpy.any(states == self.istate):
            return False
        if segid < 0:
            return False
        return True

    def _make_new_seg_index(self, niter):
        original_seg_index = numpy.array(
                                 self.westh5['iterations/iter_{:08d}/seg_index'\
                                             .format(niter)] 
                                         )
        new_seg_index = [] 
        for segid in xrange(len(original_seg_index)):
            entry = (self.new_weights[niter, segid],
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
            try:
                if int(key1[5:]) > self.last_iter:
                    break
            except:
                continue
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
        self.istate = istate
        self.fstate = fstate
        self.outputpath = outputpath
        # Indexed by iteration, segid; will hold new weights based on tracing
        # successful trajectories. One-indexed!
        self.new_weights = numpy.zeros((self.last_iter+1,
                                        self.assignh5['assignments'].shape[1]))

        # list of lists.  [[segid, new_weight],
        #                  [segid, new_weight],
        #                  ...                ]
        self.active_paths = []

        # Start removing the highest iteration index successful segments first
        # This depends on that self.succ_lis is pre-sorted!
        curseg = self.succ_list.pop()

        # Iterate through the requested iterations and begin simultaneous
        # traces.  We want to minimze the number of accesses to the disk
        for iiter in xrange(self.last_iter, 0, -1):
            print('\r  {:04d}'.format(iiter), end='')
            # For each iteration, we need the history parent of each segment
            # For each iteration with successful events, we also need the
            # weights.
            segindex = self._get_seg_index(iiter)
            
            # Convert h5py datasets to numpy arrays (in memory) for quick 
            # access.
            weights = numpy.array(segindex['weight']) 
            self.parent_ids = numpy.array(segindex['parent_id'])
            self.assignments = numpy.array(self.assignh5['assignments'][iiter-1])
            
            # Update active paths.  At this point, segments in active_paths are
            # for the current iteration.
            while curseg[0] >= iiter:
                if curseg[0] == iiter:
                    weight =  weights[curseg[1]]
                    self.active_paths.append([curseg[1], weight])
                if len(self.succ_list) > 0:
                    curseg = self.succ_list.pop()
                else:
                    break

            # Copy weights form active paths to self.new_weights:
            for activepath in self.active_paths:
                segid = activepath[0] 
                weight = activepath[1]
                self.new_weights[iiter, segid] = weight

            # segments in actve paths are still for the current iteration. 
            # This loop makes their iteration index correspond to iiter-1
            predecessorpaths = [] 
            for activepath in self.active_paths:
                predecessorid = self.parent_ids[activepath[0]] 
                weight = activepath[1]
                if self._check_active(iiter, predecessorid):
                    duplicate = False
                    # Merge common parents of active paths
                    for ip, other_predecessor in enumerate(predecessorpaths):
                        if other_predecessor[0] == predecessorid:
                            predecessorpaths[ip][1] += weight 
                            duplicate = True
                    if not duplicate:
                        predecessorpaths.append([predecessorid, weight])
            self.active_paths = predecessorpaths
                    
        self._init_output()
        self._write_new_westh5()
                
