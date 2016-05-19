#!/usr/bin/env python
from __future__ import print_function
import h5py
import networkx
import numpy
import os
import sys
import matplotlib.pyplot as pyplot

# WEGraph.remove_branch works recursively; when removing large branches it hits
# Python's internal recursion limit.
sys.setrecursionlimit(1000000)

class WEGraph:
    def __init__(self, westh5, last_iter=None):
        self.westh5 = westh5 
        if last_iter is not None:
            self.last_iter=last_iter
        else:
            self.last_iter=self._get_last_iter()

    def _get_last_iter(self):
        return self.westh5.attrs['west_current_iteration'] - 1

    def _format_iterstr(self, niter):
        return "iter_{:08d}".format(niter)

    def _return_parent_map(self, niter):
        return self.westh5['iterations'][self._format_iterstr(niter)]\
                          ['seg_index']['parent_id']

    def _get_nsegs(self, niter):
        return self.westh5['iterations'][self._format_iterstr(niter)]\
                          ['seg_index'].shape[0]

    def build(self):
        '''
        Build a NetworkX graph from a WESTPA data file.
        '''
        self.graph = networkx.DiGraph()

        # Iterate over the WE iterations, starting with the last one first.
        # The WE iteration with highest index is a special case.  
        niter = self.last_iter 
        # the parent index of child i is parentmap[i]  
        nsegs = self._get_nsegs(niter) 
        for iseg in xrange(nsegs):
            self.graph.add_node((niter, iseg))
        previous_nsegs = nsegs
        parentmap = self._return_parent_map(niter)  

        for niter in xrange(self.last_iter-1, 0, -1):
            nsegs = self._get_nsegs(niter) 
            for iseg in xrange(nsegs):
                self.graph.add_node((niter, iseg))
            
            for iseg in xrange(previous_nsegs):
                self.graph.add_edge((niter, parentmap[iseg]), (niter+1, iseg))
            previous_nsegs = nsegs
            parentmap = self._return_parent_map(niter)  
        return

    def build_ancestry_tree(self, childlist):
        '''
        Build a NetworkX graph from a WESTPA data file, including only 
        ancestors of segments in ``childlist``.
        '''
        self.ancestry_tree = networkx.DiGraph()

        # Find the highest index iteration to include        
        childlist.sort(reverse=True)
        last_iter = childlist[0][0]

        # Iterate over the WE iterations, starting with the last one first.
        niter = last_iter 
        rootlist = []
        for leaf in childlist:
            child = leaf
            while True:
                parentmap = self._return_parent_map(child[0])
                parentid = parentmap[child[1]] 
                if parentid < 0: 
                    rootlist.append(child)
                    break
                parent = (child[0]-1, parentid)
                self.ancestry_tree.add_edge(parent,
                                            child)
                child = parent

        #for curriter in xrange(last_iter, 1, -1):
        #    # the parent index of child (niter, i) is parentmap[i]  
        #    parentmap = self._return_parent_map(curriter)  
        #    for (segiter, segid) in childlist:
        #        if segiter == curriter:
        #            parent = (curriter-1, parentmap[segid])
        #            self.ancestry_tree.add_edge(parent,
        #                                        (segiter, segid)
        #                                        ) 
        #            childlist.remove((segiter, segid))
        #            childlist.append(parent)
        return rootlist

    def display_ancestry_tree(self, roots, leaves, figname='ancestrytree.pdf'):
        nodelist = self.ancestry_tree.nodes()
        nodelist.sort()
        print('calculating tree widths')
        # Calculate widths of disjoint trees
        treewidths = []
        leafcount = 0
        for root in roots:
            maxwidth=1
            parentlist = [root]
            while parentlist: 
                childlist = []
                for parent in parentlist:
                    successors = self.ancestry_tree.successors(parent)
                    childlist += successors
                    if not successors: #If this parent is a leaf
                        leafcount += 1
                if len(childlist) + leafcount > maxwidth:
                    maxwidth = len(childlist)+leafcount
                parentlist = childlist
            treewidths.append(maxwidth)
        treebounds = [0]
        for treewidth in treewidths:
            treebounds.append(treewidth+treebounds[-1]) 
        # set the positions of root nodes; each position is a tuple (x,y), with
        # the iteration as y
        positions = {} 
        for iroot, root in enumerate(roots):
            positions[root] = ((treebounds[iroot+1]+treebounds[iroot])/2, 
                               -1*root[0]) 
        print('calculating positions')
        # set the positions of all the child nodes
        for iroot, root in enumerate(roots):
            print("\r {:02d}%".format(iroot*100/len(roots)), end='')
            parentlist = [root]
            while len(parentlist) > 0:
                childlist = []
                for parent in parentlist:
                    if parent == 'ghost':
                        childlist.append(parent)
                    else:
                        successors = self.ancestry_tree.successors(parent)
                        if successors:
                            childlist += self.ancestry_tree.successors(parent)
                        else: # this is a leaf; add ghost child
                            childlist.append('ghost')
                # Check if we are done
                living_check = False 
                for child in childlist:
                    if child != 'ghost': living_check=True
                if not living_check: 
                    break 
                # Spacing between children
                delta = float((treebounds[iroot+1]-treebounds[iroot]))\
                        /(len(childlist)+1) 
                for ichild, child in enumerate(childlist):
                    if child != 'ghost':
                        positions[child] = ((ichild+1)*delta+treebounds[iroot], 
                                            -1*child[0]) 
                parentlist = childlist
        print('generating figure')
        colors=[]
        for node in self.ancestry_tree.nodes():
            if node in leaves:
                colors.append((1,0,0,1))
            elif node in roots:
                colors.append((0,1,0,1))
            else:
                colors.append((0,0,0,1))
        networkx.draw_networkx(self.ancestry_tree, pos=positions, arrows=False,
                               node_size=5, font_size=4, with_labels=False,
                               node_color=colors)


    def remove_branch(self, node):
        '''
        Recursively prune all child nodes of a given node
        ''' 
        for child in self.graph.successors(node):
            self.remove_branch(child)
        self.graph.remove_node(node)
        return
        
    
    def prune_from_ancestor(self, root, radius):
        '''
        Prune nodes within radius of root, as well as any children of pruned 
        nodes.  Each vertex counts as "1" in calculation of the radius.
        ''' 
        openlist = [root]
        for i in xrange(radius+1):
            parentlist = []
            for opennode in openlist:
                parentlist += self.graph.predecessors(opennode) 
            for opennode in openlist:
                self.remove_branch(opennode)
            openlist = parentlist 

    def prune_for_independence(self, radius):
        '''
        Find the number of independent events (for a history graph only!).
        This method is destructive, altering self.graph 
        Before calling this method, call self.build and self.load_succ_list. 
        '''
        # Check all the successful events are there to begin with!
        for node in self.succ_list:
            if node[0] > self.last_iter:
                self.succ_list.remove(node)
                continue
            if not self.graph.has_node(node):
                raise KeyError("Node {:s} found in list of successful events "
                               "but is not part of the graph! Check to make "
                               "sure the list of successful events was "
                               "generated from the same WESTPA data file "
                               "supplied to this script.".format(repr(node))) 
        # Now iteratively prune the tree, keeping count of independent segments
        count = 0
        indevntlist = []
        self.succ_list.sort(reverse=True)
        for node in self.succ_list:
            if self.graph.has_node(node):
                print("\rCurrent iteration: {:s}; "
                      "found {:s} independent events so far."\
                      .format(str(node[0]).rjust(6), str(count).rjust(6)), 
                      end='')
                count += 1
                indevntlist.append(node)
                self.prune_from_ancestor(node, radius)
            else:
                continue 
        print("\nTotal: Found {:d} independent events.".format(count))
        return count, indevntlist
                                
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
        return
