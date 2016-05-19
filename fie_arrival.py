#!/usr/bin/env python
import numpy
import h5py
import sys
# make sure wegraph.py is in the path
sys.path.append("/home/ajd98/development/weighted_ensemble_tools/")
import wegraph

def main():
    print("\nWorking on 71_60_N2NP")
    # Make a list of paths to the west.h5 data files you want to analyze
    s7160N2NPwesth5paths = ['../../71_60_analysis_200/west_rw_files/71_60_N2NP_1_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_2_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_3_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_4_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_5_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_6_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_7_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_8_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_9_westrw.h5',
                            '../../71_60_analysis_200/west_rw_files/71_60_N2NP_10_westrw.h5']

    # Iterate over each west.h5 and analyze it separately
    for i, westh5path in enumerate(s7160N2NPwesth5paths):
        westh5 = h5py.File(westh5path,'r+')

        # Initialize 
        weg = wegraph.WEGraph(westh5, last_iter=2000)
        # Build the graph. Takes on the order of 1 Gb/1.0e+6 segments
        print("building graph...")
        weg.build()
        print("Done!")

        # Load the text output from w_succ (run previously)
        weg.load_succ_list('~/projects/ss_switch/71_60_analysis_200/'
                           'succ/{:d}.out'.format(i+1))

        # Calculate the number of independent events. Outputs to STDOUT
        weg.prune_for_independence(35)

    
if __name__ == '__main__':
   main()
