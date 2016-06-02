#!/usr/bin/env python
import networkx
import os
import sys
import wegraph

def return_test_wegraph():
    r'''
           ___(1,0)___
          /           \
       (2,0)         (2,1)  
       /   \         /   \
     3,0  3,1       3,2  3,3
     /    / |       | \    
   4,0  4,1 4,2   4,3 4,4 

    '''
    # Create the graph manually rather than loading WESTPA data
    weg = wegraph.WEGraph('junk_string', last_iter=1000)

    # Build the graph by by hand
    weg.graph = networkx.DiGraph()
    weg.graph.add_edges_from([[(1,0),(2,0)],
                              [(1,0),(2,1)],
                              [(2,0),(3,0)],
                              [(2,0),(3,1)],
                              [(2,1),(3,2)],
                              [(2,1),(3,3)],
                              [(3,0),(4,0)],
                              [(3,1),(4,1)],
                              [(3,1),(4,2)],
                              [(3,2),(4,3)],
                              [(3,2),(4,4)]])
    return weg

def test_remove_branch1():
    '''Test the remove_branch method.'''
    weg = return_test_wegraph()
    # Test the remove_branch method
    weg.remove_branch((2,0))
    # Build the correct answer for comparison:
    comp_graph = networkx.DiGraph()
    comp_graph.add_edges_from([[(1,0),(2,1)],
                               [(2,1),(3,2)],
                               [(2,1),(3,3)],
                               [(3,2),(4,3)],
                               [(3,2),(4,4)]])
    assert networkx.is_isomorphic(weg.graph, comp_graph)

def test_remove_branch2():
    '''Test the remove_branch method.'''
    weg = return_test_wegraph()
    # Test the remove_branch method
    weg.remove_branch((3,2))
    # Build the correct answer for comparison:
    comp_graph = networkx.DiGraph()
    comp_graph.add_edges_from([[(1,0),(2,0)],
                               [(1,0),(2,1)],
                               [(2,0),(3,0)],
                               [(2,0),(3,1)],
                               [(2,1),(3,3)],
                               [(3,0),(4,0)],
                               [(3,1),(4,1)],
                               [(3,1),(4,2)]])
    assert networkx.is_isomorphic(weg.graph, comp_graph)

def test_remove_branch3():
    '''Test the remove_branch method; prune entire graph'''
    weg = return_test_wegraph()
    # Test the remove_branch method
    weg.remove_branch((1,0))
    # Build the correct answer for comparison:
    comp_graph = networkx.DiGraph()
    assert networkx.is_isomorphic(weg.graph, comp_graph)

def test_prune_from_ancestor1():
    '''Test the prune_from_ancestor method.'''
    weg = return_test_wegraph()
    # Test the prune_from_ancestor method
    weg.prune_from_ancestor((4,2), 1)

    # Build the correct answer for comparison:
    comp_graph = networkx.DiGraph()
    comp_graph.add_edges_from([[(1,0),(2,0)],
                               [(1,0),(2,1)],
                               [(2,0),(3,0)],
                               [(2,1),(3,2)],
                               [(2,1),(3,3)],
                               [(3,0),(4,0)],
                               [(3,2),(4,3)],
                               [(3,2),(4,4)]])
    assert networkx.is_isomorphic(weg.graph, comp_graph)

def test_prune_from_ancestor2():
    '''Test the prune_from_ancestor method.'''
    weg = return_test_wegraph()
    # Test the prune_from_ancestor method
    weg.prune_from_ancestor((4,2), 2)

    # Build the correct answer for comparison:
    comp_graph = networkx.DiGraph()
    comp_graph.add_edges_from([[(1,0),(2,1)],
                               [(2,1),(3,2)],
                               [(2,1),(3,3)],
                               [(3,2),(4,3)],
                               [(3,2),(4,4)]])
    assert networkx.is_isomorphic(weg.graph, comp_graph)

def test_prune_for_independence1():
    '''Test the prune_for_independence method.'''
    weg = return_test_wegraph()
    weg.succ_list= [(4,0),
                    (3,1),
                    (4,4)]
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try: count = weg.prune_for_independence(1)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    assert count[0] == 3

def test_prune_for_independence2():
    '''Test the prune_for_independence method.'''
    weg = return_test_wegraph()
    weg.succ_list= [(4,0),
                    (3,1),
                    (4,4)]
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try: count = weg.prune_for_independence(2)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    assert count[0] == 2

def test_prune_for_independence3():
    '''Test the prune_for_independence method.'''
    weg = return_test_wegraph()
    weg.succ_list= [(4,0),
                    (3,1),
                    (4,4)]
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try: count = weg.prune_for_independence(3)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    assert count[0] == 1

def test_prune_for_independence4():
    '''Test the prune_for_independence method
    Successful trajectories do not die in this example,
    but also should not count as independent because they
    are too near other events'''
    weg = return_test_wegraph()
    weg.succ_list= [(4,0),
                    (3,1),
                    (4,4),
                    (2,0),
                    (2,1)]
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try: count = weg.prune_for_independence(2)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    assert count[0] == 2

def test_prune_for_independence5():
    '''Test the prune_for_independence method
    Successful trajectories do not die in this example
    and should count as independent.'''
    weg = return_test_wegraph()
    weg.succ_list= [(4,0),
                    (3,1),
                    (4,4),
                    (1,0),
                    (2,1)]
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try: count = weg.prune_for_independence(2)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    assert count[0] == 3

if __name__ == '__main__':
    test_remove_branch1()
    test_remove_branch2()
    test_remove_branch3()
    test_prune_from_ancestor1()
    test_prune_from_ancestor2()
    test_prune_for_independence1()
    test_prune_for_independence2()
    test_prune_for_independence3()
    test_prune_for_independence4()
    test_prune_for_independence5()

    #print("weg.graph:")
    #for node in sorted(weg.graph.nodes()):
    #    print(node)
    #print("comp_graph")
    #for node in sorted(comp_graph.nodes()):
    #    print(node)
