from itertools import groupby
from collections import Counter

import pandas as pd

def local_role_density(annotated_hypergraph, include_focus=False):
    """
    Calculates the density of each role within a 1-step neighbourhood
    of a node, for all nodes.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: A annotated hypergraph
        include_focus [Bool]: If True, includes the roles of the focal node
                              in th calculation
    Returns:
        role_densities []: An array of dimension (# nodes x # roles) 
                           describing the density of each role.
    """
    A = annotated_hypergraph

    def get_counts(group):
        return Counter([x[1] for x in group])

    by_edge = {eid:get_counts(v) for eid, v in groupby(sorted(A.IL, key=lambda x: x[2], reverse=True), lambda x: x[2])}

    include_focus = False

    densities = {}
    for node,role,edge,_ in A.IL:
        densities[node] = densities.get(node, Counter()) + by_edge[edge]
        
        if not include_focus:
            densities[node] = densities.get(node, Counter()) - Counter([role])
            
    densities = pd.DataFrame(densities).T.fillna(0)
    densities = densities.div(densities.sum(axis=1), axis=0)

    return densities