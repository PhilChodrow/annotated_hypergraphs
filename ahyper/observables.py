from itertools import groupby
from collections import Counter

from .utils import normalise_counters

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
    keys = set(chain.from_iterable(densities.values()))
    for item in densities.values():
        item.update({key:0 for key in keys if key not in item})

    if absolute_values:
        return densities
    
    else:
        normalise_counters(densities)
        return densities