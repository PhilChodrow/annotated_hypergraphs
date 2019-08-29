from itertools import groupby, chain
from collections import Counter

from .utils import normalise_counters

def local_role_density(annotated_hypergraph, include_focus=False, absolute_values=False):
    """
    Calculates the density of each role within a 1-step neighbourhood
    of a node, for all nodes.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
        include_focus [Bool]: If True, includes the roles of the focal node
                              in th calculation.
        absolute_values [Bool]: If True, returns role counts rather than densities.
    Returns:
        role_densities []: An array of dimension (# nodes x # roles) 
                           describing the density of each role.
    """
    A = annotated_hypergraph

    def get_counts(group):
        return Counter([x.role for x in group])

    by_edge = {eid:get_counts(v) for eid, v in groupby(sorted(A.IL, key=lambda x: x.eid, reverse=True), lambda x: x.eid)}

    densities = {}
    for incidence in A.IL:
        densities[incidence.nid] = densities.get(incidence.nid, Counter()) + by_edge[incidence.eid]
        
        if not include_focus:
            densities[incidence.nid] = densities.get(incidence.nid, Counter()) - Counter([incidence.role])

    keys = set(chain.from_iterable(densities.values()))
    for item in densities.values():
        item.update({key:0 for key in keys if key not in item})

    if absolute_values:
        return densities
    
    else:
        normalise_counters(densities)
        return densities

def degree_centrality(annotated_hypergraph):
    """
    Returns the weighted degree centrality for each node in an annotated hypergraph
    with a defined role-interaction matrix.

    Note: For stylistic purposes we recreate the weighted adjacency graph for each centrality. 
          This can easily be factored out.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
    
    Output:
        degrees:
    """

    weighted_edges = annotated_hypergraph.to_weighted_projection()

    return {key:sum(targets.values()) for key,targets in weighted_edges.items()}

def eigenvector_centrality(annotated_hypergraph):
    """
    Returns the weighted eigenvector centrality for each node in an annotated hypergraph
    with a defined role-interaction matrix.

    Note: For stylistic purposes we recreate the weighted adjacency graph for each centrality. 
          This can easily be factored out.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
    
    Output:
        degrees:
    """
    raise NotImplementedError

def modularity(annotated_hypergraph, return_communities=False):
    """
    Returns the optimal modularity score for an annotated hypergraph
    with a defined role-interaction matrix.

    Note: For stylistic purposes we recreate the weighted adjacency graph for each centrality. 
          This can easily be factored out.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
    
    Output:
        degrees:
    """
    raise NotImplementedError