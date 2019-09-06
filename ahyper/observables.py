from itertools import groupby, chain
from collections import Counter

import networkx as nx

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

def _degree_centrality(weighted_projection):
    return {key:sum(targets.values()) for key,targets in weighted_projection.items()}

def degree_centrality(annotated_hypergraph):
    """
    Returns the weighted degree centrality for each node in an annotated hypergraph
    with a defined role-interaction matrix.

    Note: For stylistic purposes we recreate the weighted adjacency graph for each centrality. 
          This can easily be factored out.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
    
    Output:
        degrees (dict): A dictionary of {node:degree} pairs.
    """

    weighted_projection = annotated_hypergraph.to_weighted_projection()
    return _degree_centrality(weighted_projection)


def _eigenvector_centrality(weighted_projection, **kwargs):
    return nx.eigenvector_centrality(weighted_projection, **kwargs)

def eigenvector_centrality(annotated_hypergraph, **kwargs):
    """
    Returns the weighted eigenvector centrality for each node in an annotated hypergraph
    with a defined role-interaction matrix.

    Note: For stylistic purposes we recreate the weighted adjacency graph for each centrality,
          and create a NetworkX DiGraph. 
          This can easily be factored out.

    Note: Using networkx for simplicity at the moment but can migrate to more efficient
          library if needed.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
    
    Output:
        eigenvector (dict): A dictionary of {node:eigenvector_centrality} pairs.
    """
    weighted_projection = annotated_hypergraph.to_weighted_projection(use_networkx=True)
    
    return _eigenvector_centrality(weighted_projection)
    

def _pagerank_centrality(weighted_projection, **kwargs):
    return nx.pagerank(weighted_projection, **kwargs)

def pagerank_centrality(annotated_hypergraph, **kwargs):
    """
    Returns the weighted PageRank centrality for each node in an annotated hypergraph
    with a defined role-interaction matrix.

    Note: For stylistic purposes we recreate the weighted adjacency graph for each centrality,
          and create a NetworkX DiGraph. 
          This can easily be factored out.

    Note: Using networkx for simplicity at the moment but can migrate to more efficient
          library if needed.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
    
    Output:
        pagerank (dict): A dictionary of {node:pagerank_centrality} pairs.
    """
    weighted_projection = annotated_hypergraph.to_weighted_projection(use_networkx=True)

    return _pagerank_centrality(weighted_projection, **kwargs)


def _connected_components(weighted_projection):
    nx.number_weakly_connected_components(weighted_projection)

def connected_components(annotated_hypergraph):
    """
    Returns the number of connected components of an annotated hypergraph.

    Note: For stylistic purposes we recreate the weighted adjacency graph for each centrality. 
          This can easily be factored out.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
    
    Output:
        connected_components (int): The number of connected components.
    """
    weighted_projection = annotated_hypergraph.to_weighted_projection(use_networkx=True)

    return _connected_components(weighted_projection)