from itertools import groupby, chain
from collections import Counter

import networkx as nx

from .utils import normalise_counters
import numpy as np


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
        degrees (dict): A dictionary of {node:degree} pairs.
    """

    weighted_edges = annotated_hypergraph.to_weighted_projection()

    return {key:sum(targets.values()) for key,targets in weighted_edges.items()}

def eigenvector_centrality(annotated_hypergraph):
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
    weighted_edges = annotated_hypergraph.to_weighted_projection()
    
    # Conversion to 
    weighted_edges = {source:{target:{'weight':val} for target,val in values.items()} for source, values in weighted_edges.items()}
    G = nx.DiGraph(weighted_edges)

    return nx.eigenvector_centrality(G)

def pagerank_centrality(annotated_hypergraph):
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
    weighted_edges = annotated_hypergraph.to_weighted_projection()
    
    # Conversion to 
    weighted_edges = {source:{target:{'weight':val} for target,val in values.items()} for source, values in weighted_edges.items()}
    G = nx.DiGraph(weighted_edges)

    return nx.pagerank(G)

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
    
def random_walk(G, n_steps, alpha = 0, nonbacktracking = False, alpha_ve = None, alpha_ev = None):
    '''
    conduct a random walk on network G, optionally with teleportation parameter alpha and nonbacktracking
    '''
    
    V = np.zeros(n_steps)

    nodes = list(G.nodes())
    n = len(nodes)
    v = np.random.choice(nodes)

    # v2 eventually holds where we went last time. 
    v2 = v 
        
    for i in range(n_steps):
        N = G[v]
        v_ = np.random.choice(list(N))
        role = G[v][v_]['role']

        if v < 0:
            alpha = alpha_ev
        else:
            alpha = alpha_ve

        if np.random.rand() < alpha[role]: 
            i_ = np.random.randint(n)
            v_ = nodes[i_]

        if (v_ != v2) or (not nonbacktracking):
            v2 = v
            v = v_
            V[i] = v
    
    return(V)