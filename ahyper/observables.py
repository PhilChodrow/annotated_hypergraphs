from itertools import groupby, chain
from collections import Counter

import networkx as nx

from .utils import normalise_counters
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict
import pandas as pd

from scipy.linalg import eigh


def local_role_density(
    annotated_hypergraph, include_focus=False, absolute_values=False, as_array=False
):
    """
    Calculates the density of each role within a 1-step neighbourhood
    of a node, for all nodes.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
        include_focus [Bool]: If True, includes the roles of the focal node
                              in th calculation.
        absolute_values [Bool]: If True, returns role counts rather than densities.
        as_array [Bool]: If True, return an array rather than a Counter. 
    Returns:
        role_densities []: An array of dimension (# nodes x # roles) 
                           describing the density of each role.
    """
    A = annotated_hypergraph

    def get_counts(group):
        return Counter([x.role for x in group])

    by_edge = {
        eid: get_counts(v)
        for eid, v in groupby(
            sorted(A.IL, key=lambda x: x.eid, reverse=True), lambda x: x.eid
        )
    }

    densities = {}
    for incidence in A.IL:
        densities[incidence.nid] = (
            densities.get(incidence.nid, Counter()) + by_edge[incidence.eid]
        )

        if not include_focus:
            densities[incidence.nid] = densities.get(
                incidence.nid, Counter()
            ) - Counter([incidence.role])

    keys = set(chain.from_iterable(densities.values()))
    for item in densities.values():
        item.update({key: 0 for key in keys if key not in item})

    if not absolute_values:
        normalise_counters(densities)

    if as_array:
        densities = to_matrix(densities, A)

    return densities


def node_role_participation(
    annotated_hypergraph, absolute_values=False, as_array=False
):
    """
    Calculates the proportion of instances where each node is in each role.

    Input:
        annotated_hypergraph [AnnotatedHypergraph]: An annotated hypergraph.
        absolute_values [Bool]: If True, returns role counts rather than densities.
        as_array [Bool]: If True, return an array rather than a Counter. 
    Returns:
        role_densities []: An array of dimension (# nodes x # roles) 
                           describing the participation of each role.
    """
    A = annotated_hypergraph

    def get_counts(group):
        return Counter([x.role for x in group])

    densities = {
        nid: get_counts(v)
        for nid, v in groupby(sorted(A.IL, key=lambda x: x.nid), lambda x: x.nid)
    }

    keys = set(chain.from_iterable(densities.values()))
    for item in densities.values():
        item.update({key: 0 for key in keys if key not in item})

    if not absolute_values:
        normalise_counters(densities)

    if as_array:
        densities = to_matrix(densities, A)

    return densities


def _degree_centrality(weighted_projection):
    if isinstance(weighted_projection, nx.DiGraph):
        degrees = {
            node: weighted_projection.out_degree(node, weight="weight")
            for node in weighted_projection.nodes()
        }
        return degrees
    elif isinstance(weighted_projection, dict):
        return NotImplementedError("Specify one of 'use_networkx' or 'use_graphtool'")
    else:  # graphtool
        G = weighted_projection
        degrees = G.degree_property_map("out", weight=G.ep.weights)
        degrees = {G.vp.node_labels[v]: degrees[v] for v in G.vertices()}
        return degrees


def degree_centrality(annotated_hypergraph, **kwargs):
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

    weighted_projection = annotated_hypergraph.to_weighted_projection(**kwargs)
    return _degree_centrality(weighted_projection)


def _eigenvector_centrality(weighted_projection, **kwargs):
    if isinstance(weighted_projection, nx.DiGraph):
        return nx.eigenvector_centrality(weighted_projection, **kwargs)
    else:
        from graph_tool.centrality import eigenvector

        G = weighted_projection
        _, eigenv = eigenvector(G, weight=G.ep.weights)
        ev = {G.vp.node_labels[v]: eigenv[v] for v in G.vertices()}
        return ev


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

    return _eigenvector_centrality(weighted_projection, **kwargs)


def _pagerank_centrality(weighted_projection, **kwargs):
    if isinstance(weighted_projection, nx.DiGraph):
        return nx.pagerank(weighted_projection, **kwargs)
    else:
        from graph_tool.centrality import pagerank

        G = weighted_projection
        pr = pagerank(G, weight=G.ep.weights)
        pr = {G.vp.node_labels[v]: pr[v] for v in G.vertices()}
        return pr


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
    if isinstance(weighted_projection, nx.DiGraph):
        return nx.number_weakly_connected_components(weighted_projection)
    else:
        from graph_tool.topology import label_components

        G = weighted_projection
        G.set_directed(False)
        _, comps = label_components(G)
        G.set_directed(True)
        return len(comps)


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


def random_walk(
    G, n_steps, alpha=0, nonbacktracking=False, alpha_ve=None, alpha_ev=None
):
    """
    Conduct a random walk on a network G, optionally with teleportation parameter alpha and nonbacktracking.

    Note: Assumes a randomly chosen starting node.

    Input:
        G (nx.Graph/nx.DiGraph): A graph to conduct the random walk.
        n_steps (int): The number of steps the random walk should take.
        alpha (float): 
        nonbacktracking (bool):
        alpha_ve (float):
        alpha_ev (float):

    Output:
        V (np.array):

    TODO: Check this works (test) and make it clear differences between alphas.
    """

    V = np.zeros(n_steps)

    nodes = np.array(list(G.nodes()))
    nodes = nodes[nodes >= 0]
    n = len(nodes)
    v = np.random.choice(nodes)

    # v2 eventually holds where we went last time.
    v2 = v

    for i in range(n_steps):
        N = G[v]
        v_ = np.random.choice(list(N))
        if nonbacktracking:
            while v_ == v2:
                v_ = np.random.choice(list(N))

        role = G[v][v_]["role"]

        if v < 0:
            alpha = alpha_ev
        else:
            alpha = alpha_ve

        if np.random.rand() < alpha[role]:
            i_ = np.random.randint(n)
            v_ = nodes[i_]

        v2 = v
        v = v_
        V[i] = v

    return V


def random_walk_pagerank(
    annotated_hypergraph,
    n_steps,
    nonbacktracking=False,
    alpha_1=None,
    alpha_2=None,
    return_path=False,
):
    """
    Calculate the random walk PageRank for each node in the network
    by sampling.

    Input:
        annotated_hypergraph (AnnotatedHypergraph): The hypergraph to apply PageRank to.
        n_steps (int): The number of steps to take in the random walk.
        nonbacktracking (bool): If True, the random walk is non-backtracking.
        alpha_1 (float):
        alpha_2 (float):
        return_path (bool): If True, returns also the path of the random walk.

    Output:
        pagerank (dict): A dictionary of node:pagerank pairs.
        V (np.array) [optional]: The random walk trajectory. 

    """

    A = annotated_hypergraph

    G = A.to_bipartite_graph()

    V = random_walk(
        G, n_steps, nonbacktracking=nonbacktracking, alpha_ve=alpha_1, alpha_ev=alpha_2
    )

    counts = np.unique(V, return_counts=True)
    ix = counts[0] >= 0
    v = counts[1][ix]
    v = v / v.sum()
    labels = counts[0][ix]
    d = {labels[i]: v[i] for i in range(len(v))}
    if return_path:
        return d, V
    else:
        return d


def _assortativity(annotated_hypergraph, n_samples, by_role=True, spearman=True):
    """Helper function."""
    assort = assortativity(annotated_hypergraph, n_samples, by_role=True, spearman=True)
    assort_dict = {}
    role_map = dict(annotated_hypergraph.role_map)
    for ix, row in assort.iterrows():
        key = f"{role_map[int(ix[0])]}_{role_map[int(ix[1])]}"
        assort_dict[key] = row.deg_2
    return assort_dict


def assortativity(annotated_hypergraph, n_samples, by_role=True, spearman=True):
    """
    Return a stochastic approximation of the assortativity between nodes, optionally by roles. 

    Notes: 
        Not quite the same thing as the standard degree-assortativity coefficient for dyadic graphs due to the role of hyperedges. 
        Generalizes the uniform measure in the "Configuration Models of Random Hypergraphs"
        
    Input: 
        annotated_hypergraph (AnnotatedHypergraph): The annotated hypergraph on which to measure
        n_samples (int): The number of hyperedges to sample
        by_role (bool): If True, break out all the correlations by pairs of node roles. 
        spearman (bool): If True, replace degrees by their ranks (within each pair of node_roles if 
                         by_role)
        
    Output: 
        A pd.DataFrame containing degree-correlation coefficients. 
    """

    A = annotated_hypergraph
    # first, construct a lookup giving the number of edges incident to each pair of nodes, by role if specified.
    def discount_lookup(role_1, role_2, by_role=by_role):

        weighted_edges = defaultdict(lambda: defaultdict(lambda: 0.0))
        for eid, edge in groupby(A.get_IL(), lambda x: x.eid):
            edge = list(edge)
            for a, b in combinations(edge, 2):
                if by_role:
                    if (
                        (a.role == role_1)
                        and (b.role == role_2)
                        or (a.role == role_2)
                        and (b.role == role_1)
                    ):
                        weighted_edges[a.nid][b.nid] += 1
                else:
                    weighted_edges[a.nid][b.nid] += 1

        return weighted_edges

    D = {
        (role_1, role_2): discount_lookup(role_1, role_2)
        for role_1, role_2 in permutations(A.roles, 2)
    }
    D = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)), D)

    # next, construct the degree lookup, again by role if specified.
    degs = A.node_degrees(by_role=by_role)

    # containers for the data generated in the loop
    if isinstance(A.roles[0], int):
        max_len = max(len(str(role)) for role in A.roles)
    else:
        max_len = max(len(role) for role in A.roles)

    dtype = "S" + str(max_len)

    role_1 = np.empty(2 * n_samples, dtype=dtype)
    role_2 = np.empty(2 * n_samples, dtype=dtype)
    deg_1 = np.zeros(2 * n_samples)
    deg_2 = np.zeros(2 * n_samples)

    # generate a dict which for each eid gives the list of NodeEdgeIncidences in that edge.
    IL = A.get_IL()
    IL.sort(key=lambda e: (e.eid, e.role))
    edges = groupby(IL, lambda e: (e.eid))
    edges = {k: list(v) for k, v in edges}

    # main loop
    for k in 2 * np.arange(n_samples):

        # choose a random edge and two nodes on it.
        # i = np.random.randint(1, A.m+1)
        i = np.random.randint(0, A.m)

        edge = edges[i]

        i_ = np.random.randint(len(edge))
        j_ = np.random.randint(len(edge))

        while i_ == j_:
            i_ = np.random.randint(len(edge))
            j_ = np.random.randint(len(edge))

        u = edges[i][i_].nid
        v = edges[i][j_].nid

        u_role = edges[i][i_].role
        v_role = edges[i][j_].role

        # compute the discount -- this is the number of edges between u and v themselves.
        discount = D[(u_role, v_role)][u][v]

        role_1[k] = u_role
        role_2[k] = v_role
        role_1[k + 1] = v_role
        role_2[k + 1] = u_role

        if by_role:
            deg_1[k] = degs[u][u_role] - discount
            deg_2[k] = degs[v][v_role] - discount

            deg_1[k + 1] = degs[u][u_role] - discount
            deg_2[k + 1] = degs[v][v_role] - discount
        else:
            deg_1[k] = degs[u] - discount
            deg_2[k] = degs[v] - discount
            deg_1[k + 1] = degs[u] - discount
            deg_2[k + 1] = degs[v] - discount

    # construct a DataFrame with the results.

    role_1 = role_1.astype("U" + str(max_len))
    role_2 = role_2.astype("U" + str(max_len))

    df = pd.DataFrame(
        {"role_1": role_1, "role_2": role_2, "deg_1": deg_1, "deg_2": deg_2}
    )

    df = df.astype({"deg_1": "int32", "deg_2": "int32"})

    if by_role:
        grouped = df.groupby(["role_1", "role_2"])
        if spearman:
            df["deg_1"] = grouped["deg_1"].rank()
            df["deg_2"] = grouped["deg_2"].rank()

        df = df.groupby(["role_1", "role_2"])
        corrs = df.corr().iloc[0::2, -1]
    else:
        if spearman:
            df["deg_1"] = df["deg_1"].rank()
            df["deg_2"] = df["deg_2"].rank()
        corrs = df.corr().iloc[0::2, -1]

    return pd.DataFrame(corrs)


def multiway_spectral(A, A0, k):

    # implementation(?) of Zhang + Newman on multiway spectral partitioning

    B = A - A0

    n = B.shape[0]
    m = A.sum() / 2

    eigen = eigh(B, eigvals=(n - k, n - 1))

    lam = eigen[0]
    U = eigen[1]

    # vertex vectors (eq (10))

    r = np.sqrt(lam) * U

    ix = np.random.randint(0, n, k)

    # initialize community vectors
    R = r[ix, :]

    # random assignments
    ix = np.random.randint(0, k, n)
    L = np.zeros((n, k))
    L[np.arange(n), ix] = 1
    L_old = L.copy()

    ell_old = np.zeros(n)

    for j in range(100):
        # # until convergence

        Rr = np.dot(r, R.T)

        # current assignment
        ell = np.argmax(Rr, axis=1)
        L = np.zeros((n, k))
        L[np.arange(n), ell] = 1
        ell = L.argmax(axis=1)
        R = np.dot(L.T, r)

        obj = (1 / (2 * m)) * (R ** 2).sum()

        if (ell == ell_old).mean() == 1:
            break
        ell_old = ell.copy()

    return (ell, obj)


def mutual_information_of_roles(annotated_hypergraph):
    """
    Calculates the mutual information between an individuals' roles and the roles of their neighbours.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        M = local_role_density(annotated_hypergraph, as_array=True)
        p_ = M.mean(axis=0)
        return 1 / A.n * np.nansum((np.log(M / p_) * M))


def MI(X, Y, normalize=False, return_joint=False):
    # assumes X and Y are vectors of integer labels of the same length and on the same alphabet space
    n = len(X)
    k = max(X.max(), Y.max())
    joint = np.zeros((k + 1, k + 1))
    p_x = np.zeros(k + 1)
    p_y = np.zeros(k + 1)

    for i in range(n):
        joint[X[i], Y[i]] += 1
        p_x[X[i]] += 1
        p_y[Y[i]] += 1
    joint = joint / joint.sum()

    p_x = p_x / p_x.sum()
    p_y = p_y / p_y.sum()

    mat = joint * np.log(joint / np.outer(p_x, p_y))
    mat[np.isnan(mat)] = 0

    I_XY = mat.sum()

    if normalize:
        H_x = -p_x * np.log(p_x)
        H_x[np.isnan(H_x)] = 0
        H_x = H_x.sum()

        H_y = -p_y * np.log(p_y)
        H_y[np.isnan(H_y)] = 0
        H_y = H_y.sum()

        NMI = I_XY / (0.5 * (H_x + H_y))
        out = NMI
    else:
        out = I_XY

    if return_joint:
        return (out, joint)
    else:
        return out


def to_matrix(C, A):
    """
        Convert a counter, such as that returned by local_role_densities(), to a matrix in which each column corresponds to a role of A. 
    """
    M = np.zeros((len(C), len(A.roles)))

    k = list(C.keys())
    for i in range(len(k)):
        for j in range(len(A.roles)):
            M[k[i], j] = C[i][A.roles[j]]
    return M
