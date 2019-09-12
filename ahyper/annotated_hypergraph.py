from .utils import *


from collections import Counter, defaultdict
from itertools import permutations

from collections import Counter
from copy import deepcopy
from random import shuffle

import networkx as nx
import numpy as np


class AnnotatedHypergraph(object):
    
    def __init__(self, records, roles):
        """
        Construct an annotated hypergraph from records.

        Input records should be of the form
            {'role_1':[node1, node 2],
            'role_2':[node 3],
            }

        Input records can also include extra keys that will be stored
        as metadata for each edge.

        Input:
            records [list]: A list of records (JSON-like)
            roles [list]: A list of role labels
        """

        # Assign edge ids if not already present
        if records[0].get('eid') is None:
            for i in range(len(records)):
                records[i]['eid'] = i
        
        self.roles = roles
        self.IL = incidence_list_from_records(records, self.roles)
        self.IL.sort(key = lambda x: x.role) # sort by roles for now
        
#         self.relabel()
        
        self.set_states()

    def set_states(self):
        self.node_list = np.unique([e.nid for e in self.IL])
        self.edge_list = np.unique([e.eid for e in self.IL])
        self.n = len(self.node_list)
        self.m = len(self.edge_list)
        self.R = None
        

    def get_node_list(self):
        """"""
        return self.node_list
    
    def get_edge_list(self):
        """"""
        return self.edge_list
    
    def MCMC(self, n_steps = 1, avoid_degeneracy = True, **kwargs):
        if avoid_degeneracy:
            alg = self.degeneracy_avoiding_MCMC
        else:
            alg = self.stub_labeled_MCMC
        
        alg(n_steps, **kwargs)
    
    def stub_labeled_MCMC(self, n_steps = 1):
        """
        Can create degeneracies, probably deprecated
        """
        
        self.IL.sort(key = lambda x: x.role)
        by_role = [list(v) for role, v in groupby(self.IL, lambda x: x.role)]
        
        # distribute steps over the role partition, using coupon-collector heuristic
        N = np.array([len(l)  for l in by_role])
        steps = ((N*np.log(N)) / (N*np.log(N).sum())*n_steps).astype(int)
        
        for i in range(len(self.roles)):
            for k in range(steps[i]):
                swap_step(by_role[i])
        
        self.IL = [e for role in by_role for e in role]
    
    def degeneracy_avoiding_MCMC(self, n_steps = 1, verbose = True, role_labels = True):
        '''
        Avoids creating edges in which the same node appears multiple times. 
        Some properties need checking, but should be equivalent to stub-matching conditioned on nondegeneracy. 
        '''
        
        # prepare: easier to work on a transformed data structure in which incidences are grouped into hyper edges
        self.IL.sort(key = lambda e: (e.eid, e.role))
        grouped = groupby(self.IL, lambda e: (e.eid))
        edges = {k: list(v) for k, v in grouped}
        
        k_rejected = 0
        N = 0
        
        while(N < n_steps):
            
            # select two random hyperedges
            i, j = np.random.randint(0, self.m, 2)
            E0, E1 = edges[i], edges[j] 
            
            # select a random node-edge incidence from each
            k = np.random.randint(len(E0))
            l = np.random.randint(len(E1))
            
            # if the two node-edge incidences have different roles, then try again
            if role_labels: 
	            if E0[k].role != E1[l].role:
	                k_rejected += 1
            
            else:
                # construct the proposal swap 

                E0_prop = E0.copy()
                E0_prop[k] = NodeEdgeIncidence(E1[l].nid, E0[k].role, E0[k].eid, E0[k].meta)

                E1_prop = E1.copy()
                E1_prop[l] = NodeEdgeIncidence(E0[k].nid, E1[l].role, E1[l].eid, E1[l].meta)
                                       
                # if either of the edges would become degenerate, reject the proposal
                if (check_degenerate(E0_prop) or check_degenerate(E1_prop)):
                    k_rejected += 1
                # otherwise, accept the proposal
                else:
                    edges[i] = E0_prop                        
                    edges[j] = E1_prop
                    N += 1
                        
        # update self.IL
        self.IL = [e for E in edges for e in edges[E]]
        self.IL.sort(key = lambda x: x.role)
            
        if verbose: 
            print(str(n_steps) + ' steps taken, ' + str(k_rejected) + ' steps rejected.')
            
    
    def get_IL(self):
        """"""
        return(sorted(self.IL, key = lambda x: x.eid, reverse = True))
    
    def get_records(self):
        """"""
        return records_from_incidence_list(self.IL, role_fields = self.roles)
    
    def node_degrees(self, by_role = False):
        """"""
        self.IL.sort(key = lambda x: x.role)
        if by_role:
            br = {role: list(v) for role, v in groupby(self.IL, lambda x: x.role)}
            DT = {role : Counter([e.nid for e in br[role]]) for role in self.roles}
            D = {k : {role : DT[role][k] for role in self.roles} for k in self.node_list}
            return(D)
            
        else:
            V = [e.nid for e in self.IL]
            return(dict(Counter(V)))
        
    def edge_dimensions(self, by_role = False):
        """"""
        
        self.IL.sort(key = lambda x: x.role)
        if by_role:
            br =  {role: list(v) for role, v in groupby(self.IL, lambda x: x.role)}
            DT = {role : Counter([e.eid for e in br[role]]) for role in self.roles}
            D = {k : {role : DT[role][k] for role in self.roles} for k in self.edge_list}
            return(D)
        else:
            E = [e.eid for e in self.IL]
            return(Counter(E))


    def assign_role_interaction_matrix(self, R=None):
        """
        Assigns a role-interaction matrix to the annotated hypergraph.

        The R_ij defines the weight which a node in role i interacts with a node in role j
        when they belong to the same edge.

        Input:
            R (np.array): An array with role interaction weights. Must be of the same length
                          and in the same order as AnnotatedHypergraph.roles.

        TODO: Extend this to allow edge-dependent role-interaction matrices.
        """

        num_roles = len(self.roles)
        if R is not None:
            assert R.shape[0] == num_roles
            assert R.shape[1] == num_roles
            self.R = R
        else:
            self.R = np.ones(shape=(num_roles,num_roles))

    def to_weighted_projection(self):
        """
        Projects an annotated hypergraph to a weighted, directed graph.

        If role-interaction matrix has not been defined (through
        self.assign_role_interaction_matrix) then all interactions will be assigned a
        weight of one.

        Input:
            None
        
        Output:
            weighted_edges (dict): A dictionary containing all source nodes as keys.
                                   The values are dictionaries of targets which in turn
                                   contain weights of interaction.
        """
        weighted_edges = defaultdict(lambda: defaultdict(lambda: 0.0))
        role_map = {role:ix for ix,role in enumerate(self.roles)}

        # Default behaviour if R has not been defined.
        if self.R is None:
            self.assign_role_interaction_matrix()

        for eid, edge in groupby(self.get_IL(), lambda x: x.eid):
            edge = list(edge)
            for a,b in permutations(edge, 2):
                weighted_edges[a.nid][b.nid] += self.R[role_map[a.role], role_map[b.role]]

        return weighted_edges

    
    def count_degeneracies(self):
        """Return the number of edges in which the same node appears multiple times"""
        self.IL.sort(key = lambda x: x.eid)
        by_edges = [list(v) for eid, v in groupby(self.IL, lambda x: x.eid)]
        
        return(sum([check_degenerate(E) for E in by_edges]))
        
    
    def remove_degeneracies(self, precedence):
        '''
        Removes entries from self.IL in order of precedence until each node appears only once in each edge.
        Roles with higher precedence are retained.  
        Precedence: a dict of the form {role : p}, lower p -> higher precedence
        May be overaggressive in  node removal -- further tests necessary 
        '''
        self.IL.sort(key = lambda x: (x.eid, x.nid, precedence[x.role]))
        grouped = [list(v) for eid, v in groupby(self.IL, lambda x: x.eid)]
        
        IL_ = []

        for E in grouped:
            E_ = []
            for e in E:
                if e.nid not in [e.nid for e in E_]:
                    E_.append(e)
            IL_.append(E_)
        
        IL_ = [e for E in IL_ for e in E]
            
        n_removed = len(self.IL) - len(IL_)
        print('Removed ' + str(n_removed) + ' node-edge incidences')
        self.IL = IL_
        self.IL.sort(key = lambda x: x.role)
        self.relabel()

    def remove_singletons(self):
        '''
        Removes entries from self.IL if the corresponding edge contains only one node. 
        '''
        D = self.edge_dimensions()
        to_remove = []
        for e in self.IL:
            if D[e.eid] == 1:
                to_remove.append(e)
        k_removed = len(to_remove)
        for e in to_remove:
            self.IL.remove(e)
        self.relabel()
        self.set_states()
        print('Removed '  + str(k_removed) + ' singletons.')
        
    def relabel(self):
        D = IL_to_dict(self.IL)
        
        def relabel_by_field(D, field):
            
            D.sort(key = lambda x: x[field])
            j   = 0
            old = 0
            for e in D:    
                if e[field] != old: 
                    old = e[field]
                    j += 1
                e[field] = j
            return(D)
        
        D = relabel_by_field(D, 'eid')
        D = relabel_by_field(D, 'nid')
                
        self.IL = dict_to_IL(D)
        self.set_states()
        
    def stub_matching(self):
        '''
        Return a randomized version of self constructed according to the naive stub-matching algorithm. 
        Preserves node-role and edge-role matrices, but generally introduces degeneracies. 
        '''
        
        dims = self.edge_dimensions(by_role = True)
        
        stubs = [(e.nid, e.role) for e in self.IL]
        shuffle(stubs)
        stubs.sort(key = lambda e: e[1])
        stubs = {r: list(s) for r, s in groupby(stubs, key = lambda e: e[1])}
        
        a = deepcopy(self)
        IL_ = []
        
        for e in dims: 
            for r in self.roles: 
                for i in range(dims[e][r]):
                    to_add = stubs[r].pop(0)
                    e_ = NodeEdgeIncidence(nid = to_add[0], role = to_add[1], eid = e, meta = None)
                    IL_.append(e_)
        a.IL = IL_
        a.set_states()
        return(a)
    
    def bipartite_graph(self):
        '''
        return an nx.Graph() in which both nodes and edges occur as nodes, and are linked according to incidence. 
        '''
        ebunch = [(e.nid, -e.eid, {'role' : e.role}) for e in self.get_IL()]
        G = nx.Graph()
        G.add_edges_from(ebunch)
        return(G)
    
    

        
        

def bipartite_edge_swap(e0, e1):
    """
    Creates two new swapped edges by permuting the node ids.

    Used under the assumption that e0 and e1 are members of node-edge incidence 
    list with same role, although not explicitly checked.
    """
    
    f0 = NodeEdgeIncidence(e0.nid, e1.role, e1.eid, e1.meta)
    f1 = NodeEdgeIncidence(e1.nid, e0.role, e0.eid, e0.meta)

    return(f1, f0)

def swap_step(il):
    """
    Swap two node-edge incidence entries in the node-edge incidence list.

    Technical node: entries are replaced by new copies of the data.
    """
    n = len(il)
    i,j = np.random.randint(0, n, 2)
    while il[i].eid == il[j].eid: 
        i,j = np.random.randint(0, n, 2)        
    il[i], il[j] = bipartite_edge_swap(il[i], il[j])

    
def check_degenerate(E):
    '''E is a set of node-edge incidences corresponding to a single edge'''
    E_distinct = set([e.nid for e in E])
    return(len(E_distinct) != len(E))






    
    
