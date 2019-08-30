from .utils import *
from collections import Counter

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
                records[i]['eid'] = -(i+1)
        
        self.roles = roles
        self.IL = incidence_list_from_records(records, self.roles)
        self.IL.sort(key = lambda x: x.role) # sort by roles for now
        self.node_list = np.unique([e.nid for e in self.IL])
        self.edge_list = np.unique([e.eid for e in self.IL])
        self.n = len(self.node_list)
        self.m = len(self.edge_list)
        
    def get_node_list(self):
        """"""
        return self.node_list
    
    def get_edge_list(self):
        """"""
        return self.edge_list
    
    def stub_labeled_MCMC(self, n_steps = 1):
        """"""
        
        by_role = [list(v) for role, v in groupby(self.IL, lambda x: x.role)]
        
        # distribute steps over the role partition, using coupon-collector heuristic
        N = np.array([len(l)  for l in by_role])
        steps = ((N*np.log(N)) / (N*np.log(N).sum())*n_steps).astype(int)
        
        for i in range(len(self.roles)):
            for k in range(steps[i]):
                swap_step(by_role[i])
        
        self.IL = [e for role in by_role for e in role]
    
    def get_IL(self):
        """"""
        return(sorted(self.IL, key = lambda x: x.eid, reverse = True))
    
    def get_records(self):
        """"""
        return records_from_incidence_list(self.IL, role_fields = self.roles)
    
    def node_degrees(self, by_role = False):
        """"""

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
        
        if by_role:
            br =  {role: list(v) for role, v in groupby(self.IL, lambda x: x.role)}
            DT = {role : Counter([e.eid for e in br[role]]) for role in self.roles}
            D = {k : {role : DT[role][k] for role in self.roles} for k in self.edge_list}
            return(D)
        else:
            E = [e.eid for e in self.IL]
            return(Counter(E))
    
    def count_degeneracies(self):
        """Return the number of edges in which the same node appears multiple times"""
        edges = {e: list(v) for e, v in groupby(self.IL, lambda x: x[2])}

        def check_degenerate(eid):
            nodes = [v[0] for v in edges[eid]]
            return(len(set(nodes)) != len(nodes))
        
        return(sum([check_degenerate(eid) for eid in self.edge_list]))
    
    def remove_degeneracies(self, precedence):
        '''
        Removes entries from self.IL in order of precedence until each node appears only once in each edge.
        Roles with higher precedence are retained.  
        Precedence: a dict of the form {role : p}, lower p -> higher precedence
        May be overaggressive in  node removal -- further tests needed. 
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
        print('Removed ' + str(n_removed) + ' degeneracies.')
        self.IL = IL_
        self.IL.sort(key = lambda x: x.role)

def bipartite_edge_swap(e0, e1):
    """
    Creates two new swapped edges by permuting the node ids.

    Used under the assumption that e0 and e1 are members of node-edge incidence 
    list with same role, although not explicitly checked.
    """
    
    f0 = NodeEdgeIncidence(e0.nid, e1.role, e1.eid, e1.meta)
    f1 = NodeEdgeIncidence(e1.nid, e0.role, e0.eid, e0.meta)

    return(f1, f0)

def swap_step(il, allow_degeneracies = False):
    """
    Swap two node-edge incidence entries in the node-edge incidence list.

    Technical node: entries are replaced by new copies of the data.
    """
    n = len(il)
    i,j = np.random.randint(0, n, 2)
    while il[i].eid == il[j].eid: 
        i,j = np.random.randint(0, n, 2)        
    il[i], il[j] = bipartite_edge_swap(il[i], il[j])