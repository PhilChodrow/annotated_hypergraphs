from ahyper.utils import *
from collections import Counter

class annotated_hypergraph:
    
    def __init__(self, records, roles):
        
        self.roles = roles
        self.IL = incidence_list_from_records(records, self.roles)
        self.IL.sort(key = lambda x: x[1]) # sort by roles for now
        self.node_list = np.unique([e[0] for e in self.IL])
        self.edge_list = np.unique([e[2] for e in self.IL])
        self.n = len(self.node_list)
        self.m = len(self.edge_list)
        
    def get_node_list(self):
        return self.node_list
    
    def get_edge_list(self):
        return self.edge_list
    
    def stub_labeled_MCMC(self, n_steps = 1):
        
        by_role = [list(v) for role, v in groupby(self.IL, lambda x: x[1])]
        
        # distribute steps over the role partition, using coupon-collector heuristic
        N = np.array([len(l)  for l in by_role])
        steps = ((N*np.log(N)) / (N*np.log(N).sum())*n_steps).astype(int)
        
        for i in range(len(self.roles)):
            for k in range(steps[i]):
                swap_step(by_role[i])
        
        self.IL = [e for role in by_role for e in role]
    
    def get_IL(self):
        return(sorted(self.IL, key = lambda x: x[2], reverse = True))
    
    def get_records(self):
        return records_from_incidence_list(self.IL, role_fields = self.roles)
    
    def node_degrees(self, by_role = False):
        
        if by_role:
            br = {role: list(v) for role, v in groupby(self.IL, lambda x: x[1])}
            DT = {role : Counter([e[0] for e in br[role]]) for role in self.roles}
            D = {k : {role : DT[role][k] for role in self.roles} for k in self.node_list}
            return(D)
            
        else:
            V = [e[0] for e in self.IL]
            return(dict(Counter(V)))
        
    def edge_dimensions(self, by_role = False):
        
        if by_role:
            br =  {role: list(v) for role, v in groupby(self.IL, lambda x: x[1])}
            DT = {role : Counter([e[2] for e in br[role]]) for role in self.roles}
            D = {k : {role : DT[role][k] for role in self.roles} for k in self.edge_list}
            return(D)
        else:
            E = [e[2] for e in self.IL]
            return(Counter(E))
    
def bipartite_edge_swap(e0, e1):
    '''
    assumes e0 and e1 members of node-edge incidence list with same role
    '''
    
    f0 = [e0[0], e1[1], e1[2], e1[3]]
    f1 = [e1[0], e0[1], e0[2], e0[3]]

    return(f1, f0)

def swap_step(il):
    n = len(il)
    i,j = np.random.randint(0, n, 2)
    while il[i][2] == il[j][2]: 
        i,j = np.random.randint(0, n, 2)        
    il[i], il[j] = bipartite_edge_swap(il[i], il[j])