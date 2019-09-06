from collections import defaultdict

def shuffled_ensemble_features(annotated_hypergraph, 
                               shuffle_fraction, 
                               num_shuffles, 
                               features,
                               burn_fraction=None,
                               verbose=False):
    """
    Calculate distribution of features for an ensemble of shuffled graphs.
    
    Input:
        annotated_hypergraph (AnnotatedHypergraph):
        shuffle_fraction (float): The fraction of all stubs to be shuffled each iteration
        num_shuffles (int): The number of iterations
        features (dict): The feature specification (see below)
        burn_fraction (float): The fraction of stubs to be shuffled before data collection
        verbose (bool): If True, prints progress. 

    Output:
        feature_store (dict): All features, indexed by iteration number.

    Features should be specified as:

    >>> features = {'feature1':{'func':function_for_feature1,
    >>>                         'acts_on':'annotated_hypergraph',
    >>>                         'kwargs':{'kw1':val1}
    >>>                        }
    >>>            }
    """
    
    A = annotated_hypergraph
    W = A.to_weighted_projection(use_networkx=True)
    num_stubs = len(A.IL)
    
    feature_store = defaultdict(dict)
    
    if burn_fraction is not None:
        A.stub_labeled_MCMC(n_steps=int(shuffle_fraction*num_stubs))
    
    for ix in range(num_shuffles):
        
        if verbose and (ix % (num_shuffles//10)) == 0: print(str(ix)+'%', end='\r', flush=True)
                    
        A.stub_labeled_MCMC(n_steps=int(shuffle_fraction*num_stubs))
        
        for feature,f in features.items():
            
            if f['acts_on'] == 'annotated_hypergraph':
                feature_store[ix][feature] = f['func'](A, **f['kwargs'])
            elif f['acts_on'] == 'weighted_projection':
                feature_store[ix][feature] = f['func'](W, **f['kwargs'])
            
    return feature_store