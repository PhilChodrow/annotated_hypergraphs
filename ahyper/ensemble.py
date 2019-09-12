import os, json
from collections import defaultdict

def data_features(annotated_hypergraph,
                  features):
    """
    Calculate features for a single dataset.
    """
    A = annotated_hypergraph
    
    # Check if we need to calculate the weighted projection.
    uses_projection = sum([f['acts_on']=='weighted_projection' for f in features.values()]) > 0
    if uses_projection:
        W = A.to_weighted_projection(use_networkx=True)
    
    feature_store = {}
    
    for feature,f in features.items():
        if f['acts_on'] == 'annotated_hypergraph':
            feature_store[feature] = f['func'](A, **f['kwargs'])
        elif f['acts_on'] == 'weighted_projection':
            feature_store[feature] = f['func'](W, **f['kwargs'])
            
    return feature_store

def shuffled_ensemble_features(annotated_hypergraph, 
                               shuffle_fraction, 
                               num_shuffles, 
                               features,
                               burn_fraction=None,
                               shuffle_algorithm=None,
                               verbose=False):
    """
    Calculate distribution of features for an ensemble of shuffled graphs.
    
    Input:
        annotated_hypergraph (AnnotatedHypergraph):
        shuffle_fraction (float): The fraction of all stubs to be shuffled each iteration
        num_shuffles (int): The number of iterations
        features (dict): The feature specification (see below)
        burn_fraction (float): The fraction of stubs to be shuffled before data collection
        shuffle_algorithm (str): The algorithm used to shuffle data. Must be a function of 
                                 an AnnotatedHypergraph (e.g. A.function())..
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
    num_stubs = len(A.IL)
    
    # Check if we need to calculate the weighted projection.
    uses_projection = sum([f['acts_on']=='weighted_projection' for f in features.values()]) > 0
    
    if shuffle_algorithm is None:
        shuffle = getattr(A,'degeneracy_avoiding_MCMC')
    else:
        shuffle = getattr(A, shuffle_algorithm)
    
    feature_store = defaultdict(dict)
    
    if burn_fraction is not None:
        shuffle(n_steps=int(shuffle_fraction*num_stubs))
    
    for ix in range(num_shuffles):
        
        # Logging
        if verbose and (ix % (num_shuffles//10)) == 0: print(str(ix)+'%', end='\r', flush=True)
                    
        shuffle(n_steps=int(shuffle_fraction*num_stubs))
        
        if uses_projection:
            W = A.to_weighted_projection(use_networkx=True)
        
        for feature,f in features.items():
            if f['acts_on'] == 'annotated_hypergraph':
                feature_store[ix][feature] = f['func'](A, **f['kwargs'])
            elif f['acts_on'] == 'weighted_projection':
                feature_store[ix][feature] = f['func'](W, **f['kwargs'])
            
    return feature_store

def save_feature_study(annotated_hypergraph,
                       data_name,
                       shuffle_fraction, 
                       num_shuffles, 
                       features,
                       burn_fraction=None,
                       role_preserving=True,
                       role_destroying=True
                       ):
    """
    """
    A = annotated_hypergraph
    
    # Make directory if doesn't exist.
    if not os.path.exists(f'./results/{data_name}'):
        os.makedirs(f'./results/{data_name}')
    
    # Make a log of what feature spec was used (although functions are not saved.) 
    with open(f'./results/{data_name}/feature_log.json', 'w') as file:
        file.write(json.dumps(features, default=str, indent=4))
    
    # Features of pure dataset
    data = data_features(A,
                     features)
    data = pd.Series(data)
    data.to_csv(f'./results/{data_name}/original.csv')    
        
    # Respect roles when performing shuffle
    if role_preserving:
        role_preserving_ensemble = shuffled_ensemble_features(A,
                                        shuffle_fraction,
                                        num_shuffles,
                                        features,
                                        burn_fraction=None,
                                        shuffle_algorithm='degeneracy_avoiding_MCMC',  
                                        verbose=False)
        
        role_preserving_ensemble = pd.DataFrame(role_preserving_ensemble).T 
        role_preserving_ensemble.to_csv(f'./results/{data_name}/role_preserving_ensemble.csv', index=False, header=True)

    
    # How data looks when we do not respect roles.
    if role_destroying:
        role_destroying_ensemble = shuffled_ensemble_features(A,
                                        shuffle_fraction,
                                        num_shuffles,
                                        features,
                                        burn_fraction=None,
                                        shuffle_algorithm='degeneracy_avoiding_MCMC',  
                                        verbose=False)

        role_destroying_ensemble = pd.DataFrame(role_destroying_ensemble).T
        role_destroying_ensemble.to_csv(f'./results/{data_name}/role_destroying_ensemble.csv', index=False, header=True)
    
    return None