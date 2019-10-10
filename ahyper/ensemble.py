import os, json
from collections import defaultdict

import pandas as pd


def data_features(annotated_hypergraph, features):
    """
    Calculate features for a single dataset.

    Input:
        annotated_hypergraph (AnnotatedHypergraph):
        features (dict): The feature specification (see below)

    Output:
        feature_store (dict): All features.

    >>> features = {'feature1':{'func':function_for_feature1,
    >>>                         'acts_on':'annotated_hypergraph',
    >>>                         'kwargs':{'kw1':val1}
    >>>                        }
    >>>            }    
    """
    A = annotated_hypergraph

    # Check if we need to calculate the weighted projection.
    uses_projection = (
        sum([f["acts_on"] == "weighted_projection" for f in features.values()]) > 0
    )
    if uses_projection:
        W = A.to_weighted_projection(use_graphtool=True)

    feature_store = {}

    for feature, f in features.items():
        if f["acts_on"] == "annotated_hypergraph":
            result = f["func"](A, **f["kwargs"])
            if isinstance(result, dict):
                for key, val in result.items():
                    feature_store[f"{feature}_{key}"] = val
            else:
                feature_store[feature] = result
        elif f["acts_on"] == "weighted_projection":
            result = f["func"](W, **f["kwargs"])
            if isinstance(result, dict):
                for key, val in result.items():
                    feature_store[f"{feature}_{key}"] = val
            else:
                feature_store[feature] = result

    return feature_store


def shuffled_ensemble_features(
    annotated_hypergraph,
    shuffle_fraction,
    num_shuffles,
    features,
    burn_fraction=None,
    shuffle_algorithm=None,
    verbose=False,
    fail_hard=True,
):
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
        fail_hard (bool): If True, returns an error if a single function returns an error,
                          otherwise treats that calculation as None.

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
    uses_projection = (
        sum([f["acts_on"] == "weighted_projection" for f in features.values()]) > 0
    )

    if shuffle_algorithm is None:
        shuffle = getattr(A, "_degeneracy_avoiding_MCMC")
    else:
        shuffle = getattr(A, shuffle_algorithm)

    feature_store = defaultdict(dict)

    if burn_fraction is not None:
        shuffle(n_steps=int(shuffle_fraction * num_stubs))

    for ix in range(num_shuffles):

        # Logging
        divisor = num_shuffles if num_shuffles < 20 else 20
        if verbose and (ix % (num_shuffles // divisor)) == 0:
            print(str(100 * ix / num_shuffles) + "%", end="\r", flush=True)

        shuffle(n_steps=int(shuffle_fraction * num_stubs))

        if uses_projection:
            W = A.to_weighted_projection(use_graphtool=True)

        for feature, f in features.items():
            try:
                if f["acts_on"] == "annotated_hypergraph":
                    result = f["func"](A, **f["kwargs"])
                    if isinstance(result, dict):
                        for key, val in result.items():
                            feature_store[ix][f"{feature}_{key}"] = val
                    else:
                        feature_store[ix][feature] = result
                elif f["acts_on"] == "weighted_projection":
                    result = f["func"](W, **f["kwargs"])
                    if isinstance(result, dict):
                        for key, val in result.items():
                            feature_store[ix][f"{feature}_{key}"] = val
                    else:
                        feature_store[ix][feature] = result
            except:  # I would use BaseException as e but nx does seem to inherit.
                if fail_hard:
                    raise e
                else:
                    feature_store[ix][feature] = None

    return feature_store


def save_feature_study(
    annotated_hypergraph,
    data_name,
    shuffle_fraction,
    num_shuffles,
    features,
    burn_fraction=None,
    role_preserving=True,
    role_destroying=True,
    root="./results/",
    verbose=False,
    fail_hard=True,
):
    """
    Calculate distribution of features for an ensemble of shuffled graphs.
    
    Input:
        annotated_hypergraph (AnnotatedHypergraph):
        data_name (str): The name of the dataset under study (used as directory).
        shuffle_fraction (float): The fraction of all stubs to be shuffled each iteration
        num_shuffles (int): The number of iterations
        features (dict): The feature specification (see below)
        burn_fraction (float): The fraction of stubs to be shuffled before data collection
        role_preserving (bool): If True, calculates role preserving ensemble (default True).
        role_destroying (bool): If True, calculates role destroying ensemble (default True).
        root (str): Root directory to save files.
        verbose (bool): If True, prints logging messages.
        fail_hard (bool): If True, returns an error if a single function returns an error,
                    otherwise treats that calculation as None.

    Output:
        None : Files are saved to disk. 

    Features should be specified as:

    >>> features = {'feature1':{'func':function_for_feature1,
    >>>                         'acts_on':'annotated_hypergraph',
    >>>                         'kwargs':{'kw1':val1}
    >>>                        }
    >>>            }
    """

    A = annotated_hypergraph

    # Make directory if doesn't exist.
    if not os.path.exists(f"{root}{data_name}"):
        os.makedirs(f"{root}{data_name}")

    # Make a log of what feature spec was used (although functions are not saved.)
    with open(f"{root}{data_name}/feature_log.json", "w") as file:
        file.write(json.dumps(features, default=str, indent=4))

    # Features of pure dataset
    data = data_features(A, features)
    data = pd.Series(data)
    data.to_csv(f"{root}{data_name}/original.csv", header=False)

    # Respect roles when performing shuffle
    if role_preserving:
        if verbose:
            print("Running Role Preserving MCMC...")
        role_preserving_ensemble = shuffled_ensemble_features(
            A,
            shuffle_fraction=shuffle_fraction,
            num_shuffles=num_shuffles,
            features=features,
            burn_fraction=burn_fraction,
            shuffle_algorithm="_degeneracy_avoiding_MCMC",
            verbose=verbose,
            fail_hard=fail_hard,
        )

        role_preserving_ensemble = pd.DataFrame(role_preserving_ensemble).T
        role_preserving_ensemble.to_csv(
            f"{root}{data_name}/role_preserving_ensemble.csv", index=False, header=True
        )

    # How data looks when we do not respect roles.
    if role_destroying:
        if verbose:
            print("Running Role Destroying MCMC...")
        role_destroying_ensemble = shuffled_ensemble_features(
            A,
            shuffle_fraction=shuffle_fraction,
            num_shuffles=num_shuffles,
            features=features,
            burn_fraction=burn_fraction,
            shuffle_algorithm="_MCMC_no_role",
            verbose=verbose,
            fail_hard=fail_hard,
        )

        role_destroying_ensemble = pd.DataFrame(role_destroying_ensemble).T
        role_destroying_ensemble.to_csv(
            f"{root}{data_name}/role_destroying_ensemble.csv", index=False, header=True
        )

    return None
