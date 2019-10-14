import numpy as np
import pandas as pd
import json
from itertools import groupby
from collections import namedtuple

# NodeEdgeIncidence = namedtuple('NodeEdgeIncidence',  ['nid', 'role', 'eid','meta'])


class NodeEdgeIncidence(object):

    __slots__ = ("nid", "role", "eid", "meta")

    def __init__(self, nid, role, eid, meta):

        self.nid = nid
        self.role = role
        self.eid = eid
        self.meta = meta

    def __repr__(self):
        return "NodeEdgeIncidence({})".format(
            ", ".join(
                [
                    "{}={}".format(key, getattr(self, key))
                    for key in self.__class__.__slots__
                ]
            )
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__class__.__slots__}


def incidence_list_from_records(data, role_fields):
    """
    Construct annotated node-edge incidence list from list of records.

    An entry [i,l,e,m] of this list states that "node i has role l in edge e with metadata m"

    Input:
        data [list]: A list of records (JSON-like)
        role_fields [list]: A list of role labels
    Output:
        IL [list]: A list of node-edge incidence records   
    """
    IL = []

    for role in role_fields:
        for record in data:
            for i in record[role]:
                new_row = NodeEdgeIncidence(
                    i, role, record["eid"], {"date": record["date"]}
                )
                IL.append(new_row)
    return IL


def records_from_incidence_list(IL, role_fields):
    """
    Construct list of records from node-edge incidence list.
    """

    IL = sorted(IL, key=lambda x: x.eid, reverse=True)
    chunked = [list(v) for eid, v in groupby(IL, lambda x: x.eid)]

    records = []
    for chunk in chunked:
        record = {role: [] for role in role_fields}
        record["eid"] = chunk[0].eid
        record["date"] = chunk[0].meta[
            "date"
        ]  # TODO: We need to accommodate other fields in the future
        for line in chunk:
            record[line.role].append(line.nid)
        records.append(record)

    return sorted(records, key=lambda x: x["eid"])


def normalise_counters(counters):
    """Normalise a dictionary of counters inplace."""
    for node, d in counters.items():
        total = sum(d.values())
        if total == 0.0:
            counters[node] = None
        else:
            for key in d:
                d[key] /= total


def entropy(iterable):
    """ Calculates the entropy of an iterable. """
    if iterable[0] is None:
        return None
    v = np.array([p for p in iterable if p > 0])
    v = v / v.sum()
    return (-(v * np.log2(v)).sum())


def average_entropy(func):
    """
    Takes a function that returns one or more probability distributions and returns an average entropy.
    """

    def avg_entropy(args, **kwargs):
        density = pd.DataFrame(func(args, **kwargs)).T
        return density.apply(entropy, axis=1).mean()

    return avg_entropy


def average_value(func):
    """
    Converts function output to a Pandas series and takes the mean.
    """

    def avg_value(args, **kwargs):
        return pd.Series(func(args, **kwargs)).mean()

    return avg_value


def variance_value(func):
    """
    Converts function output to a Pandas series and takes the variance.
    """

    def var_value(args, **kwargs):
        return pd.Series(func(args, **kwargs)).var()

    return var_value


def entropy_value(func):
    """
    Converts function output to a Pandas series and calculates the entropy.
    """

    def ent_value(args, **kwargs):
        values = pd.Series(func(args, **kwargs))
        values = values / values.sum()
        return entropy(values)

    return ent_value


def sort_matrix(A, v):
    row_sorted = A[np.argsort(v)]
    col_sorted = row_sorted[:, np.argsort(v)]
    return col_sorted
