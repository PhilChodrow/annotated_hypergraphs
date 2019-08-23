import numpy as np
# import pandas as pd
import json
from itertools import groupby


def incidence_list_from_records(data, role_fields):
    '''
    construct annotated node-edge incidence list from list of records
    an entry [i,l,e,t] of this list states that "node i has role l in edge e at time t"
    '''
    IL = []

    for role in role_fields:
        for record in data:
            for i in record[role]:
                new_row = [i, role, record['eid'], record['date']]
                IL.append(new_row)
    return(IL)

def records_from_incidence_list(IL, role_fields):
    '''
    Construct list of records from node-edge incidence list
    Should be the inverse of the above (TODO CHECK)
    '''
    IL = sorted(IL, key = lambda x: x[2], reverse = True)
    chunked = [list(v) for eid, v in groupby(IL, lambda x: x[2])]

    records = []
    for chunk in chunked:
        record = {role : [] for role in role_fields}
        record['eid'] = chunk[0][2]
        record['date'] = chunk[0][3]
        for line in chunk:
            record[line[1]].append(line[0])
        records.append(record)
    
    return(records)

def normalise_counters(counters):
    """Normalise a dictionary of counters inplace."""
    for d in counters.values():
        total = sum(d.values())
        for key in d:
            d[key] /= total
