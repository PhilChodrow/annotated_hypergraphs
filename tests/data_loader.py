import json

with open('data/enron_hypergraph_annotated.json') as file:
    DATA = json.load(file)
    ROLE_FIELDS = ['cc', 'from', 'to']

    for i in range(len(DATA)):
        DATA[i]['eid'] = -(i+1)