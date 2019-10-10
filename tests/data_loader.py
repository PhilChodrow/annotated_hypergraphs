import json
from collections import Counter

with open("data/enron_hypergraph_annotated.json") as file:
    DATA = json.load(file)
    ROLE_FIELDS = ["cc", "from", "to"]

    if DATA[0].get("eid") is None:
        for i in range(len(DATA)):
            DATA[i]["eid"] = i + 1

SMALL_TEST_DATA = [
    {"from": [0], "to": [1, 2, 3], "cc": [], "date": 0},
    {"from": [1], "to": [4, 5, 6], "cc": [], "date": 1},
]
